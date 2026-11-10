// Tests for libAE_ASR — streaming ASR with transducer / paraformer backends.
//
// The transducer test uses the bundled streaming zipformer model under
// models/asr/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20 and
// assets/audio/audio_short16.wav. Tests are skipped when the assets are missing.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/speech_recognition/hypothesis.h"
#include "core/speech_recognition/paraformer_utils.h"
#include "utils/symbol_table.h"
#include "interface/audio_engine_asr.h"
#include "utils/dr_wav.h"

static int CallProcess(AsrHandle handle, const short* audio_in, int samples, char* text, int text_capacity, int* text_len, int* is_final);

#define FRAME_LEN (160)  /* 10 ms @ 16 kHz */

namespace {

using soundproof::SymbolTable;

const char* kZipformerDir =
    "models/asr/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20";
const char* kZipformerEncoder =
    "models/asr/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "encoder-epoch-99-avg-1.int8.mnn";
const char* kZipformerDecoder =
    "models/asr/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "decoder-epoch-99-avg-1.int8.mnn";
const char* kZipformerJoiner =
    "models/asr/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "joiner-epoch-99-avg-1.int8.mnn";
const char* kZipformerTokens =
    "models/asr/sherpa-mnn-streaming-zipformer-bilingual-zh-en-2023-02-20/"
    "tokens.txt";
const char* kInputWav = "assets/audio/audio_short16.wav";

int g_failures = 0;

void Check(bool condition, const std::string& what)
{
    std::cout << "  [" << (condition ? " OK " : "FAIL") << "] " << what << std::endl;
    if (!condition) {
        ++g_failures;
    }
}

bool FileExists(const std::string& path)
{
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        return false;
    }
    std::fclose(fp);
    return true;
}

bool TransducerAssetsAvailable()
{
    const char* files[] = {kZipformerEncoder, kZipformerDecoder, kZipformerJoiner,
                           kZipformerTokens, kInputWav};
    for (const char* file : files) {
        if (!FileExists(file)) {
            std::cout << "  SKIP: missing " << file << std::endl;
            return false;
        }
    }
    return true;
}

bool LoadWav(const char* path, std::vector<int16_t>* samples)
{
    DrWavReader reader(path);
    if (reader.sample_rate() != 16000 || reader.num_channels() != 1) {
        return false;
    }
    const int total = reader.num_samples();
    samples->resize(static_cast<size_t>(total));
    return reader.ReadSamples(total, samples->data()) == total;
}

void FillZipformerConfig(AsrInitConfig* config, const char* model_type)
{
    std::memset(config, 0, sizeof(*config));
    config->sample_rate = 16000;
    config->frame_len = FRAME_LEN;
    config->num_threads = 1;
    config->enable_endpoint = 1;
    std::snprintf(config->model_type, sizeof(config->model_type), "%s",
                  model_type == nullptr ? "" : model_type);
    std::snprintf(config->encoder_path, sizeof(config->encoder_path), "%s",
                  kZipformerEncoder);
    std::snprintf(config->decoder_path, sizeof(config->decoder_path), "%s",
                  kZipformerDecoder);
    std::snprintf(config->joiner_path, sizeof(config->joiner_path), "%s",
                  kZipformerJoiner);
    std::snprintf(config->tokens_path, sizeof(config->tokens_path), "%s",
                  kZipformerTokens);
}

bool InitHandle(AsrHandle handle, const char* model_type)
{
    AsrInitConfig config;
    FillZipformerConfig(&config, model_type);
    const int ret = AudioEngine_Asr_Init(handle, &config);
    if (ret != AUDIO_ENGINE_SUCCESS) {
        std::cerr << "  FAIL: Init returned " << ret << std::endl;
        return false;
    }
    return true;
}

struct RunResult {
    std::string final_text;
    int final_count = 0;
    std::vector<std::string> partials;
};

// Feeds |samples| in 10 ms frames; stops early when |stop_on_final| and an
// endpoint is reached.
RunResult RunAudio(AsrHandle handle, const std::vector<int16_t>& samples,
                   bool stop_on_final)
{
    RunResult result;
    char text[8192];
    int16_t frame[FRAME_LEN];
    std::string last_text;

    size_t pos = 0;
    while (pos < samples.size()) {
        const size_t n = std::min<size_t>(FRAME_LEN, samples.size() - pos);
        std::memcpy(frame, samples.data() + pos, n * sizeof(int16_t));
        if (n < FRAME_LEN) {
            std::memset(frame + n, 0, (FRAME_LEN - n) * sizeof(int16_t));
        }

        int text_len = 0;
        int is_final = 0;
        const int ret = CallProcess(handle, frame, FRAME_LEN, text,
                                                sizeof(text), &text_len,
                                                &is_final);
        if (ret != AUDIO_ENGINE_SUCCESS) {
            std::cerr << "  FAIL: Process returned " << ret << std::endl;
            break;
        }
        std::string current(text);
        if (current != last_text) {
            result.partials.push_back(current);
            last_text = current;
        }
        if (is_final) {
            ++result.final_count;
            result.final_text = current;
            if (stop_on_final) {
                break;
            }
        }
        pos += n;
    }
    return result;
}

void TestZipformer()
{
    std::cout << "=== test_zipformer ===" << std::endl;

    std::vector<int16_t> samples;
    if (!LoadWav(kInputWav, &samples)) {
        Check(false, "load 16 kHz mono WAV");
        return;
    }
    /* Keep the test quick: 8 s of speech + 3 s of silence. */
    if (samples.size() > static_cast<size_t>(16000 * 8)) {
        samples.resize(16000 * 8);
    }
    std::vector<int16_t> silence(16000 * 3, 0);

    AsrHandle handle = AudioEngine_Asr_Create();
    Check(handle != NULL, "Create");
    Check(InitHandle(handle, ""), "Init zipformer (auto-detect)");

    RunResult speech = RunAudio(handle, samples, false);
    std::cout << "  partial text: " << speech.partials.back() << std::endl;
    Check(!speech.partials.empty() && !speech.partials.back().empty(),
          "partial text is non-empty after speech");

    RunResult tail = RunAudio(handle, silence, true);
    std::cout << "  final text: " << tail.final_text << std::endl;
    Check(tail.final_count >= 1, "endpoint reached after trailing silence");
    Check(!tail.final_text.empty(), "final text is non-empty");

    AudioEngine_Asr_Deinit(handle);
    AudioEngine_Asr_Destroy(handle);
}

void TestResetDeterminism()
{
    std::cout << "=== test_reset ===" << std::endl;

    std::vector<int16_t> samples;
    if (!LoadWav(kInputWav, &samples)) {
        Check(false, "load WAV");
        return;
    }
    if (samples.size() > static_cast<size_t>(16000 * 4)) {
        samples.resize(16000 * 4);
    }

    AsrHandle handle = AudioEngine_Asr_Create();
    if (!handle || !InitHandle(handle, "zipformer")) {
        Check(false, "Init for reset test");
        if (handle) {
            AudioEngine_Asr_Destroy(handle);
        }
        return;
    }

    const RunResult first = RunAudio(handle, samples, false);
    Check(AudioEngine_Asr_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    const RunResult second = RunAudio(handle, samples, false);
    Check(first.partials == second.partials,
          "Reset reproduces the same partial text sequence");

    AudioEngine_Asr_Deinit(handle);
    AudioEngine_Asr_Destroy(handle);
}

void TestErrorCases()
{
    std::cout << "=== test_error_cases ===" << std::endl;

    Check(AudioEngine_Asr_Destroy(NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Destroy(NULL)");
    Check(AudioEngine_Asr_Init(NULL, NULL) == AUDIO_ENGINE_ERR_INVALID_HANDLE,
          "Init(NULL, NULL)");

    AsrHandle handle = AudioEngine_Asr_Create();
    int16_t frame[FRAME_LEN] = {0};
    char text[64] = {0};
    int text_len = 0;
    int is_final = 0;
    Check(CallProcess(handle, frame, FRAME_LEN, text, sizeof(text),
                                  &text_len, &is_final) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process before Init");

    AsrInitConfig config;
    FillZipformerConfig(&config, "");
    config.sample_rate = 8000;
    config.frame_len = 80;
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "8 kHz is rejected");

    FillZipformerConfig(&config, "");
    config.frame_len = 100;
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "non-10 ms frame is rejected");

    FillZipformerConfig(&config, "");
    config.encoder_path[0] = '\0';
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "missing encoder path is rejected");

    FillZipformerConfig(&config, "zipformer");
    config.joiner_path[0] = '\0';
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INIT_FAILED,
          "transducer without joiner fails cleanly");

    FillZipformerConfig(&config, "not-a-model");
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INIT_FAILED,
          "unknown model type is rejected");

    FillZipformerConfig(&config, "");
    config.decoding_method = 2;
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "invalid decoding_method is rejected");

    FillZipformerConfig(&config, "");
    std::snprintf(config.lm_model_path, sizeof(config.lm_model_path), "%s",
                  "models/asr/dummy-lm.mnn");
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "LM path with greedy search is rejected");

    FillZipformerConfig(&config, "");
    config.enable_punctuation = 1;
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "punctuation without model/vocab is rejected");

    FillZipformerConfig(&config, "");
    config.decoding_method = 1;
    std::snprintf(config.lm_model_path, sizeof(config.lm_model_path), "%s",
                  "models/asr/does_not_exist.mnn");
    Check(AudioEngine_Asr_Init(handle, &config) ==
              AUDIO_ENGINE_ERR_INIT_FAILED,
          "missing LM model returns INIT_FAILED");

    Check(InitHandle(handle, ""), "Init with valid config after failures");

    AsrRtConfig rt = {};
    rt.blank_penalty = -1.0f;
    Check(AudioEngine_Asr_SetParam(handle, &rt) ==
              AUDIO_ENGINE_ERR_INVALID_PARAM,
          "negative blank_penalty is rejected");

    rt.blank_penalty = 0.5f;
    Check(AudioEngine_Asr_SetParam(handle, &rt) == AUDIO_ENGINE_SUCCESS,
          "SetParam(blank_penalty=0.5)");

    /* Force a non-empty text first, then pass a tiny buffer. */
    std::vector<int16_t> samples;
    if (LoadWav(kInputWav, &samples)) {
        if (samples.size() > static_cast<size_t>(16000 * 4)) {
            samples.resize(16000 * 4);
        }
        char small[4] = {0};
        bool saw_too_small = false;
        for (size_t pos = 0; pos + FRAME_LEN <= samples.size(); pos += FRAME_LEN) {
            int len = 0;
            int final = 0;
            const int ret = CallProcess(
                handle, samples.data() + pos, FRAME_LEN, small, sizeof(small),
                &len, &final);
            if (ret == AUDIO_ENGINE_ERR_INVALID_PARAM && len > (int)sizeof(small)) {
                saw_too_small = true;
                break;
            }
        }
        Check(saw_too_small, "too-small text buffer reports required size");
    }

    Check(AudioEngine_Asr_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    Check(CallProcess(handle, frame, FRAME_LEN, text, sizeof(text),
                                  &text_len, &is_final) ==
              AUDIO_ENGINE_ERR_NOT_INITIALIZED,
          "Process after Deinit");

    AudioEngine_Asr_Destroy(handle);
}

void TestHypotheses()
{
    std::cout << "=== test_hypotheses ===" << std::endl;

    soundproof::Hypothesis h1;
    h1.ys = {-1, 0, 5};
    h1.log_prob = -1.0;
    soundproof::Hypotheses hyps;
    hyps.Add(h1);

    soundproof::Hypothesis h2 = h1;
    h2.log_prob = -2.0;
    hyps.Add(h2);
    Check(hyps.Size() == 1, "same token sequence is merged");
    const soundproof::Hypothesis merged = hyps.GetMostProbable(false);
    const double expected = std::log(std::exp(-1.0) + std::exp(-2.0));
    Check(std::abs(merged.log_prob - expected) < 1e-6,
          "merged log prob uses log-sum-exp");

    h2.ys = {-1, 0, 6};
    h2.log_prob = -0.5;
    hyps.Add(h2);
    Check(hyps.Size() == 2, "different token sequences stay separate");
    Check(hyps.GetMostProbable(false).ys.back() == 6,
          "most probable hypothesis selected");
}

void TestParaformerUtils()
{
    std::cout << "=== test_paraformer_utils ===" << std::endl;

    /* LFR: 61 frames of 80 dims -> 10 frames of 560 dims (7/6). */
    std::vector<float> input(static_cast<size_t>(61) * 80);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i);
    }
    const std::vector<float> lfr = soundproof::ApplyLfr(input.data(), 61, 80, 7, 6);
    Check(lfr.size() == static_cast<size_t>(10) * 560, "LFR output shape");
    Check(lfr.size() >= 560 && lfr[0] == 0.0f &&
              lfr[560] == static_cast<float>(6 * 80),
          "LFR window/shift content");

    /* CMVN. */
    std::vector<float> frames = {1.0f, 2.0f};
    soundproof::ApplyCMVN(&frames, {1.0f, 1.0f}, {2.0f, 2.0f});
    Check(frames.size() == 2 && frames[0] == 4.0f && frames[1] == 6.0f,
          "CMVN applies (x + mean) * inv_std");

    /* Positional encoding adds finite sin/cos values. */
    std::vector<float> pe(4, 0.0f);
    soundproof::ApplyPositionalEncoding(&pe, 4, 0);
    bool finite = true;
    for (float v : pe) {
        finite = finite && std::isfinite(v);
    }
    Check(finite && (pe[0] != 0.0f || pe[2] != 0.0f),
          "positional encoding is applied");

    /* Paraformer @@ merge / CJK spacing. */
    SymbolTable symbols;
    Check(symbols.LoadFromString("ab@@ 0\ncd 1\n中文 2\n"), "symbol table");
    const std::string text =
        soundproof::ParaformerTextFromTokens({0, 1, 2}, symbols);
    Check(text.find("abcd") != std::string::npos &&
              text.find("中文") != std::string::npos,
          "@@ merge and CJK text");
}

std::string FindFileWithKey(const std::string& root,
                            const std::vector<std::string>& keys,
                            bool require_mnn)
{
    namespace fs = std::filesystem;
    if (!fs::exists(root)) {
        return {};
    }
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (require_mnn && entry.path().extension() != ".mnn") {
            continue;
        }
        std::string name = entry.path().filename().string();
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) {
                           return static_cast<char>(std::tolower(c));
                       });
        for (const std::string& key : keys) {
            if (name.find(key) != std::string::npos) {
                return entry.path().string();
            }
        }
    }
    return {};
}

void TestLmIfAvailable()
{
    std::cout << "=== test_lm ===" << std::endl;

    const std::string lm_path =
        FindFileWithKey("models/asr", {"lm", "rnn"}, true);
    if (lm_path.empty()) {
        std::cout << "  SKIP: no LM model under models/asr" << std::endl;
        return;
    }

    std::vector<int16_t> samples;
    if (!LoadWav(kInputWav, &samples)) {
        Check(false, "load WAV for LM test");
        return;
    }
    if (samples.size() > static_cast<size_t>(16000 * 3)) {
        samples.resize(16000 * 3);
    }

    AsrHandle handle = AudioEngine_Asr_Create();
    AsrInitConfig config;
    FillZipformerConfig(&config, "");
    config.decoding_method = 1;
    config.lm_shallow_fusion = 1;
    config.lm_scale = 0.5f;
    std::snprintf(config.lm_model_path, sizeof(config.lm_model_path), "%s",
                  lm_path.c_str());
    const int ret = AudioEngine_Asr_Init(handle, &config);
    Check(ret == AUDIO_ENGINE_SUCCESS, "Init modified_beam_search + LM");
    if (ret == AUDIO_ENGINE_SUCCESS) {
        const RunResult result = RunAudio(handle, samples, false);
        Check(!result.partials.empty() && !result.partials.back().empty(),
              "beam search + LM produces text");
    }
    AudioEngine_Asr_Deinit(handle);
    AudioEngine_Asr_Destroy(handle);
}

void TestPunctuationIfAvailable()
{
    std::cout << "=== test_punctuation ===" << std::endl;

    const std::string punct_model =
        FindFileWithKey("models/asr", {"punct"}, true);
    const std::string punct_vocab =
        FindFileWithKey("models/asr", {"vocab"}, false);
    if (punct_model.empty() || punct_vocab.empty()) {
        std::cout << "  SKIP: no punctuation model/vocab under models/asr"
                  << std::endl;
        return;
    }

    std::vector<int16_t> samples;
    if (!LoadWav(kInputWav, &samples)) {
        Check(false, "load WAV for punctuation test");
        return;
    }
    if (samples.size() > static_cast<size_t>(16000 * 3)) {
        samples.resize(16000 * 3);
    }
    std::vector<int16_t> silence(16000 * 3, 0);

    AsrHandle handle = AudioEngine_Asr_Create();
    AsrInitConfig config;
    FillZipformerConfig(&config, "");
    config.enable_punctuation = 1;
    std::snprintf(config.punctuation_model_path,
                  sizeof(config.punctuation_model_path), "%s",
                  punct_model.c_str());
    std::snprintf(config.punctuation_bpe_vocab_path,
                  sizeof(config.punctuation_bpe_vocab_path), "%s",
                  punct_vocab.c_str());
    const int ret = AudioEngine_Asr_Init(handle, &config);
    Check(ret == AUDIO_ENGINE_SUCCESS, "Init punctuation model");
    if (ret == AUDIO_ENGINE_SUCCESS) {
        RunAudio(handle, samples, false);
        const RunResult tail = RunAudio(handle, silence, true);
        std::cout << "  punctuated final text: " << tail.final_text
                  << std::endl;
        Check(!tail.final_text.empty(), "punctuated final text is non-empty");
    }
    AudioEngine_Asr_Deinit(handle);
    AudioEngine_Asr_Destroy(handle);
}

void TestParaformerBackendIfAvailable()
{
    std::cout << "=== test_paraformer_backend ===" << std::endl;

    namespace fs = std::filesystem;
    if (!fs::exists("models/asr")) {
        std::cout << "  SKIP: models/asr does not exist" << std::endl;
        return;
    }
    std::string model_dir;
    for (const auto& entry : fs::directory_iterator("models/asr")) {
        if (!entry.is_directory()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("paraformer") != std::string::npos) {
            model_dir = entry.path().string();
            break;
        }
    }
    if (model_dir.empty()) {
        std::cout << "  SKIP: no paraformer model under models/asr" << std::endl;
        return;
    }

    std::string encoder;
    std::string decoder;
    std::string tokens;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        const std::string path = entry.path().string();
        const std::string name = entry.path().filename().string();
        if (name.find("encoder") != std::string::npos &&
            path.size() >= 4 && path.substr(path.size() - 4) == ".mnn") {
            encoder = path;
        } else if (name.find("decoder") != std::string::npos &&
                   path.size() >= 4 && path.substr(path.size() - 4) == ".mnn") {
            decoder = path;
        } else if (name == "tokens.txt") {
            tokens = path;
        }
    }
    if (encoder.empty() || decoder.empty() || tokens.empty()) {
        std::cout << "  SKIP: incomplete paraformer model files" << std::endl;
        return;
    }

    AsrInitConfig config;
    std::memset(&config, 0, sizeof(config));
    config.sample_rate = 16000;
    config.frame_len = FRAME_LEN;
    config.num_threads = 1;
    config.enable_endpoint = 1;
    std::snprintf(config.model_type, sizeof(config.model_type), "paraformer");
    std::snprintf(config.encoder_path, sizeof(config.encoder_path), "%s",
                  encoder.c_str());
    std::snprintf(config.decoder_path, sizeof(config.decoder_path), "%s",
                  decoder.c_str());
    std::snprintf(config.tokens_path, sizeof(config.tokens_path), "%s",
                  tokens.c_str());

    AsrHandle handle = AudioEngine_Asr_Create();
    const int ret = AudioEngine_Asr_Init(handle, &config);
    Check(ret == AUDIO_ENGINE_SUCCESS, "Init paraformer backend");
    if (ret == AUDIO_ENGINE_SUCCESS) {
        std::vector<int16_t> samples;
        if (LoadWav(kInputWav, &samples)) {
            if (samples.size() > static_cast<size_t>(16000 * 8)) {
                samples.resize(16000 * 8);
            }
            const RunResult result = RunAudio(handle, samples, false);
            std::cout << "  partial text: "
                      << (result.partials.empty() ? "" : result.partials.back())
                      << std::endl;
            Check(!result.partials.empty() && !result.partials.back().empty(),
                  "paraformer produces non-empty text");
        }
    }
    AudioEngine_Asr_Deinit(handle);
    AudioEngine_Asr_Destroy(handle);
}

}  // namespace

static int CallProcess(AsrHandle handle, const short* audio_in, int samples,
                       char* text, int text_capacity, int* text_len,
                       int* is_final)
{
    if (!text || !text_len || !is_final) {
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    AsrInput_t input{audio_in, samples};
    AsrResult_t result{text, text_capacity, 0, 0};
    const int ret = AudioEngine_Asr_Process(handle, &input, &result);
    *text_len = result.text_len;
    *is_final = result.is_final;
    return ret;
}

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    std::cout << "ASR Interface Unit Test" << std::endl;
    std::cout << "=======================" << std::endl;

    TestParaformerUtils();

    if (!TransducerAssetsAvailable()) {
        std::cout << "Transducer tests skipped (assets missing)." << std::endl;
    } else {
        TestZipformer();
        TestResetDeterminism();
        TestErrorCases();
        TestLmIfAvailable();
        TestPunctuationIfAvailable();
    }
    TestHypotheses();
    TestParaformerBackendIfAvailable();

    std::cout << "All tests done." << std::endl;
    if (g_failures != 0) {
        std::cout << g_failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    return 0;
}
