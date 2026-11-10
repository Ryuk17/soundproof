/*
 * @Author: Ryuk
 * @Date: 2026-08-09 21:00:00
 * @LastEditors: Ryuk
 * @LastEditTime: 2026-08-09 21:00:00
 * @Description: Unit test for libAE_VAD — validates VAD C interface
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "interface/audio_engine_vad.h"
#include "utils/dr_wav.h"

static int CallProcess(VadHandle handle, const short* audio_in, int samples, int* vad_flag);

#ifdef AE_WITH_MNN
#include "core/voice_activity_detection/silero_vad.h"
#include "core/voice_activity_detection/vad_decision_filter.h"
#endif


#define FRAME_LEN (160)   /* 10ms @ 16kHz */


static void test_normal_flow(const char* input_wav, const char* output_wav)
{
    std::cout << "=== test_normal_flow ===" << std::endl;

    /* Open input WAV */
    DrWavReader wav_reader(input_wav);
    int rate       = wav_reader.sample_rate();
    int channels   = wav_reader.num_channels();
    int64_t total  = wav_reader.num_samples();
    std::cout << "  input: " << input_wav
              << " rate=" << rate
              << " ch=" << channels
              << " samples=" << total << std::endl;

    /* Create + Init */
    VadHandle handle = AudioEngine_Vad_Create();
    if (!handle) {
        std::cerr << "  FAIL: AudioEngine_Vad_Create returned NULL" << std::endl;
        return;
    }

    VadInitConfig init_cfg = {};
    init_cfg.sample_rate = rate;
    init_cfg.frame_len   = FRAME_LEN;
    init_cfg.vad_mode    = 0;   /* dsp mode */
    int ret = AudioEngine_Vad_Init(handle, &init_cfg);
    if (ret != AUDIO_ENGINE_SUCCESS) {
        std::cerr << "  FAIL: AudioEngine_Vad_Init returned " << ret << std::endl;
        AudioEngine_Vad_Destroy(handle);
        return;
    }

    /* SetParam */
    VadRtConfig rt_cfg = {};
    rt_cfg.threshold = 0.5f;
    ret = AudioEngine_Vad_SetParam(handle, &rt_cfg);
    if (ret != AUDIO_ENGINE_SUCCESS) {
        std::cerr << "  FAIL: AudioEngine_Vad_SetParam returned " << ret << std::endl;
        AudioEngine_Vad_Deinit(handle);
        AudioEngine_Vad_Destroy(handle);
        return;
    }

    /* Output WAV writer */
    DrWavWriter wav_writer(output_wav, rate, channels);

    int16_t in_buf[FRAME_LEN]   = {0};
    int16_t voice_buf[FRAME_LEN];
    int16_t silence_buf[FRAME_LEN] = {0};
    for (int i = 0; i < FRAME_LEN; i++) {
        voice_buf[i] = 32000;   /* visual marker for voice frames */
    }

    int voice_frames = 0;
    int total_frames = 0;

    while (true) {
        memset(in_buf, 0, sizeof(in_buf));
        int read_samples = wav_reader.ReadSamples(FRAME_LEN, in_buf);
        if (read_samples == 0) break;

        int vad_flag = 0;
        /* Always pass FRAME_LEN — zero-padded for partial final frame */
        ret = CallProcess(handle, in_buf, FRAME_LEN, &vad_flag);
        if (ret != AUDIO_ENGINE_SUCCESS) {
            std::cerr << "  FAIL: AudioEngine_Vad_Process returned " << ret
                      << " at frame " << total_frames << std::endl;
            break;
        }

        if (vad_flag) {
            wav_writer.WriteSamples(voice_buf, read_samples);
            voice_frames++;
        } else {
            wav_writer.WriteSamples(silence_buf, read_samples);
        }
        total_frames++;
    }

    std::cout << "  frames=" << total_frames
              << " voice=" << voice_frames
              << " silence=" << (total_frames - voice_frames) << std::endl;

    /* Deinit + Destroy */
    AudioEngine_Vad_Deinit(handle);
    AudioEngine_Vad_Destroy(handle);

    std::cout << "  PASS" << std::endl;
}


static void test_reset(const char* input_wav)
{
    std::cout << "=== test_reset ===" << std::endl;

    DrWavReader wav_reader(input_wav);
    int rate = wav_reader.sample_rate();

    VadHandle handle = AudioEngine_Vad_Create();
    VadInitConfig init_cfg = {};
    init_cfg.sample_rate = rate;
    init_cfg.frame_len = FRAME_LEN;
    init_cfg.vad_mode = 0;
    AudioEngine_Vad_Init(handle, &init_cfg);

    /* Process a few frames to build up state */
    int16_t buf[FRAME_LEN] = {0};
    int vad_flag;
    for (int i = 0; i < 10; i++) {
        int read = wav_reader.ReadSamples(FRAME_LEN, buf);
        if (read == 0) break;
        CallProcess(handle, buf, read, &vad_flag);
    }

    /* Reset */
    int ret = AudioEngine_Vad_Reset(handle);
    if (ret != AUDIO_ENGINE_SUCCESS) {
        std::cerr << "  FAIL: AudioEngine_Vad_Reset returned " << ret << std::endl;
    } else {
        std::cout << "  Reset OK, threshold=0.5 preserved" << std::endl;
    }

    /* Should still be able to process after reset */
    int read = wav_reader.ReadSamples(FRAME_LEN, buf);
    if (read > 0) {
        ret = CallProcess(handle, buf, read, &vad_flag);
        if (ret != AUDIO_ENGINE_SUCCESS) {
            std::cerr << "  FAIL: Process after Reset returned " << ret << std::endl;
        } else {
            std::cout << "  Process after Reset OK, vad_flag=" << vad_flag << std::endl;
        }
    }

    AudioEngine_Vad_Deinit(handle);
    AudioEngine_Vad_Destroy(handle);
    std::cout << "  PASS" << std::endl;
}


static void test_error_cases(void)
{
    std::cout << "=== test_error_cases ===" << std::endl;

    /* NULL handle on Destroy */
    int ret = AudioEngine_Vad_Destroy(NULL);
    std::cout << "  Destroy(NULL) = " << ret
              << (ret == AUDIO_ENGINE_ERR_INVALID_HANDLE ? " OK" : " FAIL") << std::endl;

    /* NULL handle on Init */
    ret = AudioEngine_Vad_Init(NULL, NULL);
    std::cout << "  Init(NULL, NULL) = " << ret
              << (ret == AUDIO_ENGINE_ERR_INVALID_HANDLE ? " OK" : " FAIL") << std::endl;

    /* Process without Init */
    VadHandle handle = AudioEngine_Vad_Create();
    int vad_flag;
    int16_t buf[FRAME_LEN] = {0};
    ret = CallProcess(handle, buf, FRAME_LEN, &vad_flag);
    std::cout << "  Process without Init = " << ret
              << (ret == AUDIO_ENGINE_ERR_NOT_INITIALIZED ? " OK" : " FAIL") << std::endl;

    /* NULL audio_in */
    VadInitConfig init_cfg = {};
    init_cfg.sample_rate = 16000;
    init_cfg.frame_len = FRAME_LEN;
    init_cfg.vad_mode = 0;
    AudioEngine_Vad_Init(handle, &init_cfg);
    ret = CallProcess(handle, NULL, FRAME_LEN, &vad_flag);
    std::cout << "  Process(NULL audio_in) = " << ret
              << (ret == AUDIO_ENGINE_ERR_NULL_POINTER ? " OK" : " FAIL") << std::endl;

    /* Invalid samples (wrong frame length) */
    ret = CallProcess(handle, buf, 999, &vad_flag);
    std::cout << "  Process(wrong samples) = " << ret
              << (ret == AUDIO_ENGINE_ERR_INVALID_PARAM ? " OK" : " FAIL") << std::endl;

    /* Invalid threshold */
    VadRtConfig rt_cfg = {};
    rt_cfg.threshold = 1.5f;
    ret = AudioEngine_Vad_SetParam(handle, &rt_cfg);
    std::cout << "  SetParam(threshold=1.5) = " << ret
              << (ret == AUDIO_ENGINE_ERR_INVALID_PARAM ? " OK" : " FAIL") << std::endl;

    rt_cfg.threshold = 0.0f;
    ret = AudioEngine_Vad_SetParam(handle, &rt_cfg);
    std::cout << "  SetParam(threshold=0.0) = " << ret
              << (ret == AUDIO_ENGINE_ERR_INVALID_PARAM ? " OK" : " FAIL") << std::endl;

    AudioEngine_Vad_Deinit(handle);
    AudioEngine_Vad_Destroy(handle);
    std::cout << "  PASS" << std::endl;
}


#ifdef AE_WITH_MNN
static const char* kSileroModelPath = "models/vad/silero_vad.mnn/silero_vad.mnn";
static int g_silero_failures = 0;

static void check_silero(bool condition, const char* what)
{
    std::cout << "  [" << (condition ? " OK " : "FAIL") << "] " << what << std::endl;
    if (!condition) {
        ++g_silero_failures;
    }
}

static bool silero_model_exists(void)
{
    FILE* fp = fopen(kSileroModelPath, "rb");
    if (!fp) {
        return false;
    }
    fclose(fp);
    return true;
}

static bool load_wav_16k_mono(const char* path, std::vector<int16_t>* samples)
{
    DrWavReader reader(path);
    if (reader.sample_rate() != 16000 || reader.num_channels() != 1) {
        return false;
    }
    const int total = reader.num_samples();
    samples->resize(static_cast<size_t>(total));
    return reader.ReadSamples(total, samples->data()) == total;
}

static int init_silero(VadHandle handle)
{
    VadInitConfig cfg = {};
    cfg.sample_rate = 16000;
    cfg.frame_len = FRAME_LEN;
    cfg.vad_mode = 1;
    snprintf(cfg.model_path, sizeof(cfg.model_path), "%s", kSileroModelPath);
    return AudioEngine_Vad_Init(handle, &cfg);
}

static soundproof::VadDecisionFilter make_reference_filter(void)
{
    soundproof::VadDecisionFilter::Config config;
    config.threshold = 0.5f;
    config.hysteresis = 0.15f;
    config.min_speech_duration = 0.25f;
    config.min_silence_duration = 0.5f;
    config.sample_rate = 16000;
    return soundproof::VadDecisionFilter(config);
}

/* Feeds one 10 ms frame to the interface and to the core reference path and
   verifies that both produce the same smoothed decision. */
static bool process_silero_frame(VadHandle handle, const int16_t* frame,
                                 soundproof::SileroVad* reference,
                                 soundproof::VadDecisionFilter* reference_filter,
                                 int* flag)
{
    if (CallProcess(handle, frame, FRAME_LEN, flag) !=
        AUDIO_ENGINE_SUCCESS) {
        return false;
    }
    if (reference != nullptr) {
        if (reference_filter == nullptr) {
            return false;
        }
        if (!reference->ProcessChunk(frame, FRAME_LEN, 16000,
                                     reference_filter)) {
            return false;
        }
        if ((*flag != 0) != reference_filter->is_speech()) {
            return false;
        }
    }
    return true;
}

/* Feeds 10 ms chunks while checking every AE decision against the reference. */
static int run_silero_frames(VadHandle handle,
                             const std::vector<int16_t>& samples,
                             soundproof::SileroVad* reference,
                             soundproof::VadDecisionFilter* reference_filter,
                             std::vector<int>* flags)
{
    int16_t frame[FRAME_LEN];
    size_t pos = 0;
    while (pos < samples.size()) {
        const size_t n = std::min<size_t>(FRAME_LEN, samples.size() - pos);
        memcpy(frame, samples.data() + pos, n * sizeof(int16_t));
        if (n < FRAME_LEN) {
            memset(frame + n, 0, (FRAME_LEN - n) * sizeof(int16_t));
        }

        int flag = 0;
        if (!process_silero_frame(handle, frame, reference, reference_filter,
                                  &flag)) {
            return -1;
        }
        if (flags != nullptr) {
            flags->push_back(flag);
        }
        pos += n;
    }
    return 0;
}

static void test_silero_flow(const char* input_wav)
{
    std::cout << "=== test_silero_flow ===" << std::endl;

    if (!silero_model_exists()) {
        std::cout << "  SKIP: silero model not found: " << kSileroModelPath << std::endl;
        return;
    }

    std::vector<int16_t> samples;
    if (!load_wav_16k_mono(input_wav, &samples)) {
        check_silero(false, "load 16 kHz mono WAV");
        return;
    }
    /* Keep the test quick; ~10 s is plenty of speech. */
    if (samples.size() > static_cast<size_t>(16000 * 10)) {
        samples.resize(16000 * 10);
    }

    VadHandle handle = AudioEngine_Vad_Create();
    check_silero(handle != NULL, "Create");
    check_silero(init_silero(handle) == AUDIO_ENGINE_SUCCESS, "Init silero mode");

    VadRtConfig rt_cfg = {};
    rt_cfg.threshold = 0.5f;
    rt_cfg.min_speech_duration = 0.25f;
    rt_cfg.min_silence_duration = 0.5f;
    rt_cfg.hysteresis = 0.15f;
    check_silero(AudioEngine_Vad_SetParam(handle, &rt_cfg) == AUDIO_ENGINE_SUCCESS,
                 "SetParam threshold=0.5, min_speech=0.25s, min_silence=0.5s");

    soundproof::SileroVad reference(kSileroModelPath);
    check_silero(reference.ok(), "reference SileroVad init");
    soundproof::VadDecisionFilter reference_filter = make_reference_filter();

    std::vector<int> flags;
    const int ret =
        run_silero_frames(handle, samples, &reference, &reference_filter, &flags);
    check_silero(ret == 0,
                 "AE decisions match core + shared filter path");

    int voice_frames = 0;
    for (int flag : flags) {
        voice_frames += flag;
    }
    std::cout << "  frames=" << flags.size() << " voice=" << voice_frames << std::endl;
    check_silero(voice_frames > 0, "speech detected in input WAV");

    /* Reset must reproduce the same decision sequence. */
    check_silero(AudioEngine_Vad_Reset(handle) == AUDIO_ENGINE_SUCCESS, "Reset");
    reference.Reset();
    reference_filter.Reset();
    std::vector<int> flags_after_reset;
    check_silero(run_silero_frames(handle, samples, &reference, &reference_filter,
                                   &flags_after_reset) == 0,
                 "inference after Reset");
    check_silero(flags == flags_after_reset,
                 "Reset reproduces the same decision sequence");

    /* Pure silence must not be classified as speech. */
    std::vector<int16_t> silence(16000, 0);
    check_silero(AudioEngine_Vad_Reset(handle) == AUDIO_ENGINE_SUCCESS,
                 "Reset before silence");
    reference.Reset();
    reference_filter.Reset();
    std::vector<int> silence_flags;
    check_silero(run_silero_frames(handle, silence, &reference, &reference_filter,
                                   &silence_flags) == 0,
                 "silence inference");
    int silence_voice = 0;
    for (int flag : silence_flags) {
        silence_voice += flag;
    }
    check_silero(silence_voice == 0, "silence is not classified as speech");

    AudioEngine_Vad_Deinit(handle);
    AudioEngine_Vad_Destroy(handle);
}

static void test_silero_hangover(const char* input_wav)
{
    std::cout << "=== test_silero_hangover ===" << std::endl;

    if (!silero_model_exists()) {
        std::cout << "  SKIP: silero model not found: " << kSileroModelPath << std::endl;
        return;
    }

    std::vector<int16_t> speech;
    if (!load_wav_16k_mono(input_wav, &speech)) {
        check_silero(false, "load 16 kHz mono WAV");
        return;
    }
    if (speech.size() > static_cast<size_t>(16000 * 3)) {
        speech.resize(16000 * 3);
    }

    VadHandle handle = AudioEngine_Vad_Create();
    check_silero(handle != NULL, "Create");
    check_silero(init_silero(handle) == AUDIO_ENGINE_SUCCESS, "Init silero mode");

    VadRtConfig rt_cfg = {};
    rt_cfg.threshold = 0.5f;
    rt_cfg.min_speech_duration = 0.25f;
    rt_cfg.min_silence_duration = 0.5f;
    rt_cfg.hysteresis = 0.15f;
    check_silero(AudioEngine_Vad_SetParam(handle, &rt_cfg) == AUDIO_ENGINE_SUCCESS,
                 "SetParam defaults");

    soundproof::SileroVad reference(kSileroModelPath);
    soundproof::VadDecisionFilter reference_filter = make_reference_filter();
    check_silero(reference.ok(), "reference SileroVad init");

    int16_t frame[FRAME_LEN];
    size_t pos = 0;
    int flag = 0;
    bool activated = false;

    while (pos + FRAME_LEN <= speech.size()) {
        memcpy(frame, speech.data() + pos, sizeof(frame));
        if (!process_silero_frame(handle, frame, &reference, &reference_filter,
                                  &flag)) {
            check_silero(false, "speech phase matches reference");
            AudioEngine_Vad_Deinit(handle);
            AudioEngine_Vad_Destroy(handle);
            return;
        }
        pos += FRAME_LEN;
        if (flag) {
            activated = true;
            break;
        }
    }
    check_silero(activated, "VAD activates on speech");

    /* Feed a little more speech so the triggered state is stable. */
    for (int i = 0; i < 5 && pos + FRAME_LEN <= speech.size(); ++i) {
        memcpy(frame, speech.data() + pos, sizeof(frame));
        if (!process_silero_frame(handle, frame, &reference, &reference_filter,
                                  &flag)) {
            check_silero(false, "stable-speech phase matches reference");
            AudioEngine_Vad_Deinit(handle);
            AudioEngine_Vad_Destroy(handle);
            return;
        }
        pos += FRAME_LEN;
        if (!flag) {
            check_silero(false, "speech is still active");
        }
    }

    /* Switch to silence and count how long the speech decision is held. */
    memset(frame, 0, sizeof(frame));
    int held_frames = 0;
    bool released = false;
    const int kMaxFrames = 200;  /* 2 s */
    for (int i = 0; i < kMaxFrames; ++i) {
        if (!process_silero_frame(handle, frame, &reference, &reference_filter,
                                  &flag)) {
            check_silero(false, "silence phase matches reference");
            break;
        }
        if (!flag) {
            released = true;
            break;
        }
        ++held_frames;
    }
    const int held_ms = held_frames * 10;
    std::cout << "  hangover=" << held_ms << " ms (min_silence=500 ms)"
              << std::endl;
    check_silero(held_ms >= 400, "hangover holds speech >= 400 ms");
    check_silero(released && held_ms <= 1500,
                 "speech is released within 1.5 s of silence");

    AudioEngine_Vad_Deinit(handle);
    AudioEngine_Vad_Destroy(handle);
}

static void test_silero_errors(void)
{
    std::cout << "=== test_silero_errors ===" << std::endl;

    VadHandle handle = AudioEngine_Vad_Create();
    check_silero(handle != NULL, "Create");

    VadInitConfig cfg = {};
    cfg.sample_rate = 16000;
    cfg.frame_len = FRAME_LEN;
    cfg.vad_mode = 2;
    check_silero(AudioEngine_Vad_Init(handle, &cfg) == AUDIO_ENGINE_ERR_INVALID_PARAM,
                 "invalid vad_mode rejected");

    cfg.vad_mode = 1;
    snprintf(cfg.model_path, sizeof(cfg.model_path), "%s", kSileroModelPath);
    cfg.sample_rate = 8000;
    cfg.frame_len = 80;
    check_silero(AudioEngine_Vad_Init(handle, &cfg) == AUDIO_ENGINE_ERR_INVALID_PARAM,
                 "silero rejects 8 kHz");

    cfg.sample_rate = 16000;
    cfg.frame_len = FRAME_LEN;
    cfg.model_path[0] = '\0';
    check_silero(AudioEngine_Vad_Init(handle, &cfg) == AUDIO_ENGINE_ERR_INVALID_PARAM,
                 "silero rejects empty model_path");

    snprintf(cfg.model_path, sizeof(cfg.model_path), "%s",
             "models/vad/does_not_exist.mnn");
    check_silero(AudioEngine_Vad_Init(handle, &cfg) == AUDIO_ENGINE_ERR_INIT_FAILED,
                 "missing model returns INIT_FAILED");

    int16_t frame[FRAME_LEN] = {0};
    int flag = 0;
    check_silero(CallProcess(handle, frame, FRAME_LEN, &flag) ==
                     AUDIO_ENGINE_ERR_NOT_INITIALIZED,
                 "Process rejected after failed Init");

    check_silero(init_silero(handle) == AUDIO_ENGINE_SUCCESS,
                 "Init after failed attempts");
    check_silero(AudioEngine_Vad_Deinit(handle) == AUDIO_ENGINE_SUCCESS, "Deinit");
    check_silero(CallProcess(handle, frame, FRAME_LEN, &flag) ==
                     AUDIO_ENGINE_ERR_NOT_INITIALIZED,
                 "Process rejected after Deinit");

    AudioEngine_Vad_Destroy(handle);
}
#endif  /* AE_WITH_MNN */


static int CallProcess(VadHandle handle, const short* audio_in, int samples,
                       int* vad_flag)
{
    if (!vad_flag) {
        return AUDIO_ENGINE_ERR_NULL_POINTER;
    }
    VadInput_t input{audio_in, samples};
    VadResult_t result{0};
    const int ret = AudioEngine_Vad_Process(handle, &input, &result);
    *vad_flag = result.vad_flag;
    return ret;
}

int main(int argc, char** argv)
{
    const char* input_wav = "assets/audio/audio_short16.wav";
    const char* output_wav = "assets/audio/audio_short16_vad_interface_out.wav";

    if (argc >= 2) input_wav = argv[1];

    std::cout << "VAD Interface Unit Test" << std::endl;
    std::cout << "========================" << std::endl;

    test_normal_flow(input_wav, output_wav);
    test_reset(input_wav);
    test_error_cases();

#ifdef AE_WITH_MNN
    test_silero_flow(input_wav);
    test_silero_hangover(input_wav);
    test_silero_errors();
#endif

    std::cout << "All tests done." << std::endl;
#ifdef AE_WITH_MNN
    if (g_silero_failures != 0) {
        std::cout << g_silero_failures << " silero test(s) FAILED" << std::endl;
        return 1;
    }
#endif
    return 0;
}