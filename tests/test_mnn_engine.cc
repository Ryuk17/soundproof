// Tests for the internal MNN inference engine.
//
// The engine is exercised with the Silero VAD model (multi-input, multi-output
// with stream state) at models/vad/silero_vad.mnn/silero_vad.mnn:
//   inputs : input, state, sr
//   outputs: output, stateN
//
// Covers: model introspection, stateful inference, Init/Deinit lifecycle,
// shared-engine multi-thread inference, negative cases, and concurrent
// multi-process inference (the test binary re-launches itself as a child).

#include "mnn_engine.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using soundproof::mnn_engine::BytesPerElement;
using soundproof::mnn_engine::DataType;
using soundproof::mnn_engine::Engine;
using soundproof::mnn_engine::Tensor;

constexpr const char kDefaultModelPath[] =
    "models/vad/silero_vad.mnn/silero_vad.mnn";
constexpr int kSileroWindowSamples = 512;
constexpr int kSileroSampleRate = 16000;

int g_failures = 0;

bool Expect(bool condition, const char* what) {
  std::printf("[%s] %s\n", condition ? " OK " : "FAIL", what);
  if (!condition) {
    ++g_failures;
  }
  return condition;
}

const Tensor* FindTensor(const std::vector<Tensor>& tensors,
                         const std::string& name) {
  for (const Tensor& tensor : tensors) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

size_t ElementCount(const std::vector<int>& dim) {
  size_t count = 1;
  for (int d : dim) {
    if (d > 0) {
      count *= static_cast<size_t>(d);
    }
  }
  return count;
}

bool FillScalar(Tensor* tensor, int64_t value) {
  if (tensor->data.size() < static_cast<size_t>(BytesPerElement(tensor->type))) {
    return false;
  }
  switch (tensor->type) {
    case DataType::kInt32:
      *reinterpret_cast<int32_t*>(tensor->data.data()) =
          static_cast<int32_t>(value);
      return true;
    case DataType::kInt64:
      *reinterpret_cast<int64_t*>(tensor->data.data()) = value;
      return true;
    case DataType::kFloat32:
      *reinterpret_cast<float*>(tensor->data.data()) =
          static_cast<float>(value);
      return true;
    default:
      return false;
  }
}

void PrintEngineInfo(const Engine& engine) {
  const std::map<std::string, std::string> metadata = engine.metadata();
  std::printf("model metadata (%d entries):\n",
              static_cast<int>(metadata.size()));
  for (const auto& kv : metadata) {
    std::printf("  %s=%s\n", kv.first.c_str(), kv.second.c_str());
  }
  for (const Tensor& info : engine.inputs()) {
    std::printf("  input  %-8s type=%d dim=[",
                info.name.c_str(), static_cast<int>(info.type));
    for (int i = 0; i < static_cast<int>(info.dim.size()); ++i) {
      std::printf("%s%d", i == 0 ? "" : ",", info.dim[i]);
    }
    std::printf("]\n");
  }
  for (const std::string& name : engine.output_names()) {
    std::printf("  output %s\n", name.c_str());
  }
}

struct SileroState {
  std::vector<Tensor> inputs;  // input, state, sr
  size_t state_index = 0;
};

bool BuildSileroInputs(const std::vector<Tensor>& model_inputs,
                       SileroState* state) {
  const Tensor* x_info = FindTensor(model_inputs, "input");
  const Tensor* state_info = FindTensor(model_inputs, "state");
  const Tensor* sr_info = FindTensor(model_inputs, "sr");
  if (x_info == nullptr || state_info == nullptr || sr_info == nullptr) {
    std::printf("FAIL: unexpected Silero VAD input names\n");
    return false;
  }
  if (x_info->type != DataType::kFloat32 ||
      state_info->type != DataType::kFloat32) {
    std::printf("FAIL: unexpected Silero VAD input types\n");
    return false;
  }

  Tensor x;
  x.name = "input";
  x.type = x_info->type;
  x.dim = x_info->dim;
  for (int& d : x.dim) {
    if (d <= 0) {
      d = kSileroWindowSamples;  // dynamic window dimension
    }
  }
  x.data.assign(ElementCount(x.dim) * BytesPerElement(x.type), 0);

  Tensor state_tensor;
  state_tensor.name = "state";
  state_tensor.type = state_info->type;
  state_tensor.dim = state_info->dim;
  for (int& d : state_tensor.dim) {
    if (d <= 0) {
      d = 1;  // dynamic batch dimension of the LSTM state
    }
  }
  state_tensor.data.assign(
      ElementCount(state_tensor.dim) * BytesPerElement(state_tensor.type), 0);

  Tensor sr;
  sr.name = "sr";
  sr.type = sr_info->type;
  sr.dim = sr_info->dim;
  sr.data.resize(ElementCount(sr.dim) * BytesPerElement(sr.type));
  if (!FillScalar(&sr, kSileroSampleRate)) {
    std::printf("FAIL: unsupported Silero VAD sample-rate input type\n");
    return false;
  }

  state->inputs = {std::move(x), std::move(state_tensor), std::move(sr)};
  state->state_index = 1;
  return true;
}

bool RunSileroFrame(Engine* engine, SileroState* state, float* probability) {
  std::vector<Tensor> outputs;
  if (!engine->Forward(state->inputs, &outputs)) {
    return false;
  }
  const Tensor* prob = FindTensor(outputs, "output");
  const Tensor* state_out = FindTensor(outputs, "stateN");
  if (prob == nullptr || state_out == nullptr) {
    std::printf("FAIL: missing Silero VAD outputs\n");
    return false;
  }
  if (prob->type != DataType::kFloat32 || prob->data.size() < sizeof(float)) {
    std::printf("FAIL: unexpected Silero VAD probability tensor\n");
    return false;
  }
  *probability = *reinterpret_cast<const float*>(prob->data.data());

  // Feed the returned state back as the next frame's input.
  Tensor& state_in = state->inputs[state->state_index];
  state_in.type = state_out->type;
  state_in.dim = state_out->dim;
  state_in.data = state_out->data;
  return true;
}

bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    return false;
  }
  const std::streamsize size = file.tellg();
  if (size <= 0) {
    return false;
  }
  out->resize(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  return file.read(reinterpret_cast<char*>(out->data()), size).good();
}

bool TestSingleProcess(const std::string& model_path) {
  std::unique_ptr<Engine> engine = Engine::CreateFromFile(model_path);
  if (!Expect(engine != nullptr, "CreateFromFile(silero_vad.mnn)")) {
    return false;
  }
  PrintEngineInfo(*engine);

  const std::vector<Tensor> model_inputs = engine->inputs();
  Expect(model_inputs.size() == 3, "silero v5 exposes 3 inputs");
  Expect(engine->output_names().size() == 2, "silero v5 exposes 2 outputs");
  bool descriptors_empty = true;
  for (const Tensor& input : model_inputs) {
    descriptors_empty = descriptors_empty && input.data.empty();
  }
  Expect(descriptors_empty, "model input descriptors carry no data");

  SileroState state;
  if (!Expect(BuildSileroInputs(model_inputs, &state), "build silero inputs")) {
    return false;
  }

  bool ok = true;
  for (int frame = 0; frame < 5; ++frame) {
    float probability = -1.0f;
    if (!Expect(RunSileroFrame(engine.get(), &state, &probability),
                "stateful frame inference")) {
      ok = false;
      break;
    }
    ok &= Expect(std::isfinite(probability) && probability >= 0.0f &&
                     probability <= 1.0f,
                 "probability is finite and within [0,1]");
  }

  std::vector<uint8_t> model_data;
  if (Expect(ReadFile(model_path, &model_data), "read model file for memory load")) {
    std::unique_ptr<Engine> memory_engine =
        Engine::CreateFromMemory(model_data.data(), model_data.size());
    ok &= Expect(memory_engine != nullptr &&
                     memory_engine->inputs().size() == 3,
                 "CreateFromMemory(silero_vad.mnn)");
    if (memory_engine != nullptr) {
      SileroState memory_state;
      float probability = -1.0f;
      ok &= Expect(BuildSileroInputs(memory_engine->inputs(), &memory_state) &&
                       RunSileroFrame(memory_engine.get(), &memory_state,
                                      &probability),
                   "inference from in-memory model");
    }
  }
  return ok;
}

bool TestDeinit(const std::string& model_path) {
  std::unique_ptr<Engine> engine = Engine::CreateFromFile(model_path);
  if (!Expect(engine != nullptr && engine->initialized(),
              "CreateFromFile for Deinit test")) {
    return false;
  }

  SileroState state;
  if (!Expect(BuildSileroInputs(engine->inputs(), &state),
              "build inputs before Deinit")) {
    return false;
  }
  float probability = -1.0f;
  bool ok = Expect(RunSileroFrame(engine.get(), &state, &probability),
                   "forward before Deinit");

  engine->Deinit();
  ok &= Expect(!engine->initialized(), "initialized() is false after Deinit");

  std::vector<Tensor> outputs;
  ok &= Expect(!engine->Forward(state.inputs, &outputs),
               "Forward is rejected after Deinit");
  ok &= Expect(engine->inputs().empty(), "input descriptors cleared by Deinit");
  ok &= Expect(engine->output_names().empty(), "output names cleared by Deinit");

  ok &= Expect(engine->Init(model_path), "Init after Deinit");
  SileroState new_state;
  ok &= Expect(BuildSileroInputs(engine->inputs(), &new_state) &&
                   RunSileroFrame(engine.get(), &new_state, &probability),
               "forward after re-Init");

  engine->Deinit();
  engine->Deinit();  // idempotent
  ok &= Expect(!engine->initialized(), "repeated Deinit is safe");
  return ok;
}

bool TestMultiThread(const std::string& model_path) {
  std::unique_ptr<Engine> engine = Engine::CreateFromFile(model_path);
  if (!Expect(engine != nullptr, "CreateFromFile for multi-thread test")) {
    return false;
  }

  const std::vector<Tensor> model_inputs = engine->inputs();
  constexpr int kThreads = 4;
  constexpr int kIterations = 20;
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&engine, &model_inputs, &failures]() {
      SileroState state;
      if (!BuildSileroInputs(model_inputs, &state)) {
        failures.fetch_add(1);
        return;
      }
      for (int i = 0; i < kIterations; ++i) {
        float probability = -1.0f;
        if (!RunSileroFrame(engine.get(), &state, &probability) ||
            !std::isfinite(probability) || probability < 0.0f ||
            probability > 1.0f) {
          failures.fetch_add(1);
          return;
        }
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  return Expect(failures.load() == 0,
                "shared Engine inference from 4 threads x 20 frames");
}

bool TestNegative(const std::string& model_path) {
  Expect(Engine::CreateFromFile("models/vad/does_not_exist.mnn") == nullptr,
         "missing model file returns nullptr");

  std::unique_ptr<Engine> engine = Engine::CreateFromFile(model_path);
  if (!Expect(engine != nullptr, "CreateFromFile for negative tests")) {
    return false;
  }
  SileroState state;
  if (!Expect(BuildSileroInputs(engine->inputs(), &state),
              "build inputs for negative tests")) {
    return false;
  }

  std::vector<Tensor> outputs;
  std::vector<Tensor> bad = state.inputs;
  bad[0].name = "not_an_input";
  Expect(!engine->Forward(bad, &outputs), "unknown input name is rejected");

  bad = state.inputs;
  bad.pop_back();
  Expect(!engine->Forward(bad, &outputs), "missing input is rejected");

  bad = state.inputs;
  bad[2].type = DataType::kFloat32;
  Expect(!engine->Forward(bad, &outputs), "wrong input type is rejected");

  Expect(!engine->Forward(state.inputs, nullptr),
         "null output pointer is rejected");

  float probability = -1.0f;
  return Expect(RunSileroFrame(engine.get(), &state, &probability),
                "engine is still usable after rejected runs");
}

int RunChild(const std::string& model_path, int iterations) {
  std::unique_ptr<Engine> engine = Engine::CreateFromFile(model_path);
  if (engine == nullptr) {
    return 2;
  }
  SileroState state;
  if (!BuildSileroInputs(engine->inputs(), &state)) {
    return 3;
  }
  for (int i = 0; i < iterations; ++i) {
    float probability = -1.0f;
    if (!RunSileroFrame(engine.get(), &state, &probability) ||
        !std::isfinite(probability)) {
      return 4;
    }
  }
  return 0;
}

#ifdef _WIN32
std::wstring Utf8ToWide(const std::string& s) {
  if (s.empty()) {
    return std::wstring();
  }
  const int len = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
  if (len <= 1) {
    return std::wstring();
  }
  std::wstring wide(static_cast<size_t>(len), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, &wide[0], len);
  wide.resize(static_cast<size_t>(len - 1));
  return wide;
}

bool TestMultiProcess(const std::string& model_path) {
  constexpr int kChildren = 3;
  constexpr int kIterations = 20;

  wchar_t exe_path[MAX_PATH] = {};
  if (GetModuleFileNameW(nullptr, exe_path, MAX_PATH) == 0) {
    return Expect(false, "GetModuleFileNameW");
  }
  const std::wstring command = L"\"" + std::wstring(exe_path) + L"\" --child \"" +
                               Utf8ToWide(model_path) + L"\" " +
                               std::to_wstring(kIterations);

  std::vector<HANDLE> processes;
  processes.reserve(kChildren);
  bool spawned = true;
  for (int i = 0; i < kChildren; ++i) {
    STARTUPINFOW startup_info{};
    startup_info.cb = sizeof(startup_info);
    PROCESS_INFORMATION process_info{};
    std::wstring mutable_command = command;  // CreateProcessW may modify it
    if (!CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup_info,
                        &process_info)) {
      std::printf("FAIL: CreateProcessW (error %lu)\n",
                  static_cast<unsigned long>(GetLastError()));
      spawned = false;
      break;
    }
    CloseHandle(process_info.hThread);
    processes.push_back(process_info.hProcess);
  }

  bool ok = spawned;
  if (spawned) {
    const DWORD wait = WaitForMultipleObjects(
        static_cast<DWORD>(processes.size()), processes.data(), TRUE, 120000);
    if (wait == WAIT_FAILED || wait == WAIT_TIMEOUT) {
      std::printf("FAIL: waiting for child processes\n");
      ok = false;
    } else {
      for (HANDLE process : processes) {
        DWORD exit_code = 1;
        if (!GetExitCodeProcess(process, &exit_code) || exit_code != 0) {
          std::printf("FAIL: child process exit code %lu\n",
                      static_cast<unsigned long>(exit_code));
          ok = false;
        }
      }
    }
  }
  for (HANDLE process : processes) {
    CloseHandle(process);
  }
  return Expect(ok, "3 concurrent child processes load and run the model");
}
#else
bool TestMultiProcess(const std::string& model_path, const char* argv0) {
  constexpr int kChildren = 3;
  constexpr int kIterations = 20;
  const std::string iterations = std::to_string(kIterations);

  std::string exe_path = argv0 != nullptr ? argv0 : "";
  char resolved[4096] = {};
  const ssize_t len = ::readlink("/proc/self/exe", resolved, sizeof(resolved) - 1);
  if (len > 0) {
    exe_path.assign(resolved, static_cast<size_t>(len));
  }
  if (exe_path.empty()) {
    return Expect(false, "resolve test executable path");
  }

  std::vector<pid_t> children;
  bool spawned = true;
  for (int i = 0; i < kChildren; ++i) {
    const pid_t pid = ::fork();
    if (pid < 0) {
      spawned = false;
      break;
    }
    if (pid == 0) {
      ::execl(exe_path.c_str(), exe_path.c_str(), "--child", model_path.c_str(),
              iterations.c_str(), static_cast<char*>(nullptr));
      ::_exit(127);
    }
    children.push_back(pid);
  }

  bool ok = spawned;
  for (pid_t pid : children) {
    int status = 0;
    if (::waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
      std::printf("FAIL: child process %d failed\n", static_cast<int>(pid));
      ok = false;
    }
  }
  return Expect(ok, "3 concurrent child processes load and run the model");
}
#endif  // _WIN32

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::strcmp(argv[1], "--child") == 0) {
    return RunChild(argv[2], std::atoi(argv[3]));
  }
  if (argc >= 3 && std::strcmp(argv[1], "--dump") == 0) {
    std::unique_ptr<Engine> engine = Engine::CreateFromFile(argv[2]);
    if (engine == nullptr) {
      std::printf("failed to load %s\n", argv[2]);
      return 1;
    }
    PrintEngineInfo(*engine);
    return 0;
  }

  std::string model_path = kDefaultModelPath;
  if (argc >= 3 && std::strcmp(argv[1], "--model") == 0) {
    model_path = argv[2];
  }

  std::ifstream probe(model_path, std::ios::binary);
  if (!probe.good()) {
    std::printf("SKIP: model file not found: %s\n", model_path.c_str());
    return 0;
  }
  probe.close();

  std::printf("== MNN engine tests, model: %s ==\n", model_path.c_str());
  TestSingleProcess(model_path);
  TestDeinit(model_path);
  TestMultiThread(model_path);
  TestNegative(model_path);
#ifdef _WIN32
  TestMultiProcess(model_path);
#else
  TestMultiProcess(model_path, argv[0]);
#endif

  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::printf("%d TEST(S) FAILED\n", g_failures);
  return 1;
}
