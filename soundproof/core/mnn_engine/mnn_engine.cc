#include "mnn_engine.h"

#include <MNN/MNNDefine.h>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include <cstdint>
#include <limits>
#include <mutex>
#include <utility>

#include "mnn_model_reader.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace soundproof {
namespace mnn_engine {
namespace {

int64_t CurrentProcessId() {
#ifdef _WIN32
  return static_cast<int64_t>(::GetCurrentProcessId());
#else
  return static_cast<int64_t>(::getpid());
#endif
}

halide_type_t ToHalideType(DataType type) {
  switch (type) {
    case DataType::kFloat32:
      return halide_type_t(halide_type_float, 32);
    case DataType::kInt32:
      return halide_type_t(halide_type_int, 32);
    case DataType::kInt64:
      return halide_type_t(halide_type_int, 64);
    case DataType::kInt8:
      return halide_type_t(halide_type_int, 8);
    case DataType::kUInt8:
      return halide_type_t(halide_type_uint, 8);
    case DataType::kFloat16:
      return halide_type_t(halide_type_float, 16);
  }
  return halide_type_t(halide_type_float, 32);
}

bool FromHalideType(const halide_type_t& type, DataType* out) {
  if (type.lanes != 1) {
    return false;
  }
  if (type.code == halide_type_float) {
    if (type.bits == 32) {
      *out = DataType::kFloat32;
      return true;
    }
    if (type.bits == 16) {
      *out = DataType::kFloat16;
      return true;
    }
    return false;
  }
  if (type.code == halide_type_int) {
    if (type.bits == 32) {
      *out = DataType::kInt32;
      return true;
    }
    if (type.bits == 64) {
      *out = DataType::kInt64;
      return true;
    }
    if (type.bits == 8) {
      *out = DataType::kInt8;
      return true;
    }
    return false;
  }
  if (type.code == halide_type_uint && type.bits == 8) {
    *out = DataType::kUInt8;
    return true;
  }
  return false;
}

bool ElementCount(const std::vector<int>& dim, size_t* count) {
  size_t n = 1;
  for (int d : dim) {
    if (d <= 0) {
      return false;
    }
    const size_t u = static_cast<size_t>(d);
    if (n > std::numeric_limits<size_t>::max() / u) {
      return false;
    }
    n *= u;
  }
  *count = n;
  return true;
}

bool ValidateInput(const Tensor& expected, const Tensor& actual) {
  if (actual.type != expected.type) {
    MNN_ERROR("[mnn_engine] input '%s': type mismatch\n",
              expected.name.c_str());
    return false;
  }
  if (actual.dim.size() != expected.dim.size()) {
    MNN_ERROR("[mnn_engine] input '%s': rank mismatch (model %d, got %d)\n",
              expected.name.c_str(), static_cast<int>(expected.dim.size()),
              static_cast<int>(actual.dim.size()));
    return false;
  }
  for (size_t i = 0; i < expected.dim.size(); ++i) {
    const int want = expected.dim[i];
    const int got = actual.dim[i];
    if (got <= 0 || (want > 0 && want != got)) {
      MNN_ERROR("[mnn_engine] input '%s': dim[%d] mismatch (model %d, got %d)\n",
                expected.name.c_str(), static_cast<int>(i), want, got);
      return false;
    }
  }
  size_t count = 0;
  if (!ElementCount(actual.dim, &count)) {
    MNN_ERROR("[mnn_engine] input '%s': invalid shape\n",
              expected.name.c_str());
    return false;
  }
  const size_t expected_bytes =
      count * static_cast<size_t>(BytesPerElement(actual.type));
  if (actual.data.size() != expected_bytes) {
    MNN_ERROR("[mnn_engine] input '%s': data size mismatch (expected %d, got %d)\n",
              expected.name.c_str(), static_cast<int>(expected_bytes),
              static_cast<int>(actual.data.size()));
    return false;
  }
  return true;
}

}  // namespace

struct Engine::Impl {
  // Runtime must outlive the module, so declare it first (members are
  // destroyed in reverse declaration order).
  std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime;
  std::unique_ptr<MNN::Express::Module> module;

  std::vector<Tensor> inputs;
  std::vector<std::string> output_names;
  std::map<std::string, std::string> metadata;

  int64_t creator_pid = 0;
  // Serializes Init/Deinit/Forward and the accessors.
  mutable std::mutex mutex;
};

int32_t BytesPerElement(DataType type) {
  switch (type) {
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kInt64:
      return 8;
    case DataType::kInt8:
    case DataType::kUInt8:
      return 1;
    case DataType::kFloat16:
      return 2;
  }
  return 0;
}

Engine::Engine() : impl_(new Impl()) {}
Engine::~Engine() = default;

bool Engine::Init(const std::string& model_path, const EngineConfig& config) {
  // Read outside the lock so a concurrent Forward() keeps using the current
  // model while the new file is being read.
  std::vector<uint8_t> model_data;
  if (!ReadModelFile(model_path, &model_data)) {
    return false;
  }
  return InitFromMemory(model_data.data(), model_data.size(), config);
}

bool Engine::InitFromMemory(const void* model_data, size_t model_size,
                            const EngineConfig& config) {
  return InitImpl(static_cast<const uint8_t*>(model_data), model_size, config);
}

bool Engine::InitImpl(const uint8_t* model_data, size_t model_size,
                      const EngineConfig& config) {
  if (impl_ == nullptr || model_data == nullptr || model_size == 0) {
    MNN_ERROR("[mnn_engine] Init: empty model data\n");
    return false;
  }
  Impl& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);

  ReleaseLocked();

  MNN::ScheduleConfig schedule;
  schedule.type = static_cast<MNNForwardType>(config.backend);
  schedule.numThread = config.num_threads > 0 ? config.num_threads : 1;
  MNN::BackendConfig backend_config;
  backend_config.memory = config.low_memory
                              ? MNN::BackendConfig::Memory_Low
                              : MNN::BackendConfig::Memory_Normal;
  schedule.backendConfig = &backend_config;

  std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime(
      MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule));
  if (!runtime) {
    MNN_ERROR("[mnn_engine] failed to create MNN runtime (backend %d, threads %d)\n",
              config.backend, schedule.numThread);
    return false;
  }

  if (!config.backend_cache_path.empty()) {
    // Per-process cache file: several processes must never share one backend
    // cache, or their writes can corrupt each other.
    runtime->setCache(config.backend_cache_path + "." +
                      std::to_string(CurrentProcessId()));
  }

  MNN::Express::Module::Config module_config;
  module_config.rearrange = true;
  module_config.shapeMutable = true;
  std::unique_ptr<MNN::Express::Module> module(MNN::Express::Module::load(
      {}, {}, model_data, model_size, runtime, &module_config));
  if (!module) {
    MNN_ERROR("[mnn_engine] failed to load MNN model\n");
    return false;
  }

  const MNN::Express::Module::Info* info = module->getInfo();
  if (info == nullptr) {
    MNN_ERROR("[mnn_engine] failed to query model info\n");
    return false;
  }

  if (info->inputs.size() != info->inputNames.size()) {
    MNN_PRINT("[mnn_engine] warning: model input info count (%d) differs from input names (%d)\n",
              static_cast<int>(info->inputs.size()),
              static_cast<int>(info->inputNames.size()));
  }
  std::vector<Tensor> inputs;
  inputs.reserve(info->inputNames.size());
  for (size_t i = 0; i < info->inputNames.size(); ++i) {
    Tensor tensor;
    tensor.name = info->inputNames[i];
    if (i < info->inputs.size()) {
      tensor.dim = info->inputs[i].dim;
      if (!FromHalideType(info->inputs[i].type, &tensor.type)) {
        MNN_ERROR("[mnn_engine] input '%s': unsupported element type\n",
                  tensor.name.c_str());
        return false;
      }
    }
    inputs.push_back(std::move(tensor));
  }

  // Commit only after everything succeeded, so a failed Init leaves the
  // engine deinitialized instead of half-configured.
  impl.runtime = std::move(runtime);
  impl.module = std::move(module);
  impl.inputs = std::move(inputs);
  impl.output_names = info->outputNames;
  impl.metadata = info->metaData;
  impl.creator_pid = CurrentProcessId();
  return true;
}

void Engine::ReleaseLocked() {
  Impl& impl = *impl_;
  impl.module.reset();
  impl.runtime.reset();
  impl.inputs.clear();
  impl.output_names.clear();
  impl.metadata.clear();
  impl.creator_pid = 0;
}

void Engine::Deinit() {
  if (impl_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  ReleaseLocked();
}

bool Engine::initialized() const {
  if (impl_ == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->module != nullptr;
}

std::unique_ptr<Engine> Engine::CreateFromFile(const std::string& model_path,
                                               const EngineConfig& config) {
  std::unique_ptr<Engine> engine(new Engine());
  if (!engine->Init(model_path, config)) {
    return nullptr;
  }
  return engine;
}

std::unique_ptr<Engine> Engine::CreateFromMemory(const void* model_data,
                                                 size_t model_size,
                                                 const EngineConfig& config) {
  std::unique_ptr<Engine> engine(new Engine());
  if (!engine->InitFromMemory(model_data, model_size, config)) {
    return nullptr;
  }
  return engine;
}

std::vector<Tensor> Engine::inputs() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->inputs;
}

std::vector<std::string> Engine::output_names() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->output_names;
}

std::map<std::string, std::string> Engine::metadata() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->metadata;
}

bool Engine::Forward(const std::vector<Tensor>& inputs,
                     std::vector<Tensor>* outputs) {
  if (impl_ == nullptr || outputs == nullptr) {
    MNN_ERROR("[mnn_engine] Forward: invalid argument\n");
    return false;
  }
  Impl& impl = *impl_;
  outputs->clear();

  // MNN Module instances are stateful; serialize the whole forward pass.
  std::lock_guard<std::mutex> lock(impl.mutex);

  if (impl.module == nullptr) {
    MNN_ERROR("[mnn_engine] Forward: model is not initialized\n");
    return false;
  }

  const int64_t pid = CurrentProcessId();
  if (pid != impl.creator_pid) {
    MNN_ERROR("[mnn_engine] Engine initialized in process %d but used in process %d; "
              "initialize a new Engine in each process\n",
              static_cast<int>(impl.creator_pid), static_cast<int>(pid));
    return false;
  }

  if (inputs.size() != impl.inputs.size()) {
    MNN_ERROR("[mnn_engine] expected %d inputs, got %d\n",
              static_cast<int>(impl.inputs.size()),
              static_cast<int>(inputs.size()));
    return false;
  }

  std::vector<MNN::Express::VARP> input_varps;
  input_varps.reserve(impl.inputs.size());
  for (const Tensor& expected : impl.inputs) {
    const Tensor* actual = nullptr;
    for (const Tensor& tensor : inputs) {
      if (tensor.name != expected.name) {
        continue;
      }
      if (actual != nullptr) {
        MNN_ERROR("[mnn_engine] duplicate input '%s'\n",
                  expected.name.c_str());
        return false;
      }
      actual = &tensor;
    }
    if (actual == nullptr) {
      MNN_ERROR("[mnn_engine] missing input '%s'\n", expected.name.c_str());
      return false;
    }
    if (!ValidateInput(expected, *actual)) {
      return false;
    }
    input_varps.emplace_back(MNN::Express::_Const(
        actual->data.data(), actual->dim, MNN::Express::NCHW,
        ToHalideType(actual->type)));
  }

  std::vector<MNN::Express::VARP> out_varps = impl.module->onForward(input_varps);
  if (out_varps.size() != impl.output_names.size()) {
    MNN_ERROR("[mnn_engine] expected %d outputs, got %d\n",
              static_cast<int>(impl.output_names.size()),
              static_cast<int>(out_varps.size()));
    return false;
  }

  outputs->reserve(out_varps.size());
  for (size_t i = 0; i < out_varps.size(); ++i) {
    const MNN::Express::Variable::Info* info = out_varps[i]->getInfo();
    if (info == nullptr) {
      MNN_ERROR("[mnn_engine] output '%s': failed to compute info\n",
                impl.output_names[i].c_str());
      return false;
    }
    if (info->order != MNN::Express::NCHW) {
      MNN_ERROR("[mnn_engine] output '%s': unsupported memory order\n",
                impl.output_names[i].c_str());
      return false;
    }
    DataType type = DataType::kFloat32;
    if (!FromHalideType(info->type, &type)) {
      MNN_ERROR("[mnn_engine] output '%s': unsupported element type\n",
                impl.output_names[i].c_str());
      return false;
    }
    const uint8_t* src = out_varps[i]->readMap<uint8_t>();
    if (src == nullptr) {
      MNN_ERROR("[mnn_engine] output '%s': failed to read data\n",
                impl.output_names[i].c_str());
      return false;
    }
    Tensor tensor;
    tensor.name = impl.output_names[i];
    tensor.type = type;
    tensor.dim = info->dim;
    const size_t bytes =
        info->size * static_cast<size_t>(info->type.bytes());
    tensor.data.assign(src, src + bytes);
    outputs->push_back(std::move(tensor));
  }
  return true;
}

}  // namespace mnn_engine
}  // namespace soundproof
