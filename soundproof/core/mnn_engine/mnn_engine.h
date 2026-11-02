#ifndef SOUNDPROOF_CORE_MNN_ENGINE_MNN_ENGINE_H_
#define SOUNDPROOF_CORE_MNN_ENGINE_MNN_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace soundproof {
namespace mnn_engine {

// Element type of a tensor. Mirrors MNN's halide_type_t without exposing MNN
// headers to consumers of the engine.
enum class DataType : int32_t {
  kFloat32 = 0,
  kInt32 = 1,
  kInt64 = 2,
  kInt8 = 3,
  kUInt8 = 4,
  kFloat16 = 5,
};

int32_t BytesPerElement(DataType type);

// One tensor, used both as a model I/O descriptor and as an inference buffer:
//  - a descriptor (returned by Engine::inputs()) has an empty |data|;
//  - an inference buffer is prepared by the caller and passed to Forward().
struct Tensor {
  std::string name;
  DataType type = DataType::kFloat32;
  // -1 (or 0) marks a dynamic dimension.
  std::vector<int> dim;
  // Raw tensor bytes, interpreted according to |type|.
  std::vector<uint8_t> data;
};

struct EngineConfig {
  int num_threads = 1;
  // MNNForwardType value. 0 is MNN_FORWARD_CPU.
  int backend = 0;
  bool low_memory = true;
  // Optional backend cache file path prefix (used by GPU-like backends).
  // A per-process suffix is appended automatically so that several processes
  // never share the same cache file.
  std::string backend_cache_path;
};

// Generic MNN inference engine: loads one .mnn model and runs named,
// multi-input / multi-output inference. Preparing inputs and consuming outputs
// (including streaming state carried between frames) is the caller's job; the
// engine only owns the model lifecycle and the forward pass.
//
// Thread safety: Init(), Deinit(), Forward() and the accessors are all
// serialized by an internal mutex, so one Engine may be shared by several
// threads of the same process. A stateful streaming model still needs one
// state (its own input/output tensors) per stream.
//
// Process safety: an Engine must be created in every process that uses it.
// Forward() refuses to execute in a process different from the one that
// initialized the model (which catches use of an Engine inherited via fork).
class Engine {
 public:
  Engine();
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Loads a model. Any previously loaded model is released first.
  // Returns false on failure; the reason is logged with MNN_ERROR.
  bool Init(const std::string& model_path, const EngineConfig& config = {});
  bool InitFromMemory(const void* model_data, size_t model_size,
                      const EngineConfig& config = {});

  // Releases the loaded model and runtime. Idempotent and thread-safe;
  // the Engine may be Init()ed again afterwards.
  void Deinit();

  bool initialized() const;

  // Convenience factories: return an initialized Engine or nullptr.
  static std::unique_ptr<Engine> CreateFromFile(
      const std::string& model_path, const EngineConfig& config = {});
  static std::unique_ptr<Engine> CreateFromMemory(const void* model_data,
                                                  size_t model_size,
                                                  const EngineConfig& config = {});

  // Copies of the model input descriptors (name/type/shape, |data| empty).
  std::vector<Tensor> inputs() const;
  // Model output names, in the order used by Forward()'s |outputs|.
  std::vector<std::string> output_names() const;
  std::map<std::string, std::string> metadata() const;

  // Runs one forward pass on caller-prepared tensors.
  // |inputs| must contain exactly one tensor per model input, matched by name
  // (order does not matter). Dynamic dimensions are taken from the supplied
  // tensor. On success |outputs| is filled in model output order.
  // Returns false on failure; the reason is logged with MNN_ERROR.
  bool Forward(const std::vector<Tensor>& inputs,
               std::vector<Tensor>* outputs);

 private:
  bool InitImpl(const uint8_t* model_data, size_t model_size,
                const EngineConfig& config);
  // Requires the internal mutex to be held.
  void ReleaseLocked();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mnn_engine
}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_MNN_ENGINE_MNN_ENGINE_H_
