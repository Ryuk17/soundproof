#ifndef SOUNDPROOF_MODEL_ONLINE_TRANSDUCER_MODEL_H_
#define SOUNDPROOF_MODEL_ONLINE_TRANSDUCER_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mnn_engine/mnn_engine.h"

namespace soundproof {

struct TransducerModelConfig {
  std::string encoder_path;
  std::string decoder_path;
  std::string joiner_path;
  int num_threads = 1;
};

// Streaming zipformer2 transducer (encoder / decoder / joiner) on top of the
// generic MNN inference engine.
//
// Encoder inputs are taken from the model itself (feature "x" plus cached
// state tensors such as cached_key_* / cached_conv* / embed_states /
// processed_lens), so state shapes always match what the model expects.
// Decoder and joiner run one hypothesis at a time (their batch dimension is
// fixed to 1 in the converted models).
//
// One instance belongs to one audio stream; it is not thread-safe.
class OnlineTransducerModel {
 public:
  OnlineTransducerModel();
  ~OnlineTransducerModel();

  OnlineTransducerModel(const OnlineTransducerModel&) = delete;
  OnlineTransducerModel& operator=(const OnlineTransducerModel&) = delete;

  bool Init(const TransducerModelConfig& config);

  int32_t ChunkSize() const { return chunk_size_; }
  int32_t ChunkShift() const { return chunk_shift_; }
  int32_t ContextSize() const { return context_size_; }
  int32_t VocabSize() const { return vocab_size_; }
  int32_t FeatureDim() const { return feature_dim_; }
  int32_t JoinerDim() const { return joiner_dim_; }
  // Value of the encoder's "model_type" metadata (e.g. "zipformer",
  // "zipformer2", "conformer"); empty if the model does not carry it.
  const std::string& ModelType() const { return model_type_; }

  // Runs the encoder for one chunk of |chunk_size| feature frames.
  // |features| has chunk_size * FeatureDim() floats (row-major).
  // On success |encoder_out| holds num_frames * out_dim floats.
  bool RunEncoder(const std::vector<float>& features,
                  std::vector<float>* encoder_out, int32_t* num_frames,
                  int32_t* out_dim);

  // Runs the stateless decoder on one hypothesis' token history
  // (|tokens| size must equal ContextSize()).
  bool RunDecoder(const std::vector<int32_t>& tokens,
                  std::vector<float>* decoder_out);

  // Runs the joiner on one encoder frame (JoinerDim() floats) and the decoder
  // output (JoinerDim() floats). |logits| receives VocabSize() floats.
  bool RunJoiner(const float* encoder_frame,
                 const std::vector<float>& decoder_out,
                 std::vector<float>* logits);

  // Clears all encoder caches (start a new stream segment).
  void ResetStates();

 private:
  bool InitEncoder(const std::string& path, int num_threads);
  bool InitDecoder(const std::string& path, int num_threads);
  bool InitJoiner(const std::string& path, int num_threads);

  mnn_engine::Engine encoder_;
  mnn_engine::Engine decoder_;
  mnn_engine::Engine joiner_;

  std::vector<mnn_engine::Tensor> encoder_inputs_;
  std::vector<mnn_engine::Tensor> encoder_state_templates_;
  std::vector<size_t> encoder_state_indices_;
  size_t encoder_feature_index_ = 0;

  std::vector<mnn_engine::Tensor> decoder_inputs_;

  std::vector<mnn_engine::Tensor> joiner_inputs_;
  size_t joiner_encoder_index_ = 0;
  size_t joiner_decoder_index_ = 1;

  int32_t chunk_size_ = 0;
  int32_t chunk_shift_ = 0;
  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
  int32_t feature_dim_ = 0;
  int32_t joiner_dim_ = 0;
  std::string model_type_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_MODEL_ONLINE_TRANSDUCER_MODEL_H_
