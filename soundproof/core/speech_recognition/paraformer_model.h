#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_MODEL_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_MODEL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "mnn_engine/mnn_engine.h"

namespace soundproof {

struct ParaformerModelConfig {
  std::string encoder_path;
  std::string decoder_path;
  int num_threads = 1;
};

// Streaming paraformer encoder/decoder wrapped around the generic MNN engine.
// Encoder inputs are (features, features_length); outputs are
// (encoder_out, encoder_out_length, alpha). Decoder inputs are
// (encoder_out, encoder_out_length, acoustic_embedding,
//  acoustic_embedding_length, states...) and outputs are
// (logits, sample_ids, new_states...).
//
// One instance belongs to one audio stream; it is not thread-safe.
class ParaformerModel {
 public:
  ParaformerModel();
  ~ParaformerModel();

  ParaformerModel(const ParaformerModel&) = delete;
  ParaformerModel& operator=(const ParaformerModel&) = delete;

  bool Init(const ParaformerModelConfig& config);

  int32_t VocabSize() const { return vocab_size_; }
  int32_t LfrWindowSize() const { return lfr_window_size_; }
  int32_t LfrWindowShift() const { return lfr_window_shift_; }
  int32_t EncoderOutputSize() const { return encoder_output_size_; }
  int32_t DecoderKernelSize() const { return decoder_kernel_size_; }
  int32_t DecoderNumBlocks() const { return decoder_num_blocks_; }
  // Post-LFR feature dimension (feature_dim * lfr_window_size).
  int32_t FeatureDim() const {
    return static_cast<int32_t>(negative_mean_.size());
  }

  const std::vector<float>& NegativeMean() const { return negative_mean_; }
  const std::vector<float>& InverseStdDev() const { return inverse_stddev_; }

  bool ForwardEncoder(const std::vector<float>& features, int32_t num_frames,
                      std::vector<float>* encoder_out, int32_t* out_frames,
                      int32_t* out_dim, std::vector<float>* alpha);

  bool ForwardDecoder(const std::vector<float>& encoder_out,
                      int32_t encoder_frames, int32_t encoder_dim,
                      const std::vector<float>& acoustic_embedding,
                      int32_t num_tokens, std::vector<int32_t>* sample_ids);

  void ResetStates();

 private:
  bool InitEncoder(const std::string& path, int num_threads);
  bool InitDecoder(const std::string& path, int num_threads);

  mnn_engine::Engine encoder_;
  mnn_engine::Engine decoder_;

  std::vector<mnn_engine::Tensor> encoder_inputs_;
  size_t encoder_feature_index_ = 0;
  size_t encoder_length_index_ = 1;

  std::vector<mnn_engine::Tensor> decoder_inputs_;
  std::vector<mnn_engine::Tensor> decoder_state_templates_;
  std::vector<size_t> decoder_state_indices_;
  size_t decoder_encoder_out_index_ = 0;
  size_t decoder_encoder_len_index_ = 1;
  size_t decoder_acoustic_index_ = 2;
  size_t decoder_acoustic_len_index_ = 3;

  int32_t vocab_size_ = 0;
  int32_t lfr_window_size_ = 0;
  int32_t lfr_window_shift_ = 0;
  int32_t encoder_output_size_ = 0;
  int32_t decoder_kernel_size_ = 0;
  int32_t decoder_num_blocks_ = 0;
  std::vector<float> negative_mean_;
  std::vector<float> inverse_stddev_;
  bool ok_ = false;
};

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_MODEL_H_
