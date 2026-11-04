#ifndef SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_UTILS_H_
#define SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace soundproof {

class SymbolTable;

// Low frame rate stacking: every output frame contains |lfr_window| input
// frames, advancing by |lfr_shift| frames (kaldi LFR).
std::vector<float> ApplyLfr(const float* input, int32_t num_frames,
                            int32_t feat_dim, int32_t lfr_window,
                            int32_t lfr_shift);

// (x + neg_mean) * inv_stddev, applied per feature dimension.
void ApplyCMVN(std::vector<float>* frames,
               const std::vector<float>& neg_mean,
               const std::vector<float>& inv_stddev);

// Sinusoidal positional encoding used by the streaming paraformer.
void ApplyPositionalEncoding(std::vector<float>* frames, int32_t feat_dim,
                             int32_t t_offset);

// Paraformer token -> text conversion (sherpa "Convert"): handles "@@"
// continuation units and inserts spaces between ASCII and CJK text.
std::string ParaformerTextFromTokens(const std::vector<int32_t>& tokens,
                                     const SymbolTable& symbols);

}  // namespace soundproof

#endif  // SOUNDPROOF_CORE_SPEECH_RECOGNITION_PARAFORMER_UTILS_H_
