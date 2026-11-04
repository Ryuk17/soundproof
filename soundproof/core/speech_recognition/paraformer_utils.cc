#include "core/speech_recognition/paraformer_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "utils/symbol_table.h"

namespace soundproof {
namespace {

bool EndsWithDoubleAt(const std::string& sym) {
  return sym.size() >= 2 && sym[sym.size() - 1] == '@' &&
         sym[sym.size() - 2] == '@';
}

bool StartsWithAscii(const std::string& sym) {
  return !sym.empty() && static_cast<uint8_t>(sym[0]) < 0x80;
}

}  // namespace

std::vector<float> ApplyLfr(const float* input, int32_t num_frames,
                            int32_t feat_dim, int32_t lfr_window,
                            int32_t lfr_shift) {
  if (input == nullptr || num_frames <= 0 || feat_dim <= 0 ||
      lfr_window <= 0 || lfr_shift <= 0 ||
      num_frames < lfr_window) {
    return {};
  }
  const int32_t out_frames = (num_frames - lfr_window) / lfr_shift + 1;
  const int32_t out_dim = feat_dim * lfr_window;
  std::vector<float> output(static_cast<size_t>(out_frames) * out_dim);

  const float* src = input;
  float* dst = output.data();
  for (int32_t i = 0; i < out_frames; ++i) {
    std::copy(src, src + out_dim, dst);
    dst += out_dim;
    src += static_cast<size_t>(lfr_shift) * feat_dim;
  }
  return output;
}

void ApplyCMVN(std::vector<float>* frames,
               const std::vector<float>& neg_mean,
               const std::vector<float>& inv_stddev) {
  if (frames == nullptr || neg_mean.empty() ||
      neg_mean.size() != inv_stddev.size()) {
    return;
  }
  const size_t dim = neg_mean.size();
  const size_t num_frames = frames->size() / dim;
  float* p = frames->data();
  for (size_t i = 0; i < num_frames; ++i) {
    for (size_t k = 0; k < dim; ++k) {
      p[k] = (p[k] + neg_mean[k]) * inv_stddev[k];
    }
    p += dim;
  }
}

void ApplyPositionalEncoding(std::vector<float>* frames, int32_t feat_dim,
                             int32_t t_offset) {
  if (frames == nullptr || feat_dim <= 0 ||
      frames->size() % static_cast<size_t>(feat_dim) != 0) {
    return;
  }
  // log(10000) / (7 * 80 / 2 - 1) for the standard 7x80 paraformer input.
  constexpr float kScale = -0.03301197265941284f;
  const int32_t num_frames =
      static_cast<int32_t>(frames->size() / static_cast<size_t>(feat_dim));
  for (int32_t t = 0; t < num_frames; ++t) {
    float* p = frames->data() + static_cast<size_t>(t) * feat_dim;
    const int32_t offset = t + 1 + t_offset;
    for (int32_t d = 0; d < feat_dim / 2; ++d) {
      const float inv_timescale = offset * std::exp(d * kScale);
      p[d] += std::sin(inv_timescale);
      p[d + feat_dim / 2] += std::cos(inv_timescale);
    }
  }
}

std::string ParaformerTextFromTokens(const std::vector<int32_t>& tokens,
                                     const SymbolTable& symbols) {
  std::string text;
  bool mergeable = false;

  for (size_t i = 0; i < tokens.size(); ++i) {
    std::string sym = symbols.GetDisplaySymbol(tokens[i]);
    if (sym.empty()) {
      continue;
    }

    if (!EndsWithDoubleAt(sym)) {
      if (StartsWithAscii(sym)) {
        if (mergeable) {
          mergeable = false;
          text += sym;
        } else {
          text += " ";
          text += sym;
        }
      } else {
        mergeable = false;
        if (i > 0) {
          const std::string prev = symbols.GetDisplaySymbol(tokens[i - 1]);
          if (StartsWithAscii(prev)) {
            text += " ";
          }
        }
        text += sym;
      }
    } else {
      sym.resize(sym.size() - 2);
      if (mergeable) {
        text += sym;
      } else {
        text += " ";
        text += sym;
        mergeable = true;
      }
    }
  }
  return text;
}

}  // namespace soundproof
