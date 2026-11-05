#include "core/speech_recognition/hypothesis.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace soundproof {
namespace {

double LogAdd(double a, double b) {
  if (a == -std::numeric_limits<double>::infinity()) {
    return b;
  }
  if (b == -std::numeric_limits<double>::infinity()) {
    return a;
  }
  const double hi = std::max(a, b);
  const double lo = std::min(a, b);
  return hi + std::log1p(std::exp(lo - hi));
}

}  // namespace

std::string Hypothesis::Key() const {
  std::string key;
  key.reserve(ys.size() * 4);
  for (size_t i = 0; i < ys.size(); ++i) {
    if (i != 0) {
      key.push_back('-');
    }
    key += std::to_string(ys[i]);
  }
  return key;
}

void Hypotheses::Add(Hypothesis hyp) {
  const std::string key = hyp.Key();
  auto it = hyps_.find(key);
  if (it == hyps_.end()) {
    hyps_.emplace(key, std::move(hyp));
  } else {
    it->second.log_prob = LogAdd(it->second.log_prob, hyp.log_prob);
  }
}

Hypothesis Hypotheses::GetMostProbable(bool length_norm) const {
  if (hyps_.empty()) {
    return Hypothesis();
  }
  return std::max_element(
             hyps_.begin(), hyps_.end(),
             [length_norm](const auto& left, const auto& right) {
               if (length_norm && !left.second.ys.empty() &&
                   !right.second.ys.empty()) {
                 return left.second.TotalLogProb() / left.second.ys.size() <
                        right.second.TotalLogProb() / right.second.ys.size();
               }
               return left.second.TotalLogProb() <
                      right.second.TotalLogProb();
             })
      ->second;
}

std::vector<Hypothesis> Hypotheses::Vec() const {
  std::vector<Hypothesis> result;
  result.reserve(hyps_.size());
  for (const auto& kv : hyps_) {
    result.push_back(kv.second);
  }
  return result;
}

}  // namespace soundproof
