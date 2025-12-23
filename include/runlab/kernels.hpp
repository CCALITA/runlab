#pragma once

#include <algorithm>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace runlab::kernels {

template <typename T>
concept DenseVector =
    requires(T t) { std::span<const float>(t.data(), t.size()); };

inline auto compute_embedding(std::span<const float> data) {
  Vec out(data.begin(), data.end());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = out[i] * 0.5f + static_cast<float>(i % 7);
  }
  return out;
}

inline auto scale(std::span<const float> data, float factor) {
  Vec out(data.begin(), data.end());
  for (auto &v : out) {
    v *= factor;
  }
  return out;
}

inline auto add(std::span<const float> a, std::span<const float> b) {
  const size_t n = std::min(a.size(), b.size());
  Vec out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(a[i] + b[i]);
  }
  return out;
}

inline auto sum(std::span<const float> data) {
  return std::accumulate(data.begin(), data.end(), 0.0f);
}

} // namespace runlab::kernels
