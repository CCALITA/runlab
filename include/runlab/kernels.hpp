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

using Vec = std::vector<float>;

inline Vec compute_embedding(std::span<const float> data) {
  Vec out(data.begin(), data.end());
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = out[i] * 0.5f + static_cast<float>(i % 7);
  }
  return out;
}

inline Vec compute_embedding(Vec data) {
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = data[i] * 0.5f + static_cast<float>(i % 7);
  }
  return data;
}

inline Vec scale(std::span<const float> data, float factor) {
  Vec out(data.begin(), data.end());
  for (auto &v : out) {
    v *= factor;
  }
  return out;
}

inline Vec scale(Vec data, float factor) {
  for (auto &v : data) {
    v *= factor;
  }
  return data;
}

inline Vec add(std::span<const float> a, std::span<const float> b) {
  const size_t n = std::min(a.size(), b.size());
  Vec out;
  out.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.push_back(a[i] + b[i]);
  }
  return out;
}

inline Vec add(Vec a, Vec b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    a[i] += b[i];
  }
  return a;
}

inline float sum(std::span<const float> data) {
  return std::accumulate(data.begin(), data.end(), 0.0f);
}

inline float sum(Vec data) {
  return std::accumulate(data.begin(), data.end(), 0.0f);
}

} // namespace runlab::kernels
