#pragma once

#include <algorithm>
#include <numeric>
#include <span>
#include <vector>

#include <stdexec/execution.hpp>

namespace runlab::kernels {

template <typename T>
concept DenseVector = requires(T t) {
  std::span<const float>(t.data(), t.size());
};

using Vec = std::vector<float>;

inline auto compute_embedding(Vec data) {
  return stdexec::then(stdexec::just(std::move(data)), [](Vec values) {
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = values[i] * 0.5f + static_cast<float>(i % 7);
    }
    return values;
  });
}

inline auto scale(Vec data, float factor) {
  return stdexec::then(stdexec::just(std::move(data)), [factor](Vec values) {
    for (auto& v : values) {
      v *= factor;
    }
    return values;
  });
}

inline auto add(Vec a, Vec b) {
  return stdexec::then(
    stdexec::just(std::make_pair(std::move(a), std::move(b))),
    [](std::pair<Vec, Vec> data) {
      auto& left = data.first;
      const auto& right = data.second;
      const size_t n = std::min(left.size(), right.size());
      for (size_t i = 0; i < n; ++i) {
        left[i] += right[i];
      }
      return left;
    });
}

inline auto sum(Vec data) {
  return stdexec::then(stdexec::just(std::move(data)), [](const Vec& values) {
    return std::accumulate(values.begin(), values.end(), 0.0f);
  });
}

}  // namespace runlab::kernels
