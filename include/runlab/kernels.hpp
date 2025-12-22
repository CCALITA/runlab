#pragma once

#include <algorithm>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include <stdexec/execution.hpp>

namespace runlab::kernels {

template <typename T>
concept DenseVector = requires(T t) {
  std::span<const float>(t.data(), t.size());
};

using Vec = std::vector<float>;

inline auto compute_embedding(std::span<const float> data) {
  return stdexec::then(stdexec::just(data), [](std::span<const float> values) {
    Vec out(values.begin(), values.end());
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = out[i] * 0.5f + static_cast<float>(i % 7);
    }
    return out;
  });
}

inline auto compute_embedding(Vec data) {
  return stdexec::then(stdexec::just(std::move(data)), [](Vec values) {
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = values[i] * 0.5f + static_cast<float>(i % 7);
    }
    return values;
  });
}

inline auto scale(std::span<const float> data, float factor) {
  return stdexec::then(
    stdexec::just(std::make_pair(data, factor)),
    [](std::pair<std::span<const float>, float> input) {
      const auto values = input.first;
      const float f = input.second;
      Vec out(values.begin(), values.end());
      for (auto& v : out) {
        v *= f;
      }
      return out;
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

inline auto add(std::span<const float> a, std::span<const float> b) {
  return stdexec::then(
    stdexec::just(std::make_pair(a, b)),
    [](std::pair<std::span<const float>, std::span<const float>> input) {
      const auto left = input.first;
      const auto right = input.second;
      const size_t n = std::min(left.size(), right.size());
      Vec out;
      out.reserve(n);
      for (size_t i = 0; i < n; ++i) {
        out.push_back(left[i] + right[i]);
      }
      return out;
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

inline auto sum(std::span<const float> data) {
  return stdexec::then(stdexec::just(data), [](std::span<const float> values) {
    return std::accumulate(values.begin(), values.end(), 0.0f);
  });
}

inline auto sum(Vec data) {
  return stdexec::then(stdexec::just(std::move(data)), [](const Vec& values) {
    return std::accumulate(values.begin(), values.end(), 0.0f);
  });
}

}  // namespace runlab::kernels
