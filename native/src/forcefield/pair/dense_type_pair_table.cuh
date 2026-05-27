#pragma once

#include <beads/core/cuda_check.cuh>
#include <beads/core/cuda_macros.cuh>
#include <beads/core/device_buffer.cuh>
#include <beads/core/types.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace beads {
namespace forcefield {
namespace pair {

template <typename T>
struct DenseTypePairTableDeviceView {
  type_id_t active_type_count = 0;
  type_id_t stride = 0;
  const T* values = nullptr;

  BEADS_HOST_DEVICE BEADS_FORCE_INLINE const T& at(
      type_id_t type_i,
      type_id_t type_j
  ) const noexcept
  {
    const auto row = static_cast<int>(type_i - 1);
    const auto col = static_cast<int>(type_j - 1);
    return values[row * static_cast<int>(stride) + col];
  }
};

template <typename T>
class DenseTypePairTable {
  static_assert(
      std::is_trivially_copyable<T>::value,
      "DenseTypePairTable values must be trivially copyable.");

 public:
  void resize(type_id_t active_type_count) {
    require_not_uploaded("DenseTypePairTable");
    if (active_type_count < 1) {
      throw std::invalid_argument(
          "DenseTypePairTable active_type_count must be positive.");
    }

    const auto count = static_cast<std::size_t>(active_type_count);
    if (count > std::numeric_limits<std::size_t>::max() / count) {
      throw std::overflow_error("DenseTypePairTable size overflow.");
    }

    active_type_count_ = active_type_count;
    const std::size_t table_size = count * count;
    values_.assign(table_size, T{});
    filled_.assign(table_size, false);
  }

  type_id_t active_type_count() const noexcept { return active_type_count_; }
  std::size_t size() const noexcept { return values_.size(); }

  void set_symmetric(
      type_id_t type_i,
      type_id_t type_j,
      const T& value,
      const char* context) {
    require_not_uploaded(context);
    const std::size_t ij = checked_index(type_i, type_j, context);
    const std::size_t ji = checked_index(type_j, type_i, context);

    if (filled_[ij]) {
      throw std::invalid_argument(
          std::string(context) + " duplicate coeff for type pair.");
    }

    values_[ij] = value;
    values_[ji] = value;
    filled_[ij] = true;
    filled_[ji] = true;
  }

  const T& at(type_id_t type_i, type_id_t type_j, const char* context) const {
    require_not_uploaded(context);
    const std::size_t index = checked_index(type_i, type_j, context);
    if (!filled_[index]) {
      throw std::logic_error(std::string(context) + " coeff is not set.");
    }
    return values_[index];
  }

  void require_complete(const char* context) const {
    require_not_uploaded(context);
    for (const bool is_filled : filled_) {
      if (!is_filled) {
        throw std::invalid_argument(
            std::string(context) + " requires complete type-pair coeff coverage.");
      }
    }
  }

  void upload_and_release_host(const char* context) {
    require_not_uploaded(context);
    require_complete(context);

    device_values_.resize(values_.size());
    if (!values_.empty()) {
      BEADS_CUDA_CHECK(cudaMemcpy(
          device_values_.data(),
          values_.data(),
          values_.size() * sizeof(T),
          cudaMemcpyHostToDevice));
    }

    values_.clear();
    values_.shrink_to_fit();
    filled_.clear();
    filled_.shrink_to_fit();
    uploaded_ = true;
  }

  DenseTypePairTableDeviceView<T> device_view() const noexcept {
    return DenseTypePairTableDeviceView<T>{
        active_type_count_,
        active_type_count_,
        device_values_.data()};
  }

 private:
  void require_not_uploaded(const char* context) const {
    if (uploaded_) {
      throw std::logic_error(
          std::string(context) + " host staging is no longer available.");
    }
  }

  std::size_t checked_index(
      type_id_t type_i,
      type_id_t type_j,
      const char* context) const {
    if (active_type_count_ < 1) {
      throw std::logic_error(std::string(context) + " table is not initialized.");
    }
    if (type_i < 1 || type_i > active_type_count_ || type_j < 1 ||
        type_j > active_type_count_) {
      throw std::out_of_range(
          std::string(context) + " type id is outside active type range.");
    }

    const auto row = static_cast<std::size_t>(type_i - 1);
    const auto col = static_cast<std::size_t>(type_j - 1);
    const auto stride = static_cast<std::size_t>(active_type_count_);
    return row * stride + col;
  }

  type_id_t active_type_count_ = 0;
  std::vector<T> values_;
  std::vector<bool> filled_;
  DeviceBuffer<T> device_values_;
  bool uploaded_ = false;
};

}  // namespace pair
}  // namespace forcefield
}  // namespace beads
