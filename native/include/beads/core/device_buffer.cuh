#pragma once

#include <beads/core/cuda_check.cuh>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace beads {

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t size) {
    resize(size);
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release_noexcept();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  ~DeviceBuffer() {
    release_noexcept();
  }

  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }

  void resize(std::size_t size) {
    if (size <= capacity_) {
      size_ = size;
      return;
    }

    T* new_data = allocate(size);
    release_noexcept();
    data_ = new_data;
    size_ = size;
    capacity_ = size;
  }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

 private:
  static T* allocate(std::size_t size) {
    if (size == 0) {
      return nullptr;
    }
    if (size > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::overflow_error("DeviceBuffer allocation size overflows.");
    }

    T* data = nullptr;
    BEADS_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    return data;
  }

  void release_noexcept() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

}  // namespace beads
