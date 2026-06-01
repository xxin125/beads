#pragma once

#include <beads/core/types.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace beads {
namespace forcefield {

inline int force_grid_size(index_t item_count, int block_size) {
  if (block_size <= 0) {
    throw std::invalid_argument("CUDA block size must be positive.");
  }
  const auto items = static_cast<std::size_t>(item_count);
  const auto block = static_cast<std::size_t>(block_size);
  const std::size_t block_count = (items + block - 1) / block;
  if (block_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("CUDA grid size exceeds launch capacity.");
  }
  return static_cast<int>(block_count);
}

}  // namespace forcefield
}  // namespace beads
