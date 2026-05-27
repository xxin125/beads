#include "exclusions.cuh"

#include <beads/core/cuda_check.cuh>

#include <cub/cub.cuh>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace beads {
namespace system::topology {
namespace {

constexpr int kExclusionBlockSize = 256;

template <typename T>
void upload_vector(DeviceBuffer<T>& buffer, const std::vector<T>& values) {
  buffer.resize(values.size());
  if (!values.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        buffer.data(),
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice));
  }
}

template <typename T>
void download_vector(const DeviceBuffer<T>& buffer, std::vector<T>& values) {
  values.resize(buffer.size());
  if (!values.empty()) {
    BEADS_CUDA_CHECK(cudaMemcpy(
        values.data(),
        buffer.data(),
        values.size() * sizeof(T),
        cudaMemcpyDeviceToHost));
  }
}

int grid_size(index_t n_items) {
  const auto count = static_cast<std::size_t>(n_items);
  const auto block = static_cast<std::size_t>(kExclusionBlockSize);
  const std::size_t blocks = (count + block - 1) / block;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Slot exclusion grid size exceeds int capacity.");
  }
  return static_cast<int>(blocks);
}

int require_cub_int_count(index_t value, const char* label) {
  if (value > static_cast<index_t>(std::numeric_limits<int>::max())) {
    std::ostringstream message;
    message << label << " exceeds CUB int-count capacity.";
    throw std::overflow_error(message.str());
  }
  return static_cast<int>(value);
}

std::pair<index_t, index_t> canonical_pair(index_t lhs, index_t rhs) {
  return lhs < rhs ? std::make_pair(lhs, rhs) : std::make_pair(rhs, lhs);
}

index_t require_index_count(std::size_t count, const char* label) {
  if (count > static_cast<std::size_t>(std::numeric_limits<index_t>::max())) {
    std::ostringstream message;
    message << label << " exceeds supported index_t range.";
    throw std::overflow_error(message.str());
  }
  return static_cast<index_t>(count);
}

struct HostTagCsr {
  index_t tag_count = 0;
  std::vector<index_t> offsets;
  std::vector<index_t> partners;
};

HostTagCsr build_directed_tag_csr(
    index_t tag_count,
    const std::vector<std::pair<index_t, index_t>>& canonical_pairs) {
  std::vector<index_t> degree_by_tag(
      static_cast<std::size_t>(tag_count) + 1u,
      0);
  for (const auto& [tag_i, tag_j] : canonical_pairs) {
    ++degree_by_tag[tag_i];
    ++degree_by_tag[tag_j];
  }

  std::vector<index_t> offsets(static_cast<std::size_t>(tag_count) + 2u, 0);
  std::size_t running = 0;
  for (index_t tag = 1; tag <= tag_count; ++tag) {
    offsets[tag] = require_index_count(running, "tag CSR offset");
    running += static_cast<std::size_t>(degree_by_tag[tag]);
    require_index_count(running, "tag CSR partner count");
  }
  offsets[static_cast<std::size_t>(tag_count) + 1u] =
      require_index_count(running, "tag CSR partner count");

  std::vector<index_t> partners(running);
  std::vector<index_t> cursor = offsets;
  for (const auto& [tag_i, tag_j] : canonical_pairs) {
    partners[cursor[tag_i]++] = tag_j;
    partners[cursor[tag_j]++] = tag_i;
  }

  for (index_t tag = 1; tag <= tag_count; ++tag) {
    std::sort(
        partners.begin() + offsets[tag],
        partners.begin() + offsets[tag + 1u]);
  }
  return HostTagCsr{tag_count, std::move(offsets), std::move(partners)};
}

__device__ inline void sort_inline_segment(index_t* values, index_t count) {
  for (index_t i = 0; i < count; ++i) {
    for (index_t j = i + 1; j < count; ++j) {
      if (values[j] < values[i]) {
        const index_t tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
      }
    }
  }
}

__global__ void project_slot_overflow_counts_kernel(
    state::DeviceParticlesConstView particles,
    ExclusionSourceConstView source,
    index_t* overflow_counts) {
  const index_t slot = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= particles.n_particles) {
    return;
  }
  const index_t tag = particles.tag[slot];
  overflow_counts[slot] = source.overflow_count_by_tag[tag];
}

__global__ void fill_inline_slot_exclusions_kernel(
    state::DeviceParticlesConstView particles,
    ExclusionSourceConstView source,
    const index_t* slots_by_tag,
    ExclusionSlotInfo* slot_info,
    index_t* inline_partners) {
  const index_t slot = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= particles.n_particles) {
    return;
  }

  index_t* inline_base =
      inline_partners + slot * SlotExclusionRuntime::inline_capacity;
  const index_t tag = particles.tag[slot];
  const index_t begin = source.offsets[tag];
  const index_t degree = source.degree_by_tag[tag];
  slot_info[slot] = ExclusionSlotInfo{degree, 0, 0};
  for (index_t ordinal = 0; ordinal < degree; ++ordinal) {
    const index_t partner_tag = source.partners[begin + ordinal];
    inline_base[ordinal] = slots_by_tag[partner_tag];
  }
  sort_inline_segment(inline_base, degree);
}

__global__ void fill_overflow_slot_exclusions_kernel(
    state::DeviceParticlesConstView particles,
    ExclusionSourceConstView source,
    const index_t* slots_by_tag,
    const index_t* overflow_offsets,
    ExclusionSlotInfo* slot_info,
    index_t* inline_partners,
    index_t* overflow_partners) {
  const index_t slot = static_cast<index_t>(
      blockIdx.x * blockDim.x + threadIdx.x);
  if (slot >= particles.n_particles) {
    return;
  }

  index_t* inline_base =
      inline_partners + slot * SlotExclusionRuntime::inline_capacity;
  const index_t tag = particles.tag[slot];
  const index_t begin = source.offsets[tag];
  const index_t degree = source.degree_by_tag[tag];
  const ExclusionSlotInfo info{
      degree,
      overflow_offsets[slot],
      source.overflow_count_by_tag[tag]};
  slot_info[slot] = info;
  for (index_t ordinal = 0; ordinal < info.degree; ++ordinal) {
    const index_t partner_tag = source.partners[begin + ordinal];
    const index_t partner_slot = slots_by_tag[partner_tag];
    if (ordinal < SlotExclusionRuntime::inline_capacity) {
      inline_base[ordinal] = partner_slot;
    } else {
      overflow_partners[
          info.overflow_offset +
          ordinal - SlotExclusionRuntime::inline_capacity] = partner_slot;
    }
  }

  sort_inline_segment(
      inline_base,
      info.degree < SlotExclusionRuntime::inline_capacity
          ? info.degree
          : SlotExclusionRuntime::inline_capacity);
}

void require_valid_source_pair(index_t tag_count, const ExcludedTagPair& pair) {
  if (pair.tag_i == 0 || pair.tag_i > tag_count ||
      pair.tag_j == 0 || pair.tag_j > tag_count) {
    throw std::invalid_argument(
        "bond graph exclusion pair references a tag outside the active system.");
  }
  if (pair.tag_i == pair.tag_j) {
    throw std::invalid_argument("bond graph exclusion pair cannot reference itself.");
  }
  if (pair.tag_i > pair.tag_j) {
    throw std::invalid_argument(
        "bond graph exclusion pairs must be canonical tag_i < tag_j.");
  }
}

}  // namespace

const char* exclusion_slot_runtime_mode_name(
    ExclusionSlotRuntimeMode mode) noexcept {
  switch (mode) {
    case ExclusionSlotRuntimeMode::None:
      return "none";
    case ExclusionSlotRuntimeMode::InlineOnly:
      return "inline_only";
    case ExclusionSlotRuntimeMode::InlinePlusOverflow:
      return "inline_plus_overflow";
  }
  return "none";
}

ExclusionSourceRuntime::ExclusionSourceRuntime(
    index_t tag_count,
    const std::vector<ExcludedTagPair>& pairs) {
  assign(tag_count, pairs);
}

void ExclusionSourceRuntime::assign(
    index_t tag_count,
    const std::vector<ExcludedTagPair>& pairs) {
  tag_count_ = tag_count;
  unique_pair_count_ =
      require_index_count(pairs.size(), "bond graph exclusion pair count");
  max_degree_ = 0;
  overflow_slot_count_ = 0;
  total_overflow_partners_ = 0;
  if (pairs.empty()) {
    offsets_.resize(0);
    partners_.resize(0);
    degree_by_tag_.resize(0);
    overflow_count_by_tag_.resize(0);
    return;
  }

  std::vector<std::pair<index_t, index_t>> canonical_pairs;
  canonical_pairs.reserve(pairs.size());
  for (const ExcludedTagPair& pair : pairs) {
    require_valid_source_pair(tag_count, pair);
    canonical_pairs.emplace_back(pair.tag_i, pair.tag_j);
  }
  std::sort(canonical_pairs.begin(), canonical_pairs.end());
  if (std::adjacent_find(canonical_pairs.begin(), canonical_pairs.end()) !=
      canonical_pairs.end()) {
    throw std::invalid_argument(
        "bond graph exclusion source contains a duplicate canonical pair.");
  }
  const HostTagCsr csr = build_directed_tag_csr(tag_count, canonical_pairs);

  std::vector<index_t> degree_by_tag(
      static_cast<std::size_t>(tag_count) + 1u,
      0);
  std::vector<index_t> overflow_count_by_tag(
      static_cast<std::size_t>(tag_count) + 1u,
      0);
  for (index_t tag = 1; tag <= tag_count; ++tag) {
    const index_t degree = csr.offsets[tag + 1u] - csr.offsets[tag];
    degree_by_tag[tag] = degree;
    max_degree_ = std::max(max_degree_, degree);
    if (degree > SlotExclusionRuntime::inline_capacity) {
      const index_t overflow = degree - SlotExclusionRuntime::inline_capacity;
      overflow_count_by_tag[tag] = overflow;
      ++overflow_slot_count_;
      total_overflow_partners_ += overflow;
    }
  }

  upload_vector(offsets_, csr.offsets);
  upload_vector(partners_, csr.partners);
  upload_vector(degree_by_tag_, degree_by_tag);
  upload_vector(overflow_count_by_tag_, overflow_count_by_tag);
}

ExclusionSourceConstView ExclusionSourceRuntime::view() const noexcept {
  if (empty()) {
    return {};
  }
  return ExclusionSourceConstView{
      tag_count_,
      offsets_.data(),
      partners_.data(),
      degree_by_tag_.data(),
      overflow_count_by_tag_.data()};
}

void ExclusionSourceRuntime::download_offsets(std::vector<index_t>& host) const {
  download_vector(offsets_, host);
}

void ExclusionSourceRuntime::download_partners(std::vector<index_t>& host) const {
  download_vector(partners_, host);
}

void SlotExclusionRuntime::reset_none(index_t slot_count) {
  slot_count_ = slot_count;
  mode_ = ExclusionSlotRuntimeMode::None;
  max_degree_ = 0;
  total_overflow_partners_ = 0;
  slot_generation_ = 0;
  slot_info_.resize(0);
  inline_partners_.resize(0);
  overflow_offsets_.resize(0);
  overflow_counts_.resize(0);
  overflow_partners_.resize(0);
  overflow_sorted_partners_.resize(0);
}

void SlotExclusionRuntime::rebuild_from_source(
    const ExclusionSourceRuntime& source,
    const state::DeviceParticles& particles,
    const state::TagToSlotMap& tag_to_slot_map,
    cudaStream_t stream,
    std::uint64_t slot_generation) {
  if (!tag_to_slot_map.is_current()) {
    throw std::logic_error("SlotExclusionRuntime requires a current TagToSlotMap.");
  }
  slot_count_ = particles.n_particles();
  if (source.empty()) {
    reset_none(particles.n_particles());
    slot_generation_ = slot_generation;
    return;
  }

  max_degree_ = source.max_degree();
  total_overflow_partners_ = source.total_overflow_partners();
  mode_ = total_overflow_partners_ == 0
      ? ExclusionSlotRuntimeMode::InlineOnly
      : ExclusionSlotRuntimeMode::InlinePlusOverflow;

  const std::size_t slot_count =
      static_cast<std::size_t>(particles.n_particles());
  slot_info_.resize(slot_count);
  inline_partners_.resize(slot_count * static_cast<std::size_t>(inline_capacity));

  if (mode_ == ExclusionSlotRuntimeMode::InlineOnly) {
    overflow_counts_.resize(0);
    overflow_offsets_.resize(0);
    overflow_partners_.resize(0);
    overflow_sorted_partners_.resize(0);
    fill_inline_slot_exclusions_kernel<<<
        grid_size(particles.n_particles()),
        kExclusionBlockSize,
        0,
        stream>>>(
            particles.view(),
            source.view(),
            tag_to_slot_map.slots_by_tag().data(),
            slot_info_.data(),
            inline_partners_.data());
    BEADS_CUDA_CHECK(cudaGetLastError());

    slot_generation_ = slot_generation;
    return;
  }

  overflow_counts_.resize(slot_count + 1u);
  overflow_offsets_.resize(slot_count + 1u);
  overflow_partners_.resize(total_overflow_partners_);
  overflow_sorted_partners_.resize(total_overflow_partners_);

  // The projection kernel writes counts for slots [0, N). The extra scan
  // element at N is a sentinel so the segmented-sort end offset is defined.
  BEADS_CUDA_CHECK(cudaMemsetAsync(
      overflow_counts_.data() + slot_count,
      0,
      sizeof(index_t),
      stream));
  project_slot_overflow_counts_kernel<<<
      grid_size(particles.n_particles()),
      kExclusionBlockSize,
      0,
      stream>>>(
          particles.view(),
          source.view(),
          overflow_counts_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());

  ensure_scan_workspace(particles.n_particles() + 1u, stream);
  BEADS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      scan_workspace_.data(),
      scan_workspace_bytes_,
      overflow_counts_.data(),
      overflow_offsets_.data(),
      require_cub_int_count(
          particles.n_particles() + 1u,
          "Slot exclusion overflow scan"),
      stream));

  fill_overflow_slot_exclusions_kernel<<<
      grid_size(particles.n_particles()),
      kExclusionBlockSize,
      0,
      stream>>>(
          particles.view(),
          source.view(),
          tag_to_slot_map.slots_by_tag().data(),
          overflow_offsets_.data(),
          slot_info_.data(),
          inline_partners_.data(),
          overflow_partners_.data());
  BEADS_CUDA_CHECK(cudaGetLastError());

  ensure_sort_workspace(total_overflow_partners_, particles.n_particles(), stream);
  BEADS_CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(
      sort_workspace_.data(),
      sort_workspace_bytes_,
      overflow_partners_.data(),
      overflow_sorted_partners_.data(),
      require_cub_int_count(total_overflow_partners_, "Slot exclusion overflow"),
      require_cub_int_count(particles.n_particles(), "Slot exclusion segment"),
      overflow_offsets_.data(),
      overflow_offsets_.data() + 1,
      stream));

  slot_generation_ = slot_generation;
}

SlotExclusionConstView SlotExclusionRuntime::view() const noexcept {
  if (mode_ == ExclusionSlotRuntimeMode::None) {
    return {};
  }
  return SlotExclusionConstView{
      mode_,
      slot_count_,
      slot_info_.data(),
      inline_partners_.data(),
      mode_ == ExclusionSlotRuntimeMode::InlinePlusOverflow
          ? overflow_sorted_partners_.data()
          : nullptr};
}

void SlotExclusionRuntime::download_slot_info(
    std::vector<ExclusionSlotInfo>& host) const {
  download_vector(slot_info_, host);
}

void SlotExclusionRuntime::download_inline_partners(
    std::vector<index_t>& host) const {
  download_vector(inline_partners_, host);
}

void SlotExclusionRuntime::download_overflow_partners(
    std::vector<index_t>& host) const {
  download_vector(overflow_sorted_partners_, host);
}

void SlotExclusionRuntime::ensure_sort_workspace(
    index_t total_overflow_partners,
    index_t slot_count,
    cudaStream_t stream) {
  std::size_t required_bytes = 0;
  BEADS_CUDA_CHECK(cub::DeviceSegmentedSort::SortKeys(
      nullptr,
      required_bytes,
      overflow_partners_.data(),
      overflow_sorted_partners_.data(),
      require_cub_int_count(total_overflow_partners, "Slot exclusion overflow"),
      require_cub_int_count(slot_count, "Slot exclusion segment"),
      overflow_offsets_.data(),
      overflow_offsets_.data() + 1,
      stream));
  if (required_bytes > sort_workspace_bytes_) {
    sort_workspace_.resize(required_bytes);
    sort_workspace_bytes_ = required_bytes;
  }
}

void SlotExclusionRuntime::ensure_scan_workspace(
    index_t scan_count,
    cudaStream_t stream) {
  std::size_t required_bytes = 0;
  BEADS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
      nullptr,
      required_bytes,
      overflow_counts_.data(),
      overflow_offsets_.data(),
      require_cub_int_count(scan_count, "Slot exclusion overflow scan"),
      stream));
  if (required_bytes > scan_workspace_bytes_) {
    scan_workspace_.resize(required_bytes);
    scan_workspace_bytes_ = required_bytes;
  }
}

std::vector<ExcludedTagPair> compile_bond_graph_excluded_pairs(
    const HostTopology& topology,
    index_t n_particles,
    index_t distance) {
  if (distance < 1 || distance > 3) {
    throw std::invalid_argument("bond graph exclusion distance must be 1, 2, or 3.");
  }

  std::vector<std::pair<index_t, index_t>> canonical_bonds;
  canonical_bonds.reserve(topology.bonds().size());
  for (const BondRecord& bond : topology.bonds()) {
    if (bond.type < 1) {
      throw std::invalid_argument(
          "bond graph exclusion bond type must be positive.");
    }
    if (bond.tag_i == 0 || bond.tag_i > n_particles ||
        bond.tag_j == 0 || bond.tag_j > n_particles) {
      throw std::invalid_argument(
          "bond graph exclusion bond references a tag outside the active system.");
    }
    if (bond.tag_i == bond.tag_j) {
      throw std::invalid_argument(
          "bond graph exclusion bond cannot reference the same tag twice.");
    }
    canonical_bonds.push_back(canonical_pair(bond.tag_i, bond.tag_j));
  }
  std::sort(canonical_bonds.begin(), canonical_bonds.end());
  if (std::adjacent_find(canonical_bonds.begin(), canonical_bonds.end()) !=
      canonical_bonds.end()) {
    throw std::invalid_argument(
        "bond graph exclusion bonds contain a duplicate canonical pair.");
  }
  const HostTagCsr adjacency =
      build_directed_tag_csr(n_particles, canonical_bonds);

  std::vector<std::pair<index_t, index_t>> unique_pairs;
  std::vector<std::uint32_t> seen_generation(
      static_cast<std::size_t>(n_particles) + 1u,
      0);
  std::uint32_t current_generation = 0;
  std::vector<std::pair<index_t, index_t>> queue;
  queue.reserve(static_cast<std::size_t>(distance) * 8u + 1u);
  for (index_t start = 1; start <= n_particles; ++start) {
    ++current_generation;
    if (current_generation == 0) {
      std::fill(seen_generation.begin(), seen_generation.end(), 0);
      current_generation = 1;
    }
    queue.clear();
    std::size_t read = 0;
    seen_generation[start] = current_generation;
    queue.emplace_back(start, index_t{0});
    while (read < queue.size()) {
      const auto [tag, depth] = queue[read++];
      if (depth == distance) {
        continue;
      }
      for (index_t offset = adjacency.offsets[tag];
           offset < adjacency.offsets[tag + 1u];
           ++offset) {
        const index_t partner = adjacency.partners[offset];
        if (seen_generation[partner] == current_generation) {
          continue;
        }
        const index_t next_depth = depth + 1;
        seen_generation[partner] = current_generation;
        if (next_depth <= distance) {
          unique_pairs.push_back(canonical_pair(start, partner));
          queue.emplace_back(partner, next_depth);
        }
      }
    }
  }
  std::sort(unique_pairs.begin(), unique_pairs.end());
  unique_pairs.erase(
      std::unique(unique_pairs.begin(), unique_pairs.end()),
      unique_pairs.end());

  std::vector<ExcludedTagPair> pairs;
  pairs.reserve(unique_pairs.size());
  for (const auto& pair : unique_pairs) {
    pairs.push_back(ExcludedTagPair{pair.first, pair.second});
  }
  return pairs;
}

ExclusionSourceRuntime compile_bond_graph_exclusion_source(
    const HostTopology& topology,
    index_t n_particles,
    index_t distance) {
  return ExclusionSourceRuntime(
      n_particles,
      compile_bond_graph_excluded_pairs(topology, n_particles, distance));
}

}  // namespace system::topology
}  // namespace beads
