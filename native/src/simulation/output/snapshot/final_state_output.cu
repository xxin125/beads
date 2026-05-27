#include "final_state_output.cuh"

#include <beads/core/cuda_check.cuh>
#include <beads/core/device_buffer.cuh>
#include <system/geometry/box_geometry.hpp>
#include <system/state/device_particles.cuh>
#include <system/units/unit_system.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace beads {
namespace simulation::output::snapshot {
namespace {

constexpr char kMagic[] = "BEADS_SNAPSHOT";
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304u;

std::string snapshot_path_from_prefix(const std::string& prefix) {
  if (prefix.empty()) {
    throw std::invalid_argument("output.final_state.prefix must not be empty.");
  }
  return prefix + ".state.beadsbin";
}

template <typename T>
void write_scalar(std::ofstream& stream, const T& value, const char* label) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!stream) {
    throw std::runtime_error(std::string("failed to write ") + label + ".");
  }
}

void write_bytes(
    std::ofstream& stream,
    const void* data,
    std::size_t byte_count,
    const char* label) {
  if (byte_count == 0) {
    return;
  }
  stream.write(reinterpret_cast<const char*>(data), byte_count);
  if (!stream) {
    throw std::runtime_error(std::string("failed to write ") + label + ".");
  }
}

template <typename T>
void download_array(
    const DeviceBuffer<T>& device,
    std::vector<T>& host,
    const char* label) {
  host.resize(device.size());
  if (host.empty()) {
    return;
  }
  const cudaError_t status = cudaMemcpy(
      host.data(),
      device.data(),
      host.size() * sizeof(T),
      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("failed to download ") + label + ": " +
        cudaGetErrorString(status));
  }
}

template <typename T>
void write_vector(
    std::ofstream& stream,
    const std::vector<T>& values,
    const char* label) {
  write_bytes(
      stream,
      values.data(),
      values.size() * sizeof(T),
      label);
}

}  // namespace

class FinalStateOutput::Impl {
 public:
  explicit Impl(FinalStateOutputConfig config)
      : path_(snapshot_path_from_prefix(config.prefix)) {}

  void prepare(
      index_t n_particles,
      const system::units::UnitSystem& units) {
    n_particles_ = n_particles;
    units_ = units;
    prepared_ = true;
  }

  void write_end_state(
      runstep_t step,
      const system::state::DeviceParticles& particles,
      const system::geometry::BoxGeometry& box) {
    if (!prepared_) {
      throw std::logic_error("final-state output was not prepared before write.");
    }
    if (particles.n_particles() != n_particles_) {
      throw std::logic_error("final-state output received unexpected particle count.");
    }

    // The caller owns the producer-stream synchronization boundary.

    std::vector<index_t> tag;
    std::vector<type_id_t> type;
    std::vector<index_t> molecule_id;
    std::vector<real_t> mass;
    std::vector<real_t> position_x;
    std::vector<real_t> position_y;
    std::vector<real_t> position_z;
    std::vector<real_t> velocity_x;
    std::vector<real_t> velocity_y;
    std::vector<real_t> velocity_z;
    std::vector<image_t> image_x;
    std::vector<image_t> image_y;
    std::vector<image_t> image_z;

    download_array(particles.tag(), tag, "snapshot tag");
    download_array(particles.type(), type, "snapshot type");
    download_array(particles.molecule_id(), molecule_id, "snapshot molecule_id");
    download_array(particles.mass(), mass, "snapshot mass");
    download_array(particles.position_x(), position_x, "snapshot position_x");
    download_array(particles.position_y(), position_y, "snapshot position_y");
    download_array(particles.position_z(), position_z, "snapshot position_z");
    download_array(particles.velocity_x(), velocity_x, "snapshot velocity_x");
    download_array(particles.velocity_y(), velocity_y, "snapshot velocity_y");
    download_array(particles.velocity_z(), velocity_z, "snapshot velocity_z");
    download_array(particles.image_x(), image_x, "snapshot image_x");
    download_array(particles.image_y(), image_y, "snapshot image_y");
    download_array(particles.image_z(), image_z, "snapshot image_z");

    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("failed to open final-state output file " + path_ + ".");
    }

    write_bytes(output, kMagic, sizeof(kMagic) - 1, "snapshot magic");
    write_scalar(output, kVersion, "snapshot version");
    write_scalar(output, static_cast<std::uint32_t>(sizeof(real_t)), "snapshot real size");
    write_scalar(output, static_cast<std::uint32_t>(sizeof(index_t)), "snapshot index size");
    write_scalar(output, static_cast<std::uint32_t>(sizeof(image_t)), "snapshot image size");
    write_scalar(output, static_cast<std::uint32_t>(sizeof(type_id_t)), "snapshot type size");
    write_scalar(output, static_cast<std::uint32_t>(sizeof(index_t)), "snapshot tag size");
    write_scalar(output, kEndianMarker, "snapshot endian marker");

    const std::string units_name = units_.public_name;
    const auto units_length = static_cast<std::uint64_t>(units_name.size());
    write_scalar(output, units_length, "snapshot units length");
    write_bytes(output, units_name.data(), units_name.size(), "snapshot units");
    write_scalar(output, static_cast<std::uint64_t>(step), "snapshot step");
    write_scalar(output, static_cast<std::uint64_t>(n_particles_), "snapshot particle count");
    write_bytes(output, box.lower.data(), 3 * sizeof(real_t), "snapshot box lower");
    write_bytes(output, box.upper.data(), 3 * sizeof(real_t), "snapshot box upper");

    write_vector(output, tag, "snapshot tag");
    write_vector(output, type, "snapshot type");
    write_vector(output, molecule_id, "snapshot molecule_id");
    write_vector(output, mass, "snapshot mass");
    write_vector(output, position_x, "snapshot position_x");
    write_vector(output, position_y, "snapshot position_y");
    write_vector(output, position_z, "snapshot position_z");
    write_vector(output, velocity_x, "snapshot velocity_x");
    write_vector(output, velocity_y, "snapshot velocity_y");
    write_vector(output, velocity_z, "snapshot velocity_z");
    write_vector(output, image_x, "snapshot image_x");
    write_vector(output, image_y, "snapshot image_y");
    write_vector(output, image_z, "snapshot image_z");

    output.flush();
    if (!output) {
      throw std::runtime_error("failed to write final-state output file " + path_ + ".");
    }
  }

 private:
  std::string path_;
  system::units::UnitSystem units_;
  index_t n_particles_ = 0;
  bool prepared_ = false;
};

FinalStateOutput::FinalStateOutput(FinalStateOutputConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

FinalStateOutput::~FinalStateOutput() = default;
FinalStateOutput::FinalStateOutput(FinalStateOutput&&) noexcept = default;
FinalStateOutput& FinalStateOutput::operator=(FinalStateOutput&&) noexcept = default;

void FinalStateOutput::prepare(
    index_t n_particles,
    const system::units::UnitSystem& units) {
  impl_->prepare(n_particles, units);
}

void FinalStateOutput::write_end_state(
    runstep_t step,
    const system::state::DeviceParticles& particles,
    const system::geometry::BoxGeometry& box) {
  impl_->write_end_state(step, particles, box);
}

}  // namespace simulation::output::snapshot
}  // namespace beads
