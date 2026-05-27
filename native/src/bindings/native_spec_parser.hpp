#pragma once

#include <input/native_spec.hpp>
#include <pybind11/pybind11.h>

namespace beads {
namespace bindings {

input::SimulationSpec parse_native_spec(const pybind11::object& spec);

}  // namespace bindings
}  // namespace beads
