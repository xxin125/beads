#include <pybind11/pybind11.h>

#include "native_spec_parser.hpp"

#include <simulation/simulation_run.hpp>

#include <stdexcept>

namespace py = pybind11;

namespace beads {
namespace bindings {
namespace {

py::dict build_info() {
  py::dict info;
  info["real_precision"] = "single";
  info["real_dtype"] = "float32";
  info["type_dtype"] = "int32";
  info["index_dtype"] = "uint32";
  info["image_dtype"] = "int32";
  info["runstep_dtype"] = "uint64";
  return info;
}

void check_python_interrupt() {
  py::gil_scoped_acquire acquire;
  if (PyErr_CheckSignals() != 0) {
    throw py::error_already_set();
  }
}

void execute_simulation(const py::object& spec) {
  const input::SimulationSpec native_spec = parse_native_spec(spec);
  simulation::SimulationRun simulation_run(
      native_spec,
      []() { check_python_interrupt(); });

  try {
    py::gil_scoped_release release;
    simulation_run.execute();
  } catch (const simulation::NotImplementedFeature& error) {
    PyErr_SetString(PyExc_NotImplementedError, error.what());
    throw py::error_already_set();
  }
}

}  // namespace
}  // namespace bindings
}  // namespace beads

PYBIND11_MODULE(_native, module) {
  module.doc() = "BEADS native extension";

  module.def("build_info", &beads::bindings::build_info);
  module.def("execute_simulation", &beads::bindings::execute_simulation);
}
