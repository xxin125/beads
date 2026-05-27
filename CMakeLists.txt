cmake_minimum_required(VERSION 3.24)

project(beads LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(beads_native_config INTERFACE)
target_include_directories(
  beads_native_config
  INTERFACE
    native/include
    native/src
)

set(
  BEADS_NATIVE_SYSTEM_STATE_SOURCES
  native/src/system/state/device_forces.cu
  native/src/system/state/device_particles.cu
  native/src/system/state/device_state.cu
  native/src/system/state/host_forces.cpp
  native/src/system/state/host_particles.cpp
  native/src/system/state/host_state.cpp
  native/src/system/state/tag_to_slot_map.cu
)

set(
  BEADS_NATIVE_SYSTEM_GEOMETRY_SOURCES
  native/src/system/geometry/box_geometry.cpp
)

set(
  BEADS_NATIVE_SYSTEM_UNITS_SOURCES
  native/src/system/units/unit_system.cpp
)

set(
  BEADS_NATIVE_SYSTEM_TOPOLOGY_SOURCES
  native/src/system/topology/exclusions.cu
  native/src/system/topology/topology.cpp
)

set(
  BEADS_NATIVE_FORCEFIELD_SOURCES
  native/src/forcefield/force_eval.cu
  native/src/forcefield/force_field_model.cpp
  native/src/forcefield/angle/angle_factory.cpp
  native/src/forcefield/angle/angle_harmonic.cu
  native/src/forcefield/angle/angle_model.cpp
  native/src/forcefield/bond/bond_factory.cpp
  native/src/forcefield/bond/bond_harmonic.cu
  native/src/forcefield/bond/bond_model.cpp
  native/src/forcefield/dihedral/dihedral_factory.cpp
  native/src/forcefield/dihedral/dihedral_harmonic.cu
  native/src/forcefield/dihedral/dihedral_model.cpp
  native/src/forcefield/pair/pair_factory.cpp
  native/src/forcefield/pair/pair_lj.cu
  native/src/forcefield/pair/pair_model.cpp
)

set(
  BEADS_NATIVE_SIMULATION_NEIGHBOR_SOURCES
  native/src/simulation/neighbor/full/direct_neighbor_builder.cu
  native/src/simulation/neighbor/full/regular_cell_neighbor_builder.cu
  native/src/simulation/neighbor/full/small_dimension_neighbor_builder.cu
  native/src/simulation/neighbor/neighbor_list.cu
  native/src/simulation/neighbor/neighbor_plan.cpp
  native/src/simulation/neighbor/rebuild_tracker.cu
  native/src/simulation/neighbor/neighbor_system.cpp
  native/src/simulation/neighbor/cell_list_build.cu
)

set(
  BEADS_NATIVE_DYNAMICS_SOURCES
  native/src/dynamics/dynamics_program.cu
  native/src/dynamics/integrator/velocity_verlet.cu
  native/src/dynamics/thermostat/berendsen_thermostat.cu
)

set(
  BEADS_NATIVE_SIMULATION_SOURCES
  native/src/simulation/output/output_system.cu
  native/src/simulation/output/host_writer.cpp
  native/src/simulation/output/log/log_output.cpp
  native/src/simulation/output/scalar_reduction.cu
  native/src/simulation/output/snapshot/final_state_output.cu
  native/src/simulation/output/thermo/thermo_output.cu
  native/src/simulation/output/trajectory/trajectory_output.cu
  native/src/simulation/simulation_run.cpp
)

add_library(
  beads_native
  STATIC
  ${BEADS_NATIVE_SYSTEM_GEOMETRY_SOURCES}
  ${BEADS_NATIVE_SYSTEM_UNITS_SOURCES}
  ${BEADS_NATIVE_SYSTEM_TOPOLOGY_SOURCES}
  ${BEADS_NATIVE_FORCEFIELD_SOURCES}
  ${BEADS_NATIVE_DYNAMICS_SOURCES}
  ${BEADS_NATIVE_SIMULATION_NEIGHBOR_SOURCES}
  ${BEADS_NATIVE_SYSTEM_STATE_SOURCES}
  ${BEADS_NATIVE_SIMULATION_SOURCES}
)
target_link_libraries(
  beads_native
  PUBLIC
    beads_native_config
    CUDA::cudart
    Threads::Threads
)

set(
  BEADS_NATIVE_BINDING_SOURCES
  native/src/bindings/module.cpp
  native/src/bindings/native_spec_parser.cpp
)

pybind11_add_module(
  _native
  MODULE
  NO_EXTRAS
  ${BEADS_NATIVE_BINDING_SOURCES}
)
target_link_libraries(
  _native
  PRIVATE
    beads_native
)

set_target_properties(
  _native
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python/beads
)

add_custom_command(
  TARGET _native
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/python/beads
          ${CMAKE_CURRENT_BINARY_DIR}/python/beads
)

install(
  TARGETS _native
  LIBRARY DESTINATION beads
  RUNTIME DESTINATION beads
)
