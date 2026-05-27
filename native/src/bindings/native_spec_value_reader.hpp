#pragma once

#include "py_object_reader.hpp"

#include <beads/core/types.hpp>
#include <input/native_spec.hpp>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace beads {
namespace bindings {
namespace native_spec_values {

inline std::uint64_t require_uint64(
    const pybind11::object& value,
    const std::string& path) {
  if (pybind11::isinstance<pybind11::bool_>(value) ||
      !pybind11::isinstance<pybind11::int_>(value)) {
    py_object::throw_type_error(path + " must be an integer.");
  }

  const pybind11::int_ zero(0);
  const int is_negative = PyObject_RichCompareBool(value.ptr(), zero.ptr(), Py_LT);
  if (is_negative < 0) {
    PyErr_Clear();
    py_object::throw_type_error(path + " must be an integer.");
  }
  if (is_negative != 0) {
    py_object::throw_value_error(path + " must be non-negative.");
  }

  const unsigned long long result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    py_object::throw_value_error(
        path + " contains values outside the supported uint64 range.");
  }
  return static_cast<std::uint64_t>(result);
}

inline std::int64_t require_int64(
    const pybind11::object& value,
    const std::string& path) {
  if (pybind11::isinstance<pybind11::bool_>(value) ||
      !pybind11::isinstance<pybind11::int_>(value)) {
    py_object::throw_type_error(path + " must be an integer.");
  }

  const long long result = PyLong_AsLongLong(value.ptr());
  if (PyErr_Occurred()) {
    PyErr_Clear();
    py_object::throw_value_error(
        path + " contains values outside the supported int64 range.");
  }
  return static_cast<std::int64_t>(result);
}

inline index_t require_positive_index(
    const pybind11::object& value,
    const std::string& path) {
  const std::uint64_t result = require_uint64(value, path);
  if (result == 0) {
    py_object::throw_value_error(path + " must be positive.");
  }
  if (result > std::numeric_limits<index_t>::max()) {
    py_object::throw_value_error(
        path + " contains values outside the supported uint32 range.");
  }
  return static_cast<index_t>(result);
}

inline index_t require_index(
    const pybind11::object& value,
    const std::string& path) {
  const std::uint64_t result = require_uint64(value, path);
  if (result > std::numeric_limits<index_t>::max()) {
    py_object::throw_value_error(
        path + " contains values outside the supported uint32 range.");
  }
  return static_cast<index_t>(result);
}

inline type_id_t require_positive_type_id(
    const pybind11::object& value,
    const std::string& path) {
  const std::uint64_t result = require_uint64(value, path);
  if (result == 0) {
    py_object::throw_value_error(path + " must be positive.");
  }
  if (result > static_cast<std::uint64_t>(
                   std::numeric_limits<type_id_t>::max())) {
    py_object::throw_value_error(
        path + " contains values outside the supported int32 range.");
  }
  return static_cast<type_id_t>(result);
}

inline runstep_t require_positive_uint64(
    const pybind11::object& value,
    const std::string& path) {
  const runstep_t result = require_uint64(value, path);
  if (result == 0) {
    py_object::throw_value_error(path + " must be positive.");
  }
  return result;
}

inline double require_finite_real(
    const pybind11::object& value,
    const std::string& path) {
  if (pybind11::isinstance<pybind11::bool_>(value) ||
      !(pybind11::isinstance<pybind11::float_>(value) ||
        pybind11::isinstance<pybind11::int_>(value))) {
    py_object::throw_type_error(path + " must be a real number.");
  }
  const double result = value.cast<double>();
  if (!std::isfinite(result)) {
    py_object::throw_value_error(path + " must be finite.");
  }
  return result;
}

inline input::StyleParamValue parse_style_param_value(
    const pybind11::object& value,
    const std::string& path) {
  if (pybind11::isinstance<pybind11::bool_>(value)) {
    return value.cast<bool>();
  }
  if (pybind11::isinstance<pybind11::int_>(value)) {
    return require_int64(value, path);
  }
  if (pybind11::isinstance<pybind11::float_>(value)) {
    const double result = value.cast<double>();
    if (!std::isfinite(result)) {
      py_object::throw_value_error(path + " must be finite.");
    }
    return result;
  }
  if (pybind11::isinstance<pybind11::str>(value)) {
    return value.cast<std::string>();
  }

  py_object::throw_type_error(path + " must be bool, int, float, or string.");
}

inline input::StyleParamMap parse_style_params(
    const pybind11::object& value,
    const std::string& path) {
  const pybind11::dict params = py_object::require_dict(value, path);
  input::StyleParamMap result;
  for (const auto& item : params) {
    const pybind11::object key =
        pybind11::reinterpret_borrow<pybind11::object>(item.first);
    const pybind11::object param_value =
        pybind11::reinterpret_borrow<pybind11::object>(item.second);
    const std::string name = py_object::require_string(key, path + " key");
    result.emplace(
        name,
        parse_style_param_value(param_value, path + "." + name));
  }
  return result;
}

}  // namespace native_spec_values
}  // namespace bindings
}  // namespace beads
