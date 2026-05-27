#pragma once

#include "py_object_reader.hpp"

#include <input/native_spec.hpp>
#include <pybind11/numpy.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace beads {
namespace bindings {
namespace ndarray_reader {

inline std::string dtype_name(const pybind11::dtype& dtype) {
  return pybind11::str(dtype.attr("name")).cast<std::string>();
}

inline std::string shape_string(const std::vector<std::size_t>& shape) {
  std::ostringstream out;
  out << "(";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) {
      out << ", ";
    }
    out << shape[index];
  }
  if (shape.size() == 1) {
    out << ",";
  }
  out << ")";
  return out.str();
}

template <typename T>
input::ArrayViewSpec<T> require_array(
    const pybind11::object& value,
    const std::string& path,
    std::vector<std::size_t> expected_shape) {
  if (!pybind11::isinstance<pybind11::array>(value)) {
    py_object::throw_type_error(path + " must be a numpy array.");
  }

  const pybind11::array array = value.cast<pybind11::array>();
  const pybind11::dtype expected_dtype = pybind11::dtype::of<T>();
  if (!array.dtype().is(expected_dtype)) {
    py_object::throw_type_error(
        path + " must have dtype " + dtype_name(expected_dtype) + ".");
  }

  if (static_cast<std::size_t>(array.ndim()) != expected_shape.size()) {
    py_object::throw_value_error(
        path + " must have shape " + shape_string(expected_shape) + ".");
  }

  for (std::size_t axis = 0; axis < expected_shape.size(); ++axis) {
    if (static_cast<std::size_t>(array.shape(axis)) != expected_shape[axis]) {
      py_object::throw_value_error(
          path + " must have shape " + shape_string(expected_shape) + ".");
    }
  }

  if ((array.flags() & pybind11::array::c_style) != pybind11::array::c_style) {
    py_object::throw_value_error(path + " must be C-contiguous.");
  }

  if (array.data() == nullptr) {
    py_object::throw_value_error(path + " data pointer must not be null.");
  }

  return input::ArrayViewSpec<T>{
      static_cast<const T*>(array.data()),
      std::move(expected_shape)};
}

}  // namespace ndarray_reader
}  // namespace bindings
}  // namespace beads
