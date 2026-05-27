#pragma once

#include <pybind11/pybind11.h>

#include <initializer_list>
#include <string>
#include <string_view>

namespace beads {
namespace bindings {
namespace py_object {

[[noreturn]] inline void throw_type_error(const std::string& message) {
  throw pybind11::type_error(message);
}

[[noreturn]] inline void throw_value_error(const std::string& message) {
  throw pybind11::value_error(message);
}

inline pybind11::dict require_dict(
    const pybind11::object& value,
    const std::string& path) {
  if (!pybind11::isinstance<pybind11::dict>(value)) {
    throw_type_error(path + " must be a dict.");
  }
  return value.cast<pybind11::dict>();
}

inline pybind11::object require_item(
    const pybind11::dict& dict,
    const char* key,
    const std::string& path) {
  const pybind11::str py_key(key);
  if (!dict.contains(py_key)) {
    throw_value_error(path + " is missing required field \"" + key + "\".");
  }
  return pybind11::reinterpret_borrow<pybind11::object>(dict[py_key]);
}

inline pybind11::dict require_dict_item(
    const pybind11::dict& dict,
    const char* key,
    const std::string& path) {
  return require_dict(require_item(dict, key, path), path + "." + key);
}

inline pybind11::list require_list(
    const pybind11::object& value,
    const std::string& path) {
  if (!pybind11::isinstance<pybind11::list>(value)) {
    throw_type_error(path + " must be a list.");
  }
  return value.cast<pybind11::list>();
}

inline std::string require_string(
    const pybind11::object& value,
    const std::string& path) {
  if (!pybind11::isinstance<pybind11::str>(value)) {
    throw_type_error(path + " must be a string.");
  }
  const std::string result = value.cast<std::string>();
  if (result.empty()) {
    throw_value_error(path + " must not be empty.");
  }
  return result;
}

inline void require_only_keys(
    const pybind11::dict& dict,
    const std::initializer_list<std::string_view> allowed_keys,
    const std::string& path) {
  for (const auto& item : dict) {
    const pybind11::object key =
        pybind11::reinterpret_borrow<pybind11::object>(item.first);
    const std::string name = require_string(key, path + " key");
    bool allowed = false;
    for (const std::string_view allowed_key : allowed_keys) {
      if (name == allowed_key) {
        allowed = true;
        break;
      }
    }
    if (!allowed) {
      throw_value_error(path + " contains unsupported field \"" + name + "\".");
    }
  }
}

}  // namespace py_object
}  // namespace bindings
}  // namespace beads
