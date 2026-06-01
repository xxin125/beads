#pragma once

#include <beads/core/types.hpp>
#include <input/native_spec.hpp>

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>

namespace beads {
namespace forcefield {

inline bool has_style_param_key(
    const input::StyleParamMap& params,
    const char* key) {
  return params.find(key) != params.end();
}

inline void require_exact_style_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> required_keys,
    std::string_view context) {
  std::unordered_set<std::string> required;
  for (const char* key : required_keys) {
    required.emplace(key);
    if (!has_style_param_key(params, key)) {
      throw std::invalid_argument(
          std::string(context) + "." + key + " is required.");
    }
  }

  for (const auto& [key, _] : params) {
    if (required.find(key) == required.end()) {
      throw std::invalid_argument(
          std::string(context) + " has unsupported parameter \"" + key + "\".");
    }
  }
}

inline double require_real_style_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view context) {
  const auto iter = params.find(key);
  if (iter == params.end()) {
    throw std::invalid_argument(
        std::string(context) + "." + key + " is required.");
  }
  if (std::holds_alternative<double>(iter->second)) {
    return std::get<double>(iter->second);
  }
  if (std::holds_alternative<std::int64_t>(iter->second)) {
    return static_cast<double>(std::get<std::int64_t>(iter->second));
  }
  throw std::invalid_argument(
      std::string(context) + "." + key + " must be a real value.");
}

inline real_t require_positive_real_style_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view context) {
  const double value = require_real_style_parameter(params, key, context);
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(
        std::string(context) + "." + key + " must be finite and positive.");
  }
  const real_t result = static_cast<real_t>(value);
  if (!std::isfinite(static_cast<double>(result)) || !(result > real_t{0})) {
    throw std::invalid_argument(
        std::string(context) + "." + key + " must be finite and positive.");
  }
  return result;
}

inline real_t require_nonnegative_real_style_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view context) {
  const double value = require_real_style_parameter(params, key, context);
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(
        std::string(context) + "." + key +
        " must be finite and non-negative.");
  }
  const real_t result = static_cast<real_t>(value);
  if (!std::isfinite(static_cast<double>(result)) || result < real_t{0}) {
    throw std::invalid_argument(
        std::string(context) + "." + key +
        " must be finite and non-negative.");
  }
  return result;
}

inline std::int64_t require_integer_style_parameter(
    const input::StyleParamMap& params,
    const char* key,
    std::string_view context) {
  const auto iter = params.find(key);
  if (iter == params.end()) {
    throw std::invalid_argument(
        std::string(context) + "." + key + " is required.");
  }
  if (std::holds_alternative<std::int64_t>(iter->second)) {
    return std::get<std::int64_t>(iter->second);
  }
  throw std::invalid_argument(
      std::string(context) + "." + key + " must be an integer.");
}

inline void require_allowed_style_parameter_keys(
    const input::StyleParamMap& params,
    std::initializer_list<const char*> allowed_keys,
    std::string_view context) {
  std::unordered_set<std::string_view> allowed(allowed_keys.begin(), allowed_keys.end());
  for (const auto& [key, _] : params) {
    if (allowed.find(key) == allowed.end()) {
      throw std::invalid_argument(
          std::string(context) + " has unsupported parameter \"" + key + "\".");
    }
  }
}

inline bool optional_boolean_style_parameter(
    const input::StyleParamMap& params,
    const char* key,
    bool default_value,
    std::string_view context) {
  const auto iter = params.find(key);
  if (iter == params.end()) {
    return default_value;
  }
  if (std::holds_alternative<bool>(iter->second)) {
    return std::get<bool>(iter->second);
  }
  throw std::invalid_argument(
      std::string(context) + "." + key + " must be a boolean.");
}

}  // namespace forcefield
}  // namespace beads
