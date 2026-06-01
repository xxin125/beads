#pragma once

#include <cstdint>

namespace beads {

// Floating-point precision is selected at build time. Define BEADS_REAL64
// (CMake: -DBEADS_REAL64=ON) for the double-precision engine; the default is
// single precision. The active precision is reported by build_info() and drives
// the NumPy dtypes and binary real-size used on the Python side, so the whole
// stack stays consistent with whichever real_t this build was compiled with.
#ifdef BEADS_REAL64
using real_t = double;
#else
using real_t = float;
#endif
using type_id_t = std::int32_t;
using index_t = std::uint32_t;
using image_t = std::int32_t;
using runstep_t = std::uint64_t;

}  // namespace beads
