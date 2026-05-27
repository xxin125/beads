#pragma once

#if defined(__CUDACC__)
#define BEADS_HOST __host__
#define BEADS_DEVICE __device__
#define BEADS_HOST_DEVICE __host__ __device__
#define BEADS_FORCE_INLINE __forceinline__
#else
#define BEADS_HOST
#define BEADS_DEVICE
#define BEADS_HOST_DEVICE
#define BEADS_FORCE_INLINE inline
#endif

