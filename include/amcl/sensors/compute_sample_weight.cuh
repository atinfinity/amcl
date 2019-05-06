#ifndef PF_GPU_H
#define PF_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cstdio>
#include "amcl/pf/pf.h"
#include "amcl/sensors/amcl_laser.h"

double cu_compute_sample_weight(const amcl::AMCLLaserData *data, pf_sample_t *samples, const int sample_count);

#ifdef __cplusplus
}
#endif

#endif // PF_GPU_H
