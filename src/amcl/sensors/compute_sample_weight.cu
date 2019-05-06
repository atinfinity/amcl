#include <cassert>
#include "amcl/sensors/compute_sample_weight.cuh"

using namespace amcl;

#define CUDA_SAFE_CALL(call)                            \
{                                                       \
    const cudaError_t error = call;                     \
    if (error != cudaSuccess)                           \
    {                                                   \
        printf("Error: %s:%d,  ", __FILE__, __LINE__);  \
        printf("code:%d, reason: %s\n", error,          \
            cudaGetErrorString(error));                 \
        exit(1);                                        \
    }                                                   \
}

__device__
pf_vector_t dev_pf_vector_coord_add(const pf_vector_t a, const pf_vector_t b)
{
  pf_vector_t c;

  c.v[0] = b.v[0] + a.v[0] * cos(b.v[2]) - a.v[1] * sin(b.v[2]);
  c.v[1] = b.v[1] + a.v[0] * sin(b.v[2]) + a.v[1] * cos(b.v[2]);
  c.v[2] = b.v[2] + a.v[2];
  c.v[2] = atan2(sin(c.v[2]), cos(c.v[2]));

  return c;
}

__device__
double atomicAdd_double(double* address, double val)
{
  unsigned long long int* address_as_ull =(unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__global__
void dev_compute_sample_weight(pf_sample_t samples[], const int max_count,
  const double z_hit, const double z_rand, double const z_hit_denom, const double z_rand_mult, 
  const int step, const pf_vector_t laser_pose, const map_t *map, const map_cell_t map_cells[], 
  const int range_count, const double range_max, const double (*dev_ranges)[2], double *total_weight)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  i = max(i, 0);
  i = min(i, max_count);

  if (i < max_count)
  {
    // Take account of the laser pose relative to the robot
    pf_vector_t pose;
    pose = dev_pf_vector_coord_add(laser_pose, samples[i].pose);

    double p = 1.0;
    int j;

    for (j = 0; j < range_count; j += step)
    {
      double z, pz;
      pf_vector_t hit;
      int mi, mj;
      double obs_range = dev_ranges[j][0];
      double obs_bearing = dev_ranges[j][1];

      // This model ignores max range readings
      if (obs_range >= range_max)
      {
        continue;
      }

      // Check for NaN
      if (obs_range != obs_range)
      {
        continue;
      }

      pz = 0.0;

      // Compute the endpoint of the beam
      hit.v[0] = pose.v[0] + obs_range * cos(pose.v[2] + obs_bearing);
      hit.v[1] = pose.v[1] + obs_range * sin(pose.v[2] + obs_bearing);

      // Convert to map grid coords.
      mi = MAP_GXWX(map, hit.v[0]);
      mj = MAP_GYWY(map, hit.v[1]);

      // Part 1: Get distance from the hit to closest obstacle.
      // Off-map penalized as max distance
      if(!MAP_VALID(map, mi, mj))
      {
        z = map->max_occ_dist;
      }
      else
      {
        z = map_cells[MAP_INDEX(map,mi,mj)].occ_dist;
      }

      // Gaussian model
      // NOTE: this should have a normalization of 1/(sqrt(2pi)*sigma)
      pz += z_hit * exp(-(z * z) / z_hit_denom);

      // Part 2: random measurements
      pz += z_rand * z_rand_mult;

      // TODO: outlier rejection for short readings

      //      p *= pz;
      // here we have an ad-hoc weighting scheme for combining beam probs
      // works well, though...
      p += pz*pz*pz;
    }

    samples[i].weight *= p;

     __syncthreads(); 
    atomicAdd_double(total_weight, samples[i].weight);
  }
}


double cu_compute_sample_weight(const amcl::AMCLLaserData *data, pf_sample_t *samples, const int sample_count)
{
  AMCLLaser *self = (AMCLLaser*) data->sensor;
  double total_weight = 0.0;

  // pre-compute a couple of things
  const pf_vector_t laser_pose = self->laser_pose;
  const double sigma_hit = self->sigma_hit;
  const int max_beams = self->max_beams;
  const map_t *map = self->map;
  const double z_hit = self->z_hit;
  const double z_rand = self->z_rand;
  const double z_hit_denom = 2 * sigma_hit * sigma_hit;
  const double z_rand_mult = 1.0 / data->range_max;
  const int range_count = data->range_count;
  const double range_max = data->range_max;
  int step = (range_count - 1) / (max_beams - 1);

  // Step size must be at least 1
  if (step < 1)
  {
    step = 1;
  }

  pf_sample_t *dev_samples;
  map_t  *dev_map;
  map_cell_t *dev_map_cells;
  double (*dev_ranges)[2];
  double *dev_total_weight;

  CUDA_SAFE_CALL(cudaMalloc(&dev_samples, sizeof(pf_sample_t) * sample_count));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map, sizeof(map_t)));
  CUDA_SAFE_CALL(cudaMalloc(&dev_map_cells, sizeof(map_cell_t) * map->size_x * map->size_y));
  CUDA_SAFE_CALL(cudaMalloc(&dev_ranges, sizeof(double) * range_count * 2));
  CUDA_SAFE_CALL(cudaMalloc(&dev_total_weight, sizeof(double)));

  CUDA_SAFE_CALL(cudaMemcpy(dev_samples, samples, sizeof(pf_sample_t) * sample_count, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map, map, sizeof(map_t), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_map_cells, map->cells, sizeof(map_cell_t) * map->size_x * map->size_y, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_ranges, data->ranges, sizeof(double) * range_count * 2, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(dev_total_weight, &total_weight, sizeof(double), cudaMemcpyHostToDevice));

  dim3 block(256);
  dim3 grid((sample_count + block.x - 1) / block.x);

  dev_compute_sample_weight<<<grid, block>>>(dev_samples, sample_count,
    z_hit, z_rand, z_hit_denom, z_rand_mult, step, laser_pose,
    dev_map, dev_map_cells, range_count, range_max,
    dev_ranges, dev_total_weight);

  CUDA_SAFE_CALL(cudaGetLastError());
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  CUDA_SAFE_CALL(cudaMemcpy(samples, dev_samples, sizeof(pf_sample_t) * sample_count, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(&total_weight, dev_total_weight, sizeof(double), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(dev_samples));
  CUDA_SAFE_CALL(cudaFree(dev_map));
  CUDA_SAFE_CALL(cudaFree(dev_map_cells));
  CUDA_SAFE_CALL(cudaFree(dev_ranges));
  CUDA_SAFE_CALL(cudaFree(dev_total_weight));

  return(total_weight);
}
