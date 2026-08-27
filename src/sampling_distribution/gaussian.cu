#include "mppi_controller_cuda/sampling_distribution/gaussian.cuh"
#include "mppi_controller_cuda/core/mppi_common.cuh"
#include "mppi_controller_cuda/utils/math_utils.h"

namespace mppi
{
namespace sampling_distributions
{
#define GAUSSIAN_TEMPLATE template <template <int> class PARAMS_TEMPLATE, class DYN_PARAMS_T, class SELF_T>
#define GAUSSIAN_CLASS GaussianDistributionImpl<PARAMS_TEMPLATE, DYN_PARAMS_T, SELF_T>

__global__ void setGaussianControls(const float* __restrict__ mean_d, const float* __restrict__ std_dev_d,
                                    float* __restrict__ control_samples_d, const int control_dim,
                                    const int num_timesteps, const int num_rollouts, const int num_distributions,
                                    const int optimization_stride, const float std_dev_decay,
                                    const float pure_noise_percentage, const bool time_specific_std_dev,
                                    const float* __restrict__ min_control_d, const float* __restrict__ max_control_d)
{
  const int trajectory_index = threadIdx.x + blockDim.x * blockIdx.x;
  const int distribution_index = threadIdx.z + blockDim.z * blockIdx.z;
  const int time_index = threadIdx.y + blockDim.y * blockIdx.y;
  const bool valid_index =
      trajectory_index < num_rollouts && time_index < num_timesteps && distribution_index < num_distributions;
  const auto& num_timesteps_block = blockDim.y;
  const auto& num_rollouts_block = blockDim.x;
  const int shared_noise_index = threadIdx.z * num_timesteps_block * num_rollouts_block * control_dim +
                                 threadIdx.x * num_timesteps_block * control_dim + threadIdx.y * control_dim;
  const int global_noise_index =
      min(distribution_index, num_distributions) * num_timesteps * num_rollouts * control_dim +
      min(trajectory_index, num_rollouts) * num_timesteps * control_dim + min(time_index, num_timesteps) * control_dim;
  const int shared_mean_index = distribution_index * num_timesteps * control_dim + time_index * control_dim;
  // Std Deviation setup
  int std_dev_size = num_distributions * control_dim;
  int shared_std_dev_index = threadIdx.z * num_timesteps * control_dim + threadIdx.y * control_dim;
  int global_std_dev_index = min(distribution_index, num_distributions) * num_timesteps * control_dim +
                             min(time_index, num_timesteps) * control_dim;
  shared_std_dev_index = time_specific_std_dev ? shared_std_dev_index : 0;
  global_std_dev_index = time_specific_std_dev ? global_std_dev_index : 0;
  std_dev_size = time_specific_std_dev ? num_timesteps * std_dev_size : std_dev_size;

  // local variables
  int i, j, k;

  // Shared memory setup
  /**
   * @brief Shared memory setup
   * This kernel has three shared memory arrays, mean_shared, std_dev_shared, and control_samples_shared.
   * In order to prevent memory alignment issues, the memory is being over-allocated to ensure that they are start on
   * the float4 boundary mean_shared - size should be num_timesteps * num_distributions * control_dim std_dev_shared =
   * num_distributions * control_dim if time_specific_std_dev is false std_dev_shared = num_distributions *
   * num_timesteps * control_dim if time_specific_std_dev is true control_samples_shared = BLOCKSIZE_X * BLOCKSIZE_Y *
   * BLOCKSIZE_Z * control_dim
   *
   */
  extern __shared__ float entire_buffer[];
  // Create memory_aligned shared memory pointers
  float* mean_shared = entire_buffer;
  float* std_dev_shared = &mean_shared[mppi::math::nearest_multiple_4(num_timesteps * num_distributions * control_dim)];
  float* control_samples_shared = &std_dev_shared[mppi::math::nearest_multiple_4(std_dev_size)];
  if (control_dim % 4 == 0)
  {
    // Step 1: copy means into shared memory
    for (i = threadIdx.z; i < num_distributions; i += blockDim.z)
    {
      for (j = threadIdx.y; j < num_timesteps; j += blockDim.y)
      {
        const int mean_index = (i * num_timesteps + j) * control_dim;
        float4* mean_shared4 = reinterpret_cast<float4*>(&mean_shared[mean_index]);
        const float4* mean_d4 = reinterpret_cast<const float4*>(&mean_d[mean_index]);
        for (k = threadIdx.x; k < control_dim / 4; k += blockDim.x)
        {
          mean_shared4[k] = mean_d4[k];
        }
      }
    }

    // Step 2: load std_dev to shared memory
    const float4* std_dev_d4 = reinterpret_cast<const float4*>(&std_dev_d[global_std_dev_index]);
    float4* std_dev_shared4 = reinterpret_cast<float4*>(&std_dev_shared[shared_std_dev_index]);
    for (i = threadIdx.x; i < control_dim / 4; i += blockDim.x)
    {
      std_dev_shared4[i] = std_dev_decay * std_dev_d4[i];
    }

    // Step 3: load noise into shared memory
    float4* control_samples_shared4 = reinterpret_cast<float4*>(&control_samples_shared[shared_noise_index]);
    float4* control_samples_d4 = reinterpret_cast<float4*>(&control_samples_d[global_noise_index]);
    // Create const pointre to mean in shared memory as it shouldn't change henceforth
    const float4* mean_shared4 = reinterpret_cast<const float4*>(&mean_shared[shared_mean_index]);
    for (i = 0; valid_index && i < control_dim / 4; i++)
    {
      control_samples_shared4[i] = control_samples_d4[i];
    }

    __syncthreads();  // wait for all copying from global to shared memory to finish
    // Step 4: do mean + variance calculations
    if (valid_index && (trajectory_index == 0 || time_index < optimization_stride))
    {  // 0 noise trajectory
      for (i = 0; i < control_dim / 4; i++)
      {
        control_samples_shared4[i] = mean_shared4[i];
      }
    }
    else if (valid_index && trajectory_index >= (1.0f - pure_noise_percentage) * num_rollouts)
    {  // doing zero mean trajectories
      for (i = 0; i < control_dim / 4; i++)
      {
        control_samples_shared4[i] = std_dev_shared4[i] * control_samples_shared4[i];
      }
    }
    else if (valid_index)
    {
      for (i = 0; i < control_dim / 4; i++)
      {
        control_samples_shared4[i] = mean_shared4[i] + std_dev_shared4[i] * control_samples_shared4[i];
      }
    }

    // save back to global memory
    for (i = 0; valid_index && i < control_dim / 4; i++)
    {
      control_samples_d4[i] = control_samples_shared4[i];
    }
  }
  else if (control_dim % 2 == 0)
  {
    // Step 1: copy means into shared memory
    for (i = threadIdx.z; i < num_distributions; i += blockDim.z)
    {
      for (j = threadIdx.y; j < num_timesteps; j += blockDim.y)
      {
        const int mean_index = (i * num_timesteps + j) * control_dim;
        float2* mean_shared2 = reinterpret_cast<float2*>(&mean_shared[mean_index]);
        const float2* mean_d2 = reinterpret_cast<const float2*>(&mean_d[mean_index]);
        for (k = threadIdx.x; k < control_dim / 2; k += blockDim.x)
        {
          mean_shared2[k] = mean_d2[k];
        }
      }
    }

    // Step 2: load std_dev to shared memory
    const float2* std_dev_d2 = reinterpret_cast<const float2*>(&std_dev_d[global_std_dev_index]);
    float2* std_dev_shared2 = reinterpret_cast<float2*>(&std_dev_shared[shared_std_dev_index]);
    for (i = threadIdx.x; i < control_dim / 2; i += blockDim.x)
    {
      std_dev_shared2[i] = std_dev_decay * std_dev_d2[i];
    }

    // Step 3: load noise into shared memory
    float2* control_samples_shared2 = reinterpret_cast<float2*>(&control_samples_shared[shared_noise_index]);
    float2* control_samples_d2 = reinterpret_cast<float2*>(&control_samples_d[global_noise_index]);
    // Create const pointer to mean in shared memory as it shouldn't change henceforth
    const float2* mean_shared2 = reinterpret_cast<const float2*>(&mean_shared[shared_mean_index]);
    for (i = 0; valid_index && i < control_dim / 2; i++)
    {
      control_samples_shared2[i] = control_samples_d2[i];
    }

    __syncthreads();  // wait for all copying from global to shared memory to finish
    // Step 4: do mean + variance calculations
    if (valid_index && (trajectory_index == 0 || time_index < optimization_stride))
    {  // 0 noise trajectory
      for (i = 0; i < control_dim / 2; i++)
      {
        control_samples_shared2[i] = mean_shared2[i];
      }
    }
    else if (valid_index && trajectory_index >= (1.0f - pure_noise_percentage) * num_rollouts)
    {  // doing zero mean trajectories
      for (i = 0; i < control_dim / 2; i++)
      {
        control_samples_shared2[i] = std_dev_shared2[i] * control_samples_shared2[i];
      }
    }
    else if (valid_index)
    {
      for (i = 0; i < control_dim / 2; i++)
      {
        control_samples_shared2[i] = mean_shared2[i] + std_dev_shared2[i] * control_samples_shared2[i];
      }
    }

    // save back to global memory
    for (i = 0; valid_index && i < control_dim / 2; i++)
    {
      control_samples_d2[i] = control_samples_shared2[i];
    }
  }
  else
  {  // No memory alignment to take advantage of
    // Step 1: copy means into shared memory
    for (i = threadIdx.z; i < num_distributions; i += blockDim.z)
    {
      for (j = threadIdx.y; j < num_timesteps; j += blockDim.y)
      {
        const int mean_index = (i * num_timesteps + j) * control_dim;
        for (k = threadIdx.x; k < control_dim; k += blockDim.x)
        {
          mean_shared[mean_index + k] = mean_d[mean_index + k];
        }
      }
    }

    // Step 2: load std_dev to shared memory
    for (i = threadIdx.x; i < control_dim; i += blockDim.x)
    {
      std_dev_shared[shared_std_dev_index + i] = std_dev_decay * std_dev_d[global_std_dev_index + i];
    }

    // Step 3: load noise into shared memory
    for (i = 0; valid_index && i < control_dim; i++)
    {
      control_samples_shared[shared_noise_index + i] = control_samples_d[global_noise_index + i];
    }

    __syncthreads();  // wait for all copying from global to shared memory to finish
    // Step 4: do mean + variance calculations
    if (valid_index && (trajectory_index == 0 || time_index < optimization_stride))
    {  // 0 noise trajectory
      for (i = 0; i < control_dim; i++)
      {
        control_samples_shared[shared_noise_index + i] = mean_shared[shared_mean_index + i];
      }
    }
    else if (valid_index && trajectory_index >= (1.0f - pure_noise_percentage) * num_rollouts)
    {  // doing zero mean trajectories
      for (i = 0; i < control_dim; i++)
      {
        control_samples_shared[shared_noise_index + i] =
            std_dev_shared[shared_std_dev_index + i] * control_samples_shared[shared_noise_index + i];
      }
    }
    else if (valid_index)
    {
      for (i = 0; i < control_dim; i++)
      {
        control_samples_shared[shared_noise_index + i] =
            mean_shared[shared_mean_index + i] +
            std_dev_shared[shared_std_dev_index + i] * control_samples_shared[shared_noise_index + i];
      }
    }
    if (valid_index && min_control_d != nullptr && max_control_d != nullptr)
    {
      const int global_bounds_index = min(distribution_index, num_distributions) * control_dim;
      for (i = 0; i < control_dim; i++)
      {
        const float v = control_samples_shared[shared_noise_index + i];
        const float lo = min_control_d[global_bounds_index + i];
        const float hi = max_control_d[global_bounds_index + i];
        control_samples_shared[shared_noise_index + i] = fminf(fmaxf(v, lo), hi);
      }
    }
    __syncthreads();
    // save back to global memory
    for (i = 0; valid_index && i < control_dim; i++)
    {
      control_samples_d[global_noise_index + i] = control_samples_shared[shared_noise_index + i];
    }
  }
}

GAUSSIAN_TEMPLATE
GAUSSIAN_CLASS::GaussianDistributionImpl(cudaStream_t stream) : Managed(stream)
{
  this->SHARED_MEM_REQUEST_GRD_BYTES = sizeof(SAMPLING_PARAMS_T);
  this->SHARED_MEM_REQUEST_BLK_BYTES = 0;
}

GAUSSIAN_TEMPLATE
GAUSSIAN_CLASS::GaussianDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream)
  : params_{ params }, Managed(stream)
{
  this->SHARED_MEM_REQUEST_GRD_BYTES = sizeof(SAMPLING_PARAMS_T);
  this->SHARED_MEM_REQUEST_BLK_BYTES = 0;
}

/*************************************
 * Methods that used to live in the generic SamplingDistribution CRTP base class
 *************************************/

GAUSSIAN_TEMPLATE
void GAUSSIAN_CLASS::GPUSetup()
{
  if (!GPUMemStatus_)
  {
    sampling_d_ = Managed::GPUSetup<DEVICE_SELF_T>(static_cast<DEVICE_SELF_T*>(this));
    allocateCUDAMemory();
    resizeVisualizationControlTrajectories(true);
  }
  else
  {
    this->logger_->debug("%s: GPU Memory already set.\n", getSamplingDistributionName().c_str());
  }
  paramsToDevice();
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::resizeVisualizationControlTrajectories(bool synchronize)
{
  if (GPUMemStatus_)
  {
    if (vis_control_samples_d_)
    {  // deallocate previous memory for control samples
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(vis_control_samples_d_, vis_stream_));
#else
      HANDLE_ERROR(cudaFree(vis_control_samples_d_));
#endif
    }
    if (this->getNumVisRollouts() == 0)
    {  // No need to allocate memory if it will end up being zero
      vis_control_samples_d_ = nullptr;
    }
    else
    {
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaMallocAsync((void**)&vis_control_samples_d_,
                                   sizeof(float) * this->getNumDistributions() * this->getNumVisRollouts() *
                                       this->getNumTimesteps() * CONTROL_DIM,
                                   vis_stream_));
#else
      HANDLE_ERROR(cudaMalloc((void**)&vis_control_samples_d_, sizeof(float) * this->getNumDistributions() *
                                                                   this->getNumVisRollouts() * this->getNumTimesteps() *
                                                                   CONTROL_DIM));
#endif
    }
    HANDLE_ERROR(cudaMemcpyAsync(&sampling_d_->vis_control_samples_d_, &vis_control_samples_d_, sizeof(float*),
                                 cudaMemcpyHostToDevice, vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(vis_stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::allocateCUDAMemory(bool synchronize)
{
  if (GPUMemStatus_)
  {
    if (control_samples_d_)
    {  // deallocate previous memory for control samples
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(control_samples_d_, stream_));
#else
      HANDLE_ERROR(cudaFree(control_samples_d_));
#endif
    }
    if (cached_raw_noise_d_)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(cached_raw_noise_d_, stream_));
#else
      HANDLE_ERROR(cudaFree(cached_raw_noise_d_));
#endif
    }
    const size_t control_samples_bytes = sizeof(float) * this->getNumDistributions() * this->getNumRollouts() *
                                          this->getNumTimesteps() * CONTROL_DIM;
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
    HANDLE_ERROR(cudaMallocAsync((void**)&control_samples_d_, control_samples_bytes, stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&cached_raw_noise_d_, control_samples_bytes, stream_));
#else
    HANDLE_ERROR(cudaMalloc((void**)&control_samples_d_, control_samples_bytes));
    HANDLE_ERROR(cudaMalloc((void**)&cached_raw_noise_d_, control_samples_bytes));
#endif
    HANDLE_ERROR(cudaMemcpyAsync(&sampling_d_->control_samples_d_, &control_samples_d_, sizeof(float*),
                                 cudaMemcpyHostToDevice, stream_));
  }
  // Allocate Gaussian-specific memory (std_dev / means) directly, no CRTP dispatch needed anymore
  allocateCUDAMemoryHelper();
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::freeCudaMem()
{
  if (GPUMemStatus_)
  {
    HANDLE_ERROR(cudaFree(sampling_d_));
    HANDLE_ERROR(cudaFree(control_samples_d_));
    HANDLE_ERROR(cudaFree(cached_raw_noise_d_));
    HANDLE_ERROR(cudaFree(vis_control_samples_d_));
    HANDLE_ERROR(cudaFree(control_means_d_));
    HANDLE_ERROR(cudaFree(std_dev_d_));
    HANDLE_ERROR(cudaFree(min_control_d_));
    HANDLE_ERROR(cudaFree(max_control_d_));
    GPUMemStatus_ = false;
    sampling_d_ = nullptr;
    control_samples_d_ = nullptr;
    cached_raw_noise_d_ = nullptr;
    vis_control_samples_d_ = nullptr;
    control_means_d_ = nullptr;
    std_dev_d_ = nullptr;
    min_control_d_ = nullptr;
    max_control_d_ = nullptr;
  }
}

GAUSSIAN_TEMPLATE
__device__ void GAUSSIAN_CLASS::initializeDistributions(const float* __restrict__ output, const float t_0,
                                                        const float dt, float* __restrict__ theta_d)
{
  SAMPLING_PARAMS_T* shared = reinterpret_cast<SAMPLING_PARAMS_T*>(theta_d);
  *shared = this->params_;
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::paramsToDevice(bool synchronize)
{
  if (GPUMemStatus_)
  {
    HANDLE_ERROR(
        cudaMemcpyAsync(&sampling_d_->params_, &params_, sizeof(SAMPLING_PARAMS_T), cudaMemcpyHostToDevice, stream_));
    if (this->params_.time_specific_std_dev)
    {
      HANDLE_ERROR(cudaMemcpyAsync(this->std_dev_d_, this->params_.std_dev,
                                   sizeof(float) * CONTROL_DIM * this->getNumTimesteps() * this->getNumDistributions(),
                                   cudaMemcpyHostToDevice, this->stream_));
    }
    else
    {
      HANDLE_ERROR(cudaMemcpyAsync(this->std_dev_d_, this->params_.std_dev,
                                   sizeof(float) * CONTROL_DIM * this->getNumDistributions(), cudaMemcpyHostToDevice,
                                   this->stream_));
    }
    HANDLE_ERROR(cudaMemcpyAsync(this->min_control_d_, this->params_.min_control,
                                 sizeof(float) * CONTROL_DIM * this->getNumDistributions(), cudaMemcpyHostToDevice,
                                 this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->max_control_d_, this->params_.max_control,
                                 sizeof(float) * CONTROL_DIM * this->getNumDistributions(), cudaMemcpyHostToDevice,
                                 this->stream_));
    if (synchronize)
    {
      HANDLE_ERROR(cudaStreamSynchronize(stream_));
    }
  }
}

GAUSSIAN_TEMPLATE
__device__ void GAUSSIAN_CLASS::readControlSample(const int& sample_index, const int& t,
                                                  const int& distribution_index, float* __restrict__ control,
                                                  float* __restrict__ theta_d, const int& block_size,
                                                  const int& thread_index, const float* __restrict__ output)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_index >= params_p->num_distributions ? 0 : distribution_index;
  const int control_index =
      ((params_p->num_rollouts * distribution_i + sample_index) * params_p->num_timesteps + t) * CONTROL_DIM;
  if (CONTROL_DIM % 4 == 0)
  {
    float4* u4 = reinterpret_cast<float4*>(control);
    const float4* u4_d = reinterpret_cast<const float4*>(&(this->control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM / 4; i += block_size)
    {
      u4[i] = u4_d[i];
    }
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    float2* u2 = reinterpret_cast<float2*>(control);
    const float2* u2_d = reinterpret_cast<const float2*>(&(this->control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM / 2; i += block_size)
    {
      u2[i] = u2_d[i];
    }
  }
  else
  {
    float* u = reinterpret_cast<float*>(control);
    const float* u_d = reinterpret_cast<const float*>(&(this->control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM; i += block_size)
    {
      u[i] = u_d[i];
    }
  }
}

GAUSSIAN_TEMPLATE
__device__ void GAUSSIAN_CLASS::readVisControlSample(const int& sample_index, const int& t,
                                                     const int& distribution_index, float* __restrict__ control,
                                                     float* __restrict__ theta_d, const int& block_size,
                                                     const int& thread_index, const float* __restrict__ output)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_index >= params_p->num_distributions ? 0 : distribution_index;
  const int control_index =
      ((params_p->num_visualization_rollouts * distribution_i + sample_index) * params_p->num_timesteps + t) *
      CONTROL_DIM;
  if (CONTROL_DIM % 4 == 0)
  {
    float4* u4 = reinterpret_cast<float4*>(control);
    const float4* u4_d = reinterpret_cast<const float4*>(&(this->vis_control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM / 4; i += block_size)
    {
      u4[i] = u4_d[i];
    }
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    float2* u2 = reinterpret_cast<float2*>(control);
    const float2* u2_d = reinterpret_cast<const float2*>(&(this->vis_control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM / 2; i += block_size)
    {
      u2[i] = u2_d[i];
    }
  }
  else
  {
    float* u = reinterpret_cast<float*>(control);
    const float* u_d = reinterpret_cast<const float*>(&(this->vis_control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM; i += block_size)
    {
      u[i] = u_d[i];
    }
  }
}

GAUSSIAN_TEMPLATE
__host__ __device__ float* GAUSSIAN_CLASS::getControlSample(const int& sample_index, const int& t,
                                                             const int& distribution_index,
                                                             const float* __restrict__ theta_d,
                                                             const float* __restrict__ output)
{
#ifdef __CUDA_ARCH__
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
#else
  SAMPLING_PARAMS_T* params_p = &this->params_;
#endif
  const int distribution_i = distribution_index >= params_p->num_distributions ? 0 : distribution_index;
  return &this->control_samples_d_[((params_p->num_rollouts * distribution_i + sample_index) * params_p->num_timesteps +
                                    t) *
                                   CONTROL_DIM];
}

GAUSSIAN_TEMPLATE
__host__ __device__ float* GAUSSIAN_CLASS::getVisControlSample(const int& sample_index, const int& t,
                                                                const int& distribution_index,
                                                                const float* __restrict__ theta_d,
                                                                const float* __restrict__ output)
{
#ifdef __CUDA_ARCH__
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
#else
  SAMPLING_PARAMS_T* params_p = &this->params_;
#endif
  const int distribution_i = distribution_index >= params_p->num_distributions ? 0 : distribution_index;
  return &this->vis_control_samples_d_
              [((params_p->num_visualization_rollouts * distribution_i + sample_index) * params_p->num_timesteps + t) *
               CONTROL_DIM];
}

GAUSSIAN_TEMPLATE
__device__ void GAUSSIAN_CLASS::writeControlSample(const int& sample_index, const int& t,
                                                    const int& distribution_index, const float* __restrict__ control,
                                                    float* __restrict__ theta_d, const int& block_size,
                                                    const int& thread_index, const float* __restrict__ output)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_index >= params_p->num_distributions ? 0 : distribution_index;
  const int control_index =
      ((params_p->num_rollouts * distribution_i + sample_index) * params_p->num_timesteps + t) * CONTROL_DIM;
  if (CONTROL_DIM % 4 == 0)
  {
    const float4* u4 = reinterpret_cast<const float4*>(control);
    float4* u4_d = reinterpret_cast<float4*>(&(this->control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM / 4; i += block_size)
    {
      u4_d[i] = u4[i];
    }
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    const float2* u2 = reinterpret_cast<const float2*>(control);
    float2* u2_d = reinterpret_cast<float2*>(&(this->control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM / 2; i += block_size)
    {
      u2_d[i] = u2[i];
    }
  }
  else
  {
    const float* u = reinterpret_cast<const float*>(control);
    float* u_d = reinterpret_cast<float*>(&(this->control_samples_d_[control_index]));
    for (int i = thread_index; i < CONTROL_DIM; i += block_size)
    {
      u_d[i] = u[i];
    }
  }
}

// By default, call the update from device method by first putting the weights into gpu memory
GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::updateDistributionParamsFromHost(
    const Eigen::Ref<const Eigen::MatrixXf>& trajectory_weights, float normalizer, const int& distribution_i,
    bool synchronize)
{
  float* trajectory_weights_d;
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
  HANDLE_ERROR(cudaMallocAsync((void**)&trajectory_weights_d, sizeof(float) * this->getNumRollouts(), this->stream_));
#else
  HANDLE_ERROR(cudaMalloc((void**)&trajectory_weights_d, sizeof(float) * this->getNumRollouts()));
#endif
  HANDLE_ERROR(cudaMemcpyAsync(trajectory_weights_d, trajectory_weights.data(), sizeof(float) * this->getNumRollouts(),
                               cudaMemcpyHostToDevice, this->stream_));
  updateDistributionParamsFromDevice(trajectory_weights_d, normalizer, distribution_i, false);
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
  HANDLE_ERROR(cudaFreeAsync(trajectory_weights_d, this->stream_));
#else
  HANDLE_ERROR(cudaFree(trajectory_weights_d));
#endif

  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

/*************************************
 * Gaussian-specific methods
 *************************************/

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::allocateCUDAMemoryHelper()
{
  if (this->GPUMemStatus_)
  {
    if (std_dev_d_)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(std_dev_d_, this->stream_));
#else
      HANDLE_ERROR(cudaFree(std_dev_d_));
#endif
    }
    if (control_means_d_)
    {  // deallocate previous memory control trajectory means
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(control_means_d_, this->stream_));
#else
      HANDLE_ERROR(cudaFree(control_means_d_));
#endif
    }
    if (min_control_d_)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(min_control_d_, this->stream_));
#else
      HANDLE_ERROR(cudaFree(min_control_d_));
#endif
    }
    if (max_control_d_)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
      HANDLE_ERROR(cudaFreeAsync(max_control_d_, this->stream_));
#else
      HANDLE_ERROR(cudaFree(max_control_d_));
#endif
    }

    int std_dev_size = CONTROL_DIM * this->getNumDistributions();
    if (this->params_.time_specific_std_dev)
    {
      std_dev_size *= this->getNumTimesteps();
    }
    const int bounds_size = CONTROL_DIM * this->getNumDistributions();
#if defined(CUDART_VERSION) && CUDART_VERSION > 11200
    HANDLE_ERROR(cudaMallocAsync((void**)&std_dev_d_, sizeof(float) * std_dev_size, this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&control_means_d_,
                                 sizeof(float) * this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM,
                                 this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&min_control_d_, sizeof(float) * bounds_size, this->stream_));
    HANDLE_ERROR(cudaMallocAsync((void**)&max_control_d_, sizeof(float) * bounds_size, this->stream_));
#else
    HANDLE_ERROR(cudaMalloc((void**)&std_dev_d_, sizeof(float) * std_dev_size));
    HANDLE_ERROR(cudaMalloc((void**)&control_means_d_,
                            sizeof(float) * this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&min_control_d_, sizeof(float) * bounds_size));
    HANDLE_ERROR(cudaMalloc((void**)&max_control_d_, sizeof(float) * bounds_size));
#endif
    means_.resize(this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM);
    // Ensure that the device side point knows where the the standard deviation memory is located
    HANDLE_ERROR(cudaMemcpyAsync(&this->sampling_d_->std_dev_d_, &std_dev_d_, sizeof(float*), cudaMemcpyHostToDevice,
                                 this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(&this->sampling_d_->control_means_d_, &control_means_d_, sizeof(float*),
                                 cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(&this->sampling_d_->min_control_d_, &min_control_d_, sizeof(float*),
                                 cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(cudaMemcpyAsync(&this->sampling_d_->max_control_d_, &max_control_d_, sizeof(float*),
                                 cudaMemcpyHostToDevice, this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::generateSamples(const int& optimization_stride, const int& iteration_num,
                                              curandGenerator_t& gen, bool synchronize, bool regenerate_noise)
{
  const size_t total_samples_bytes =
      sizeof(float) * this->getNumDistributions() * this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM;
  if (regenerate_noise)
  {
    if (this->params_.use_same_noise_for_all_distributions)
    {
      HANDLE_CURAND_ERROR(curandGenerateNormal(
          gen, this->control_samples_d_, this->getNumTimesteps() * this->getNumRollouts() * CONTROL_DIM, 0.0f, 1.0f));
      for (int i = 1; i < this->getNumDistributions(); i++)
      {
        HANDLE_ERROR(cudaMemcpyAsync(
            &this->control_samples_d_[this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM * i],
            this->control_samples_d_, sizeof(float) * this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM,
            cudaMemcpyDeviceToDevice, this->stream_));
      }
    }
    else
    {
      HANDLE_CURAND_ERROR(curandGenerateNormal(
          gen, this->control_samples_d_,
          this->getNumTimesteps() * this->getNumRollouts() * this->getNumDistributions() * CONTROL_DIM, 0.0f, 1.0f));
    }
    HANDLE_ERROR(cudaMemcpyAsync(this->cached_raw_noise_d_, this->control_samples_d_, total_samples_bytes,
                                 cudaMemcpyDeviceToDevice, this->stream_));
  }
  else
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->control_samples_d_, this->cached_raw_noise_d_, total_samples_bytes,
                                 cudaMemcpyDeviceToDevice, this->stream_));
  }
  const int BLOCKSIZE_X = this->params_.rewrite_controls_block_dim.x;
  const int BLOCKSIZE_Y = this->params_.rewrite_controls_block_dim.y;
  const int BLOCKSIZE_Z = this->params_.rewrite_controls_block_dim.z;
  /**
   * Generate noise samples with mean added
   **/
  dim3 control_writing_grid;
  control_writing_grid.x = mppi::math::int_ceil(this->getNumRollouts(), BLOCKSIZE_X);
  control_writing_grid.y = mppi::math::int_ceil(this->getNumTimesteps(), BLOCKSIZE_Y);
  control_writing_grid.z = mppi::math::int_ceil(this->getNumDistributions(), BLOCKSIZE_Z);
  unsigned int std_dev_mem_size = this->getNumDistributions() * CONTROL_DIM;
  // Allocate shared memory for std_deviations per timestep or constant across the trajectory
  std_dev_mem_size = mppi::math::nearest_multiple_4(
      this->params_.time_specific_std_dev ? std_dev_mem_size * this->getNumTimesteps() : std_dev_mem_size);
  unsigned int shared_mem_size =
      std_dev_mem_size +
      mppi::math::nearest_multiple_4(this->getNumDistributions() * this->getNumTimesteps() * CONTROL_DIM) +
      mppi::math::nearest_multiple_4(BLOCKSIZE_X * BLOCKSIZE_Y * BLOCKSIZE_Z * CONTROL_DIM);
  shared_mem_size *= sizeof(float);
  setGaussianControls<<<control_writing_grid, this->params_.rewrite_controls_block_dim, shared_mem_size,
                        this->stream_>>>(
      this->control_means_d_, this->std_dev_d_, this->control_samples_d_, CONTROL_DIM, this->getNumTimesteps(),
      this->getNumRollouts(), this->getNumDistributions(), optimization_stride,
      powf(this->params_.std_dev_decay, iteration_num), this->params_.pure_noise_trajectories_percentage,
      this->params_.time_specific_std_dev, this->min_control_d_, this->max_control_d_);

  HANDLE_ERROR(cudaGetLastError());
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::updateDistributionParamsFromDevice(const float* trajectory_weights_d, float normalizer,
                                                                 const int& distribution_i, bool synchronize)
{
  if (distribution_i >= this->getNumDistributions())
  {
    this->logger_->error(
        "Updating distributional params for distribution %d out of %d total. Distribution out of bounds.\n",
        distribution_i, this->getNumDistributions());
    return;
  }
  float* control_samples_i_d =
      &(this->control_samples_d_[distribution_i * this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM]);
  float* control_mean_i_d = &(this->control_means_d_[distribution_i * this->getNumTimesteps() * CONTROL_DIM]);
  mppi::kernels::launchWeightedReductionKernel<CONTROL_DIM>(trajectory_weights_d, control_samples_i_d, control_mean_i_d,
                                                            normalizer, this->getNumTimesteps(), this->getNumRollouts(),
                                                            this->params_.sum_strides, this->stream_, synchronize);
  HANDLE_ERROR(cudaMemcpyAsync(&means_[distribution_i * this->getNumTimesteps() * CONTROL_DIM], control_mean_i_d,
                               sizeof(float) * this->getNumTimesteps() * CONTROL_DIM, cudaMemcpyDeviceToHost,
                               this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::setHostOptimalControlSequence(float* optimal_control_trajectory,
                                                            const int& distribution_i, bool synchronize)
{
  if (distribution_i >= this->getNumDistributions())
  {
    this->logger_->error(
        "Asking for optimal control sequence from distribution %d out of %d total. Distribution out of bounds.\n",
        distribution_i, this->getNumDistributions());
    return;
  }

  HANDLE_ERROR(cudaMemcpyAsync(
      optimal_control_trajectory, &(this->control_means_d_[this->getNumTimesteps() * CONTROL_DIM * distribution_i]),
      sizeof(float) * this->getNumTimesteps() * CONTROL_DIM, cudaMemcpyDeviceToHost, this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

GAUSSIAN_TEMPLATE
__host__ __device__ float GAUSSIAN_CLASS::computeLikelihoodRatioCost(const float* __restrict__ u,
                                                                     float* __restrict__ theta_d,
                                                                     const int sample_index, const int t,
                                                                     const int distribution_idx, const float lambda,
                                                                     const float alpha)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_idx >= params_p->num_distributions ? 0 : distribution_idx;
  float* std_dev = &(params_p->std_dev[CONTROL_DIM * distribution_i]);
  if (params_p->time_specific_std_dev)
  {
    std_dev = &(params_p->std_dev[(distribution_i * params_p->num_timesteps + t) * CONTROL_DIM]);
  }
  float* mean = &(this->control_means_d_[(params_p->num_timesteps * distribution_i + t) * CONTROL_DIM]);
  float* control_cost_coeff = params_p->control_cost_coeff;

  float cost = 0.0f;
#ifdef __CUDA_ARCH__
  int i = threadIdx.y;
  int step = blockDim.y;
#else
  int i = 0;
  int step = 1;
#endif

  if (CONTROL_DIM % 4 == 0)
  {
    float4 cost_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 mean_i, std_dev_i, u_i, control_cost_coeff_i;
    for (; i < CONTROL_DIM / 4; i += step)
    {
      if (sample_index >= (1.0f - params_p->pure_noise_trajectories_percentage) * params_p->num_rollouts)
      {
        mean_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
      else
      {
        mean_i = reinterpret_cast<float4*>(mean)[i];  // read mean value from global memory only once
      }
      std_dev_i = reinterpret_cast<float4*>(std_dev)[i];
      u_i = reinterpret_cast<const float4*>(u)[i];
      control_cost_coeff_i = reinterpret_cast<float4*>(control_cost_coeff)[i];
      cost_i += control_cost_coeff_i * mean_i * (mean_i - 2.0f * u_i) / (std_dev_i * std_dev_i);  // Proper way
    }
    cost += cost_i.x + cost_i.y + cost_i.z + cost_i.w;
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    float2 cost_i = make_float2(0.0f, 0.0f);
    float2 mean_i, std_dev_i, u_i, control_cost_coeff_i;
    for (; i < CONTROL_DIM / 2; i += step)
    {
      if (sample_index >= (1.0f - params_p->pure_noise_trajectories_percentage) * params_p->num_rollouts)
      {
        mean_i = make_float2(0.0f, 0.0f);
      }
      else
      {
        mean_i = reinterpret_cast<float2*>(mean)[i];  // read mean value from global memory only once
      }
      std_dev_i = reinterpret_cast<float2*>(std_dev)[i];
      u_i = reinterpret_cast<const float2*>(u)[i];
      control_cost_coeff_i = reinterpret_cast<float2*>(control_cost_coeff)[i];
      cost_i += control_cost_coeff_i * mean_i * (mean_i - 2.0f * u_i) / (std_dev_i * std_dev_i);  // Proper way
    }
    cost += cost_i.x + cost_i.y;
  }
  else
  {
    float mean_i;
    for (; i < CONTROL_DIM; i += step)
    {
      if (sample_index >= (1.0f - params_p->pure_noise_trajectories_percentage) * params_p->num_rollouts)
      {
        mean_i = 0.0f;
      }
      else
      {
        mean_i = mean[i];  // read mean value from global memory only once
      }
      cost += control_cost_coeff[i] * mean_i * (mean_i - 2.0f * u[i]) / (std_dev[i] * std_dev[i]);  // Proper way
    }
  }
  return 0.5f * lambda * (1.0f - alpha) * cost;
}

GAUSSIAN_TEMPLATE
__host__ __device__ float GAUSSIAN_CLASS::computeFeedbackCost(const float* __restrict__ u_fb,
                                                              float* __restrict__ theta_d, const int t,
                                                              const int distribution_idx, const float lambda,
                                                              const float alpha)
{
  SAMPLING_PARAMS_T* params_p = (SAMPLING_PARAMS_T*)theta_d;
  const int distribution_i = distribution_idx >= params_p->num_distributions ? 0 : distribution_idx;
  float* std_dev = &(params_p->std_dev[CONTROL_DIM * distribution_i]);
  if (params_p->time_specific_std_dev)
  {
    std_dev = &(params_p->std_dev[(distribution_i * params_p->num_timesteps + t) * CONTROL_DIM]);
  }
  float* control_cost_coeff = params_p->control_cost_coeff;

  float cost = 0.0f;
#ifdef __CUDA_ARCH__
  int i = threadIdx.y;
  int step = blockDim.y;
#else
  int i = 0;
  int step = 1;
#endif

  if (CONTROL_DIM % 4 == 0)
  {
    float4 cost_i = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 std_dev_i, control_cost_coeff_i, u_fb_i;
    for (; i < CONTROL_DIM / 4; i += step)
    {
      std_dev_i = reinterpret_cast<float4*>(std_dev)[i];
      u_fb_i = reinterpret_cast<const float4*>(u_fb)[i];
      control_cost_coeff_i = reinterpret_cast<float4*>(control_cost_coeff)[i];
      cost_i += control_cost_coeff_i * (u_fb_i * u_fb_i) / (std_dev_i * std_dev_i);
    }
    cost += cost_i.x + cost_i.y + cost_i.z + cost_i.w;
  }
  else if (CONTROL_DIM % 2 == 0)
  {
    float2 cost_i = make_float2(0.0f, 0.0f);
    float2 std_dev_i, control_cost_coeff_i, u_fb_i;
    for (; i < CONTROL_DIM / 2; i += step)
    {
      std_dev_i = reinterpret_cast<float2*>(std_dev)[i];
      control_cost_coeff_i = reinterpret_cast<float2*>(control_cost_coeff)[i];
      u_fb_i = reinterpret_cast<const float2*>(u_fb)[i];
      cost_i += control_cost_coeff_i * (u_fb_i * u_fb_i) / (std_dev_i * std_dev_i);
    }
    cost += cost_i.x + cost_i.y;
  }
  else
  {
    for (; i < CONTROL_DIM; i += step)
    {
      cost += control_cost_coeff[i] * (u_fb[i] * u_fb[i]) / (std_dev[i] * std_dev[i]);
    }
  }
  return 0.5f * lambda * (1.0f - alpha) * cost;
}

GAUSSIAN_TEMPLATE
__host__ float GAUSSIAN_CLASS::computeLikelihoodRatioCost(const Eigen::Ref<const control_array>& u, const int t,
                                                          const int distribution_idx, const float lambda,
                                                          const float alpha)
{
  float cost = 0.0f;
  const int distribution_i = distribution_idx >= this->params_.num_distributions ? 0 : distribution_idx;
  const int mean_index = (distribution_i * this->getNumTimesteps() + t) * CONTROL_DIM;
  float* mean = &(this->means_[mean_index]);
  float* std_dev = &(this->params_.std_dev[CONTROL_DIM * distribution_i]);
  if (this->params_.time_specific_std_dev)
  {
    std_dev = &(this->params_.std_dev[(distribution_i * this->getNumTimesteps() + t) * CONTROL_DIM]);
  }
  for (int i = 0; i < CONTROL_DIM; i++)
  {
    cost += this->params_.control_cost_coeff[i] * mean[i] * (mean[i] + 2.0f * u(i)) /
            (std_dev[i] * std_dev[i]);  // Proper way
  }
  return cost;
}

GAUSSIAN_TEMPLATE
__host__ void GAUSSIAN_CLASS::copyImportanceSamplerToDevice(const float* importance_sampler,
                                                            const int& distribution_idx, bool synchronize)
{
  HANDLE_ERROR(cudaMemcpyAsync(&control_means_d_[this->getNumTimesteps() * CONTROL_DIM * distribution_idx],
                               importance_sampler, sizeof(float) * this->getNumTimesteps() * CONTROL_DIM,
                               cudaMemcpyHostToDevice, this->stream_));
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->stream_));
  }
}

#undef GAUSSIAN_TEMPLATE
#undef GAUSSIAN_CLASS

}  // namespace sampling_distributions
}  // namespace mppi
