#include "mppi_controller_cuda/sampling_distribution/colored_noise.cuh"
#include "mppi_controller_cuda/core/mppi_common.cuh"
#include "mppi_controller_cuda/utils/math_utils.h"
#include "mppi_controller_cuda/utils/gpu_err_chk.cuh"

#include <Eigen/Dense>

__global__ void configureFrequencyNoise(cufftComplex* noise, float* variance, int num_samples, int control_dim,
                                        int num_freq)
{
  int sample_index = blockDim.x * blockIdx.x + threadIdx.x;
  int freq_index = blockDim.y * blockIdx.y + threadIdx.y;
  int control_index = blockDim.z * blockIdx.z + threadIdx.z;

  if (sample_index < num_samples && freq_index < num_freq && control_index < control_dim)
  {
    int noise_index = (sample_index * control_dim + control_index) * num_freq + freq_index;
    int variance_index = control_index * num_freq + freq_index;
    noise[noise_index].x *= variance[variance_index];
    if (freq_index == 0)
    {
      noise[noise_index].y = 0;
    }
    else if (num_freq % 2 == 1 && freq_index == num_freq - 1)
    {
      noise[noise_index].y = 0;
    }
    else
    {
      noise[noise_index].y *= variance[variance_index];
    }
  }
}

__global__ void rearrangeNoise(float* input, float* output, float* variance, int num_trajectories, int num_timesteps,
                               int control_dim, int offset_t, float decay_rate)
{
  const int sample_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int time_index = blockIdx.y * blockDim.y + threadIdx.y;
  const int control_index = blockIdx.z * blockDim.z + threadIdx.z;
  const float decayed_offset = decay_rate == 0 ? 0 : powf(decay_rate, time_index);
  if (sample_index < num_trajectories && time_index < num_timesteps && control_index < control_dim)
  {  // cuFFT does not normalize inverse transforms so a division by num_timesteps is required
    output[(sample_index * num_timesteps + time_index) * control_dim + control_index] =
        (input[(sample_index * control_dim + control_index) * 2 * num_timesteps + time_index] -
         input[(sample_index * control_dim + control_index) * 2 * num_timesteps + offset_t] * decayed_offset) /
        (variance[control_index] * 2 * num_timesteps);
  }
}

namespace mppi
{
namespace sampling_distributions
{
#define COLORED_TEMPLATE template <template <int> class PARAMS_TEMPLATE, class DYN_PARAMS_T>
#define COLORED_NOISE ColoredNoiseDistributionImpl<PARAMS_TEMPLATE, DYN_PARAMS_T>

COLORED_TEMPLATE
const int COLORED_NOISE::CONTROL_DIM;

COLORED_TEMPLATE
COLORED_NOISE::ColoredNoiseDistributionImpl(cudaStream_t stream) : PARENT_CLASS(stream)
{
}

COLORED_TEMPLATE
COLORED_NOISE::ColoredNoiseDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream)
  : PARENT_CLASS(params, stream)
{
}

COLORED_TEMPLATE
__host__ void COLORED_NOISE::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    if (freq_coeffs_d_)
    {
      HANDLE_ERROR(cudaFree(freq_coeffs_d_));
    }
    if (samples_in_freq_complex_d_)
    {
      HANDLE_ERROR(cudaFree(samples_in_freq_complex_d_));
    }
    if (noise_in_time_d_)
    {
      HANDLE_ERROR(cudaFree(noise_in_time_d_));
    }
    if (frequency_sigma_d_)
    {
      HANDLE_ERROR(cudaFree(frequency_sigma_d_));
    }
    freq_coeffs_d_ = nullptr;
    samples_in_freq_complex_d_ = nullptr;
    noise_in_time_d_ = nullptr;
    frequency_sigma_d_ = nullptr;
    if (plan_)
    {
      cufftDestroy(plan_);
      plan_ = 0;
    }
  }
  PARENT_CLASS::freeCudaMem();
}

COLORED_TEMPLATE
__host__ void COLORED_NOISE::allocateCUDAMemoryHelper()
{
  PARENT_CLASS::allocateCUDAMemoryHelper();
  if (this->GPUMemStatus_)
  {
    const int sample_num_timesteps = 2 * this->getNumTimesteps();
    const int freq_size = sample_num_timesteps / 2 + 1;
    if (frequency_sigma_d_)
    {
      HANDLE_ERROR(cudaFree(frequency_sigma_d_));
    }
    if (samples_in_freq_complex_d_)
    {
      HANDLE_ERROR(cudaFree(samples_in_freq_complex_d_));
    }
    if (noise_in_time_d_)
    {
      HANDLE_ERROR(cudaFree(noise_in_time_d_));
    }
    if (freq_coeffs_d_)
    {
      HANDLE_ERROR(cudaFree(freq_coeffs_d_));
    }
    HANDLE_ERROR(cudaMalloc((void**)&freq_coeffs_d_, sizeof(float) * freq_size * this->CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&frequency_sigma_d_, sizeof(float) * this->CONTROL_DIM));
    HANDLE_ERROR(cudaMalloc((void**)&samples_in_freq_complex_d_, sizeof(cufftComplex) * this->getNumRollouts() *
                                                                     this->CONTROL_DIM * freq_size *
                                                                     this->getNumDistributions()));
    HANDLE_ERROR(cudaMalloc((void**)&noise_in_time_d_, sizeof(float) * this->getNumRollouts() * this->CONTROL_DIM *
                                                           sample_num_timesteps * this->getNumDistributions()));
    if (plan_)
    {
      cufftDestroy(plan_);
    }
    HANDLE_CUFFT_ERROR(cufftPlan1d(&plan_, sample_num_timesteps, CUFFT_C2R,
                                   this->getNumRollouts() * this->getNumDistributions() * this->CONTROL_DIM));
    HANDLE_CUFFT_ERROR(cufftSetStream(plan_, this->stream_));
  }
}

COLORED_TEMPLATE
__host__ void COLORED_NOISE::generateSamples(const int& optimization_stride, const int& iteration_num,
                                             curandGenerator_t& gen, bool synchronize, bool regenerate_noise)
{
  if (!this->params_.use_colored_noise)
  {
    PARENT_CLASS::generateSamples(optimization_stride, iteration_num, gen, synchronize, regenerate_noise);
    return;
  }

  const size_t total_samples_bytes =
      sizeof(float) * this->getNumDistributions() * this->getNumRollouts() * this->getNumTimesteps() * CONTROL_DIM;

  if (regenerate_noise)
  {
    const int BLOCKSIZE_X = 32;
    const int BLOCKSIZE_Y = 32;
    const int BLOCKSIZE_Z = 1;
    const int num_trajectories = this->getNumRollouts() * this->getNumDistributions();

    std::vector<float> sample_freq;
    const int sample_num_timesteps = 2 * this->getNumTimesteps();
    fftfreq(sample_num_timesteps, sample_freq);
    const float cutoff_freq = fmaxf(this->params_.fmin, 1.0f / sample_num_timesteps);
    const int freq_size = sample_freq.size();

    int smaller_index = 0;
    Eigen::MatrixXf sample_freqs(freq_size, CONTROL_DIM);

    // Weight each frequency bin per control dim by the configured exponent
    // (this->params_.exponents -- 0 == white noise, matching plain Gaussian).
    for (int i = 0; i < freq_size; i++)
    {
      if (sample_freq[i] < cutoff_freq)
      {
        smaller_index++;
      }
      else if (smaller_index < freq_size)
      {
        for (int j = 0; j < smaller_index; j++)
        {
          sample_freq[j] = sample_freq[smaller_index];
          for (int k = 0; k < CONTROL_DIM; k++)
          {
            sample_freqs(j, k) = powf(sample_freq[smaller_index], -this->params_.exponents[k] / 2.0f);
          }
        }
      }
      for (int j = 0; j < CONTROL_DIM; j++)
      {
        sample_freqs(i, j) = powf(sample_freq[i], -this->params_.exponents[j] / 2.0f);
      }
    }

    // Normalization constant per control dim.
    float sigma[CONTROL_DIM] = { 0 };
    for (int i = 0; i < CONTROL_DIM; i++)
    {
      for (int j = 1; j < freq_size - 1; j++)
      {
        sigma[i] += SQ(sample_freqs(j, i));
      }
      sigma[i] += SQ(sample_freqs(freq_size - 1, i) * ((1.0f + (sample_num_timesteps % 2)) / 2.0f));
      sigma[i] = 2.0f * sqrtf(sigma[i]) / sample_num_timesteps;
    }

    const int batch = num_trajectories * CONTROL_DIM;
    HANDLE_CURAND_ERROR(
        curandGenerateNormal(gen, (float*)samples_in_freq_complex_d_, 2 * batch * freq_size, 0.0, 1.0));
    HANDLE_ERROR(cudaMemcpyAsync(freq_coeffs_d_, sample_freqs.data(), sizeof(float) * freq_size * CONTROL_DIM,
                                 cudaMemcpyHostToDevice, this->stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(frequency_sigma_d_, sigma, sizeof(float) * CONTROL_DIM, cudaMemcpyHostToDevice, this->stream_));

    const int trajectories_grid_x = mppi::math::int_ceil(num_trajectories, BLOCKSIZE_X);
    const int freq_grid_y = mppi::math::int_ceil(freq_size, BLOCKSIZE_Y);
    const int control_grid_z = mppi::math::int_ceil(CONTROL_DIM, BLOCKSIZE_Z);
    dim3 grid(trajectories_grid_x, freq_grid_y, control_grid_z);
    dim3 block(BLOCKSIZE_X, BLOCKSIZE_Y, BLOCKSIZE_Z);
    configureFrequencyNoise<<<grid, block, 0, this->stream_>>>(samples_in_freq_complex_d_, freq_coeffs_d_,
                                                               num_trajectories, CONTROL_DIM, freq_size);
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_CUFFT_ERROR(cufftExecC2R(plan_, samples_in_freq_complex_d_, noise_in_time_d_));

    const int time_grid_y = mppi::math::int_ceil(this->getNumTimesteps(), BLOCKSIZE_Y);
    dim3 reorder_grid(trajectories_grid_x, time_grid_y, control_grid_z);
    rearrangeNoise<<<reorder_grid, block, 0, this->stream_>>>(
        noise_in_time_d_, this->control_samples_d_, frequency_sigma_d_, num_trajectories, this->getNumTimesteps(),
        CONTROL_DIM, optimization_stride, this->getOffsetDecayRate());
    HANDLE_ERROR(cudaGetLastError());

    HANDLE_ERROR(cudaMemcpyAsync(this->cached_raw_noise_d_, this->control_samples_d_, total_samples_bytes,
                                 cudaMemcpyDeviceToDevice, this->stream_));
  }
  else
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->control_samples_d_, this->cached_raw_noise_d_, total_samples_bytes,
                                 cudaMemcpyDeviceToDevice, this->stream_));
  }

  // Combine the raw noise now sitting in control_samples_d_ with the current
  // mean/std_dev -- identical to GaussianDistributionImpl::generateSamples().
  const int BLOCKSIZE_X = this->params_.rewrite_controls_block_dim.x;
  const int BLOCKSIZE_Y = this->params_.rewrite_controls_block_dim.y;
  const int BLOCKSIZE_Z = this->params_.rewrite_controls_block_dim.z;
  dim3 control_writing_grid;
  control_writing_grid.x = mppi::math::int_ceil(this->getNumRollouts(), BLOCKSIZE_X);
  control_writing_grid.y = mppi::math::int_ceil(this->getNumTimesteps(), BLOCKSIZE_Y);
  control_writing_grid.z = mppi::math::int_ceil(this->getNumDistributions(), BLOCKSIZE_Z);
  unsigned int std_dev_mem_size = this->getNumDistributions() * CONTROL_DIM;
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

#undef COLORED_TEMPLATE
#undef COLORED_NOISE

}  // namespace sampling_distributions
}  // namespace mppi
