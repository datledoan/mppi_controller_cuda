#ifndef COLORED_NOISE_CUH
#define COLORED_NOISE_CUH

#include "mppi_controller_cuda/sampling_distribution/gaussian.cuh"
#include <cufft.h>
#include <curand.h>
#include <vector>

__global__ void configureFrequencyNoise(cufftComplex* noise, float* variance, int num_samples, int control_dim,
                                        int num_freq);
__global__ void rearrangeNoise(float* input, float* output, float* variance, int num_trajectories, int num_timesteps,
                               int control_dim, int offset_t, float decay_rate = 0.0f);

inline void fftfreq(const int num_samples, std::vector<float>& result, const float spacing = 1)
{
  // result is of size floor(n/2) + 1
  int result_size = num_samples / 2 + 1;
  result.clear();
  result.resize(result_size);
  for (int i = 0; i < result_size; i++)
  {
    result[i] = i / (spacing * num_samples);
  }
}

namespace mppi
{
namespace sampling_distributions
{
template <int C_DIM, int MAX_DISTRIBUTIONS = 2>
struct ColoredNoiseParamsImpl : public GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS>
{
  float exponents[C_DIM * MAX_DISTRIBUTIONS] = { 0.0f };
  float offset_decay_rate = 0.0f;
  float fmin = 0.0f;
  bool use_colored_noise = true;

  ColoredNoiseParamsImpl(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS>(num_rollouts, num_timesteps, num_distributions)
  {
  }
};

template <int C_DIM>
using ColoredNoiseParams = ColoredNoiseParamsImpl<C_DIM, 2>;

/**
 * @brief Colored-noise sampling distribution. Extends this fork's
 * GaussianDistributionImpl, replacing only how raw noise is generated in
 * generateSamples() -- the mean/std_dev combine step (setGaussianControls),
 * likelihood-ratio cost, feedback cost, control-sample read/write, and
 * noise-reuse-across-cycles (cached_raw_noise_d_/regenerate_noise) machinery
 * are all inherited as-is.
 *
 * Passes itself as GaussianDistributionImpl's SELF_T so the inherited
 * GPUSetup()/sampling_d_ (see gaussian.cuh's class comment) correctly mirror
 * a *ColoredNoiseDistributionImpl* on the device, not a sliced base object --
 * required for sampling-distribution-templated kernel launches (which expect
 * exactly this class as their SAMPLING_T) to type-check.
 */
template <template <int> class PARAMS_TEMPLATE = ColoredNoiseParams, class DYN_PARAMS_T = DynamicsParams>
class ColoredNoiseDistributionImpl
  : public GaussianDistributionImpl<PARAMS_TEMPLATE, DYN_PARAMS_T, ColoredNoiseDistributionImpl<PARAMS_TEMPLATE, DYN_PARAMS_T>>
{
public:
  using PARENT_CLASS =
      GaussianDistributionImpl<PARAMS_TEMPLATE, DYN_PARAMS_T, ColoredNoiseDistributionImpl<PARAMS_TEMPLATE, DYN_PARAMS_T>>;
  using SAMPLING_PARAMS_T = typename PARENT_CLASS::SAMPLING_PARAMS_T;
  using control_array = typename PARENT_CLASS::control_array;

  static const int CONTROL_DIM = PARENT_CLASS::CONTROL_DIM;

  ColoredNoiseDistributionImpl(cudaStream_t stream = 0);
  ColoredNoiseDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream = 0);

  ~ColoredNoiseDistributionImpl()
  {
    freeCudaMem();
  }

  __host__ virtual std::string getSamplingDistributionName() const override
  {
    return "Colored Noise";
  }

  __host__ virtual void allocateCUDAMemoryHelper() override;

  __host__ virtual void freeCudaMem() override;

  __host__ virtual void generateSamples(const int& optimization_stride, const int& iteration_num,
                                        curandGenerator_t& gen, bool synchronize = true,
                                        bool regenerate_noise = true) override;

  __host__ __device__ float getOffsetDecayRate() const
  {
    return this->params_.offset_decay_rate;
  }

  void setOffsetDecayRate(const float decay_rate)
  {
    this->params_.offset_decay_rate = decay_rate;
  }

protected:
  // 0 is not a valid cuFFT handle, so this doubles as a "no plan yet" sentinel
  // for the destroy-before-recreate guard in allocateCUDAMemoryHelper().
  cufftHandle plan_ = 0;
  float* frequency_sigma_d_ = nullptr;
  float* noise_in_time_d_ = nullptr;
  cufftComplex* samples_in_freq_complex_d_ = nullptr;
  float* freq_coeffs_d_ = nullptr;
};

// Convenience alias, mirroring gaussian.cuh's GaussianDistribution<DYN_PARAMS_T>
template <class DYN_PARAMS_T>
using ColoredNoiseDistribution = ColoredNoiseDistributionImpl<ColoredNoiseParams, DYN_PARAMS_T>;

}  // namespace sampling_distributions
}  // namespace mppi

#endif  // COLORED_NOISE_CUH
