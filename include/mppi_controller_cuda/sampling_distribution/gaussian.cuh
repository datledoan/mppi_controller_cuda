#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

#include <vector>
#include <string>
#include <type_traits>
#include <cfloat>
#include "mppi_controller_cuda/dynamics/dynamics.cuh"
#include "mppi_controller_cuda/utils/managed.cuh"

namespace mppi
{
namespace sampling_distributions
{
template <int C_DIM>
struct alignas(float4) SamplingParams
{
  static const int CONTROL_DIM = C_DIM;
  bool use_same_noise_for_all_distributions = true;
  int num_rollouts = 1;
  int num_timesteps = 1;
  int num_distributions = 1;
  int num_visualization_rollouts = 0;
  SamplingParams(int num_rollouts, int num_timesteps, int num_distributions = 1)
    : num_rollouts{ num_rollouts }, num_timesteps{ num_timesteps }, num_distributions{ num_distributions }
  {
  }
  SamplingParams() = default;
};

__global__ void setGaussianControls(const float* __restrict__ mean_d, const float* __restrict__ std_dev_d,
                                    float* __restrict__ control_samples_d, const int control_dim,
                                    const int num_timesteps, const int num_rollouts, const int num_distributions,
                                    const int optimization_stride, const float std_dev_decay,
                                    const float pure_noise_percentage, const bool time_specific_std_dev = false,
                                    const float* __restrict__ min_control_d = nullptr,
                                    const float* __restrict__ max_control_d = nullptr);

// Set the default number of distributions to 2 since that is currently the most we would use
template <int C_DIM, int MAX_DISTRIBUTIONS_T = 2>
struct GaussianParamsImpl : public SamplingParams<C_DIM>
{
  static const int MAX_DISTRIBUTIONS = MAX_DISTRIBUTIONS_T;
  float std_dev[C_DIM * MAX_DISTRIBUTIONS] MPPI_ALIGN(sizeof(float4)) = { 0.0f };
  float control_cost_coeff[C_DIM] MPPI_ALIGN(sizeof(float4)) = { 0.0f };
  float min_control[C_DIM * MAX_DISTRIBUTIONS] MPPI_ALIGN(sizeof(float4)) = { 0.0f };
  float max_control[C_DIM * MAX_DISTRIBUTIONS] MPPI_ALIGN(sizeof(float4)) = { 0.0f };
  float pure_noise_trajectories_percentage = 0.01f;
  float std_dev_decay = 1.0f;
  // Kernel launching params
  dim3 rewrite_controls_block_dim = dim3(32, 16, 1);
  int sum_strides = 32;
  // Various flags
  bool time_specific_std_dev = false;

  GaussianParamsImpl(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : SamplingParams<C_DIM>::SamplingParams(num_rollouts, num_timesteps, num_distributions)
  {
    for (int i = 0; i < this->CONTROL_DIM * MAX_DISTRIBUTIONS; i++)
    {
      std_dev[i] = 1.0f;
      min_control[i] = -FLT_MAX;
      max_control[i] = FLT_MAX;
    }
  }

  void copyStdDevToDistribution(const int src_distribution_idx, const int dest_distribution_idx)
  {
    bool src_out_of_distribution = src_distribution_idx >= MAX_DISTRIBUTIONS;
    if (src_out_of_distribution || dest_distribution_idx >= MAX_DISTRIBUTIONS)
    {
      printf("%s Distribution %d is out of range. There are only %d total distributions\n",
             src_out_of_distribution ? "Src" : "Dest",
             src_out_of_distribution ? src_distribution_idx : dest_distribution_idx, MAX_DISTRIBUTIONS);
      return;
    }
    float* std_dev_src = std_dev[this->CONTROL_DIM * src_distribution_idx];
    float* std_dev_dest = std_dev[this->CONTROL_DIM * dest_distribution_idx];
    for (int i = 0; i < this->CONTROL_DIM; i++)
    {
      std_dev_dest[i] = std_dev_src[i];
    }
  }
};

template <int C_DIM>
using GaussianParams = GaussianParamsImpl<C_DIM, 2>;

template <int C_DIM, int MAX_TIMESTEPS = 1, int MAX_DISTRIBUTIONS_T = 2>
struct GaussianTimeVaryingStdDevParams : public GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS_T>
{
  float std_dev[C_DIM * MAX_TIMESTEPS * MAX_DISTRIBUTIONS_T] = { 0.0f };
  GaussianTimeVaryingStdDevParams(int num_rollouts = 1, int num_timesteps = 1, int num_distributions = 1)
    : GaussianParamsImpl<C_DIM, MAX_DISTRIBUTIONS_T>::GaussianParamsImpl(num_rollouts, num_timesteps, num_distributions)
  {
    this->time_specific_std_dev = true;
    for (int i = 0; i < this->CONTROL_DIM * MAX_TIMESTEPS * this->MAX_DISTRIBUTIONS; i++)
    {
      std_dev[i] = 1.0f;
    }
  }

  void copyStdDevToDistribution(const int src_distribution_idx, const int dest_distribution_idx)
  {
    bool src_out_of_distribution = src_distribution_idx >= this->MAX_DISTRIBUTIONS;
    if (src_out_of_distribution || dest_distribution_idx >= this->MAX_DISTRIBUTIONS)
    {
      printf("%s Distribution %d is out of range. There are only %d total distributions\n",
             src_out_of_distribution ? "Src" : "Dest",
             src_out_of_distribution ? src_distribution_idx : dest_distribution_idx, this->MAX_DISTRIBUTIONS);
      return;
    }
    float* std_dev_src = std_dev[this->CONTROL_DIM * this->num_timesteps * src_distribution_idx];
    float* std_dev_dest = std_dev[this->CONTROL_DIM * this->num_timesteps * dest_distribution_idx];
    for (int i = 0; i < this->CONTROL_DIM * this->num_timesteps; i++)
    {
      std_dev_dest[i] = std_dev_src[i];
    }
  }
};

/**
 * @brief Gaussian sampling distribution. This class used to derive (via CRTP) from a generic
 * SamplingDistribution<CLASS_T, PARAMS_TEMPLATE, DYN_PARAMS_T> base class shared by several sibling
 * implementations. That base has been merged directly into this class since Gaussian was, at the
 * time, the only sampling distribution implementation in use.
 *
 * Now that a second implementation (ColoredNoiseDistributionImpl, which derives from this class
 * rather than being a CRTP sibling) exists, one narrow piece of that removed CRTP had to come back:
 * SELF_T (defaulted to void, meaning "myself") is used ONLY to type `sampling_d_` (the device-side
 * mirror of this object) and the one `Managed::GPUSetup<...>(...)` call that creates it -- every
 * other inherited method (setNumRollouts/Timesteps/Distributions, setParams, paramsToDevice,
 * resizeVisualizationControlTrajectories, ...) just reads/writes fields *through* `sampling_d_`,
 * so they work correctly for any derived class without needing their own SELF_T-awareness or
 * overrides, as long as `sampling_d_`'s static type is the real most-derived class. Without this,
 * a derived class's `sampling_d_` would silently end up typed (and only *sizeof(base)*-allocated)
 * as this base class, which then fails to satisfy sampling-distribution-templated kernel launches
 * expecting the derived type.
 */
template <template <int> class PARAMS_TEMPLATE = GaussianParams, class DYN_PARAMS_T = DynamicsParams,
          class SELF_T = void>
class GaussianDistributionImpl : public Managed
{
public:
  /*************************************
   * Setup typedefs and aliases
   *************************************/
  using ControlIndex = typename DYN_PARAMS_T::ControlIndex;
  using OutputIndex = typename DYN_PARAMS_T::OutputIndex;
  using TEMPLATED_DYN_PARAMS = DYN_PARAMS_T;

  // See class comment above -- resolves to GaussianDistributionImpl itself when SELF_T is left
  // void (the default, used by every existing GaussianDistribution<...> instantiation), or to the
  // derived class a subclass passes as SELF_T (e.g. ColoredNoiseDistributionImpl<...>).
  using DEVICE_SELF_T = typename std::conditional<std::is_void<SELF_T>::value, GaussianDistributionImpl, SELF_T>::type;

  static const int CONTROL_DIM = C_IND_CLASS(DYN_PARAMS_T, NUM_CONTROLS);
  typedef PARAMS_TEMPLATE<CONTROL_DIM> SAMPLING_PARAMS_T;
  typedef Eigen::Matrix<float, CONTROL_DIM, 1> control_array;
  typedef Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> TEST_TYPE;

  static_assert(std::is_base_of<SamplingParams<CONTROL_DIM>, SAMPLING_PARAMS_T>::value,
                "Sampling Distribution PARAMS_T does not inherit from SamplingParams");

  /*************************************
   * Constructors and Destructors
   *************************************/
  GaussianDistributionImpl(cudaStream_t stream = 0);
  GaussianDistributionImpl(const SAMPLING_PARAMS_T& params, cudaStream_t stream = 0);

  ~GaussianDistributionImpl()
  {
    freeCudaMem();
  }

  /**
   * @brief Get the Sampling Distribution Name object
   *
   * @return std::string - name of the sampling distribution
   */
  __host__ virtual std::string getSamplingDistributionName() const
  {
    return "Gaussian";
  }

  /*************************************
   * DEFAULT CLASS METHODS THAT SHOULD NOT NEED OVERWRITING
   *************************************/

  void GPUSetup();

  /**
   * Updates the sampling distribution parameters
   * @param params
   */
  void setParams(const SAMPLING_PARAMS_T& params, bool synchronize = true)
  {
    bool reallocate_memory = params_.num_timesteps != params.num_timesteps ||
                             params_.num_rollouts != params.num_rollouts ||
                             params_.num_distributions != params.num_distributions;
    bool reallocate_vis_memory = params_.num_timesteps != params.num_timesteps ||
                                 params_.num_visualization_rollouts != params.num_visualization_rollouts ||
                                 params_.num_distributions != params.num_distributions;

    params_ = params;
    if (GPUMemStatus_)
    {
      if (reallocate_memory)
      {
        allocateCUDAMemory(false);
      }
      if (reallocate_vis_memory)
      {
        resizeVisualizationControlTrajectories(true);
      }
      paramsToDevice(synchronize);
    }
  }

  __host__ __device__ const SAMPLING_PARAMS_T getParams() const
  {
    return params_;
  }

  __host__ __device__ int getNumTimesteps() const
  {
    return this->params_.num_timesteps;
  }

  __host__ __device__ int getNumRollouts() const
  {
    return this->params_.num_rollouts;
  }

  __host__ __device__ int getNumVisRollouts() const
  {
    return this->params_.num_visualization_rollouts;
  }

  __host__ __device__ int getNumDistributions() const
  {
    return this->params_.num_distributions;
  }

  __host__ void setNumTimesteps(const int num_timesteps, bool synchronize = false)
  {
    const bool reallocate_memory = params_.num_timesteps != num_timesteps;
    this->params_.num_timesteps = num_timesteps;
    if (GPUMemStatus_ && reallocate_memory)
    {
      if (reallocate_memory)
      {
        allocateCUDAMemory(false);
        resizeVisualizationControlTrajectories(true);
      }
      paramsToDevice(synchronize);
    }
  }

  __host__ void setNumVisRollouts(const int num_visualization_rollouts, bool synchronize = false)
  {
    const bool reallocate_memory = params_.num_visualization_rollouts != num_visualization_rollouts;
    this->params_.num_visualization_rollouts = num_visualization_rollouts;
    if (GPUMemStatus_ && reallocate_memory)
    {
      if (reallocate_memory)
      {
        resizeVisualizationControlTrajectories(true);
      }
      paramsToDevice(synchronize);
    }
  }

  __host__ void setNumRollouts(const int num_rollouts, bool synchronize = false)
  {
    const bool reallocate_memory = params_.num_rollouts != num_rollouts;
    this->params_.num_rollouts = num_rollouts;
    if (GPUMemStatus_ && reallocate_memory)
    {
      if (reallocate_memory)
      {
        allocateCUDAMemory(false);
      }
      paramsToDevice(synchronize);
    }
  }

  __host__ void setNumDistributions(const int num_distributions, bool synchronize = false)
  {
    const bool reallocate_memory = params_.num_distributions != num_distributions;
    this->params_.num_distributions = num_distributions;
    if (GPUMemStatus_ && reallocate_memory)
    {
      if (reallocate_memory)
      {
        allocateCUDAMemory(false);
        resizeVisualizationControlTrajectories(true);
      }
      paramsToDevice(synchronize);
    }
  }

  __host__ void resizeVisualizationControlTrajectories(bool synchronize = true);

  __host__ void setVisStream(cudaStream_t stream)
  {
    vis_stream_ = stream;
  }

  __host__ void allocateCUDAMemory(bool synchronize = false);

  /**
   * @brief deallocates the allocated cuda memory for the sampling distribution
   */
  __host__ virtual void freeCudaMem();

  /**
   * @brief Get a pointer to a specific control sample. This is useful for plugging into methods like enforceConstraints
   *
   * @param sample_index - sample number out of num_rollouts
   * @param t - timestep out of num_timesteps
   * @param distribution_index - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param output - output pointer for compatibility with a output-based sampling distribution
   * @return float* pointer to the control array that is at [distribution_index][sample_index][t]
   */
  __host__ __device__ float* getControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                              const float* __restrict__ theta_d = nullptr,
                                              const float* __restrict__ output = nullptr);

  /**
   * @brief Get a pointer to a specific visualization control sample.
   *
   * @param sample_index - sample number out of num_visualization_rollouts
   * @param t - timestep out of num_timesteps
   * @param distribution_index - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param output - output pointer for compatibility with a output-based sampling distribution
   * @return float* pointer to the control array that is at [distribution_index][sample_index][t]
   */
  __host__ __device__ float* getVisControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                                 const float* __restrict__ theta_d = nullptr,
                                                 const float* __restrict__ output = nullptr);

  /**
   * @brief Method for starting up any potential work for distributions. By default, it just loads the params into
   * shared memory
   *
   * @param output - initial output
   * @param t_0 - starting time
   * @param dt - step size
   * @param theta_d - shared memory pointer to sampling distribution space
   */
  __device__ void initializeDistributions(const float* __restrict__ output, const float t_0, const float dt,
                                          float* __restrict__ theta_d);

  __host__ void paramsToDevice(bool synchronize = true);

  /**
   * @brief Look up a specific control sample located at [distribution_index][sample_index][t] and put it into the
   * control array
   *
   * @param sample_index - sample number out of num_rollouts
   * @param t - timestep out of num_timesteps
   * @param distribution_index - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param control - pointer to fill with the specific control array
   * @param theta_d - shared memory pointer for passing through params
   * @param block_size - parallelizable step size for the gpu (normally blockDim.y)
   * @param thread_index - parallelizable index for the gpu (normally threadIdx.y)
   * @param output - output pointer for compatibility with a output-based sampling distribution
   */
  __device__ void readControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                    float* __restrict__ control, float* __restrict__ theta_d, const int& block_size = 1,
                                    const int& thread_index = 1, const float* __restrict__ output = nullptr);

  /**
   * @brief Look up a specific visualization control sample located at [distribution_index][sample_index][t] and put it
   * into the control array
   *
   * @param sample_index - sample number out of num_visualization_rollouts
   * @param t - timestep out of num_timesteps
   * @param distribution_index - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param control - pointer to fill with the specific control array
   * @param theta_d - shared memory pointer for passing through params
   * @param block_size - parallelizable step size for the gpu (normally blockDim.y)
   * @param thread_index - parallelizable index for the gpu (normally threadIdx.y)
   * @param output - output pointer for compatibility with a output-based sampling distribution
   */
  __device__ void readVisControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                       float* __restrict__ control, float* __restrict__ theta_d,
                                       const int& block_size = 1, const int& thread_index = 1,
                                       const float* __restrict__ output = nullptr);

  /**
   * @brief Update the distribution according to the weights of each sample. Should only be used if weights only exist
   * on the host side. Otherwise, use updateDistributionParamsFromDevice
   *
   * @param trajectory_weights - vector of size num_rollouts containing the weight of each sample
   * @param normalizer - the sum of all trajectory weights
   * @param distribution_i - which distribution to update
   * @param synchronize - whether or not to run cudaStreamSynchronize
   */
  __host__ void updateDistributionParamsFromHost(const Eigen::Ref<const Eigen::MatrixXf>& trajectory_weights,
                                                 float normalizer, const int& distribution_i, bool synchronize = false);

  /*************************************
   * Gaussian-specific methods
   *************************************/

  /**
   * @brief method for allocating additional CUDA memory (std_dev / means) used by the Gaussian distribution
   */
  __host__ virtual void allocateCUDAMemoryHelper();

  __host__ __device__ float computeFeedbackCost(const float* __restrict__ u_fb, float* __restrict__ theta_d,
                                                const int t, const int distribution_idx, const float lambda = 1.0,
                                                const float alpha = 0.0);

  /**
   * @brief Device method to calculate the likelihood ratio cost for a given sample u
   *
   * @param u - sampled control
   * @param theta_d - shared memory for sampling distribution
   * @param t - timestep
   * @param distribution_idx - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param lambda - MPPI temperature parameter
   * @param alpha - coeff to turn off the likelihood cost (set to 1 -> no likelihood cost, set to 0 -> all likelihood
   * cost)
   */
  __host__ __device__ float computeLikelihoodRatioCost(const float* __restrict__ u, float* __restrict__ theta_d,
                                                       const int sample_index, const int t, const int distribution_idx,
                                                       const float lambda = 1.0, const float alpha = 0.0);

  /**
   * @brief Host-side method to calculate the likelihood ration cost for a given sample u
   *
   * @param u - sampled control
   * @param t - timestep
   * @param distribution_idx - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param lambda - MPPI temperature parameter
   * @param alpha - coeff to turn off the likelihood cost (set to 1 -> no likelihood cost, set to 0 -> all likelihood
   * cost)
   */
  __host__ float computeLikelihoodRatioCost(const Eigen::Ref<const control_array>& u, const int t,
                                            const int distribution_idx, const float lambda = 1.0,
                                            const float alpha = 0.0);

  /**
   * @brief Get the latest importance sampler from time-shifting on the controller and update the device importance
   * sampler
   *
   * @param importance_sampler - host pointer to a control sequence that is NUM_TIMESTEPS * CONTROL_DIM
   * @param distribution_idx - which distribution is the importance sampler meant for
   * @param synchronize - whether or not to run cudaStreamSynchronize
   */
  __host__ void copyImportanceSamplerToDevice(const float* importance_sampler, const int& distribution_idx,
                                              bool synchronize = true);

  /**
   * @brief Generate control samples that will be on the GPU.
   *
   * @param optimization_stride - timestep to start control samples from
   * @param iteration_num - which iteration of the algorithm we are on. Useful for decaying std_dev
   * @param gen - pseudo-random noise generator
   * @param synchronize - whether or not to run cudaStreamSynchronize
   * @param regenerate_noise - if true (default, matches prior behavior), draws a fresh curand N(0,1) batch and
   * caches it in cached_raw_noise_d_ before combining with the current mean/std_dev. If false, skips the curand
   * draw entirely and re-combines the *previously cached* raw noise with the current (possibly shifted) mean/
   * std_dev instead -- matches mppi_controller_ros's NoiseGenerator, which by default (noise_generator_regenerate_
   * noises=false) generates one noise batch and reuses it every cycle rather than resampling fresh each time.
   */
  __host__ virtual void generateSamples(const int& optimization_stride, const int& iteration_num,
                                        curandGenerator_t& gen, bool synchronize = true,
                                        bool regenerate_noise = true);

  /**
   * @brief Set the Host-side Optimal Control Trajectory
   *
   * @param optimal_control_trajectory - pointer to CPU memory location to store the optimal control
   * @param distribution_idx - which distribution we are looking for the optimal control from (Useful for Tube and
   * RMPPI)
   * @param synchronize - whether or not to run cudaStreamSynchronize
   */
  __host__ void setHostOptimalControlSequence(float* optimal_control_trajectory, const int& distribution_idx,
                                              bool synchronize = true);

  /**
   * @brief takes in the cost of each sample generated and conducts an update of the distribution (mean update)
   *
   * @param trajectory_weights_d - vector of weights of size num_rollouts located on the GPU
   * @param normalizer - sum of all weights
   * @param distribution_i - which distribution to update
   * @param synchronize - whether or not to run cudaStreamSynchronize
   */
  __host__ void updateDistributionParamsFromDevice(const float* trajectory_weights_d, float normalizer,
                                                   const int& distribution_i, bool synchronize = false);

  /**
   * @brief Write to a specific control sample located at [distribution_index][sample_index][t] from the
   * control array
   *
   * @param sample_index - sample number out of num_rollouts
   * @param t - timestep out of num_timesteps
   * @param distribution_index - distribution index (if it is larger than num_distributions, it just defaults to first
   * distribution for future compatibility with sampling dynamical systems)
   * @param control - pointer to control array with the desired data
   * @param theta_d - shared memory pointer for passing through params
   * @param block_size - parallelizable step size for the gpu (normally blockDim.y)
   * @param thread_index - parallelizable index for the gpu (normally threadIdx.y)
   * @param output - output pointer for compatibility with a output-based sampling distribution
   */
  __device__ void writeControlSample(const int& sample_index, const int& t, const int& distribution_index,
                                     const float* __restrict__ control, float* __restrict__ theta_d,
                                     const int& block_size = 1, const int& thread_index = 1,
                                     const float* __restrict__ output = nullptr);

  DEVICE_SELF_T* sampling_d_ = nullptr;
  cudaStream_t vis_stream_ = nullptr;

protected:
  float* control_samples_d_ = nullptr;
  float* cached_raw_noise_d_ = nullptr;
  float* vis_control_samples_d_ = nullptr;

  SAMPLING_PARAMS_T params_;

  // Gaussian-specific device/host memory
  float* std_dev_d_ = nullptr;
  float* control_means_d_ = nullptr;
  std::vector<float> means_;
  float* min_control_d_ = nullptr;
  float* max_control_d_ = nullptr;
};

// Convenience alias preserving the previous public-facing name/usage:
// mppi::sampling_distributions::GaussianDistribution<DynamicsParamsT>
template <class DYN_PARAMS_T>
using GaussianDistribution = GaussianDistributionImpl<GaussianParams, DYN_PARAMS_T>;

}  // namespace sampling_distributions
}  // namespace mppi

#endif  // GAUSSIAN_CUH
