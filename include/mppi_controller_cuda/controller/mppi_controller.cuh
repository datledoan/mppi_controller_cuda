#ifndef MPPI_CONTROLLER_CUH
#define MPPI_CONTROLLER_CUH

#include "mppi_controller_cuda/sampling_distribution/gaussian.cuh"
#include "mppi_controller_cuda/sampling_distribution/colored_noise.cuh"
#include "mppi_controller_cuda/core/mppi_common.cuh"
#include "mppi_controller_cuda/utils/gpu_err_chk.cuh"
#include "mppi_controller_cuda/utils/math_utils.h"

#include <cfloat>
#include <utility>

void cudamain();

struct freeEnergyEstimate
{
  float increase = -1;
  float previousBaseline = -1;
  float freeEnergyMean = -1;
  float freeEnergyVariance = -1;
  float freeEnergyModifiedVariance = -1;
  float normalizerPercent = -1;
};

struct MPPIFreeEnergyStatistics
{
  freeEnergyEstimate real_sys;
};

enum class kernelType : int
{
  USE_SINGLE_KERNEL = 0,  // combine dynamics and cost calls into a single kernel
  USE_SPLIT_KERNELS,      // separate kernels for dynamics and cost calls
};

template <int S_DIM, int C_DIM, int MAX_TIMESTEPS>
struct ControllerParams
{
  static const int TEMPLATED_STATE_DIM = S_DIM;
  static const int TEMPLATED_CONTROL_DIM = C_DIM;
  static const int TEMPLATED_MAX_TIMESTEPS = MAX_TIMESTEPS;
  float dt_;
  float lambda_ = 1.0;  // Value of the temperature in the softmax.
  float alpha_ = 0.0;   //
  int num_timesteps_ = MAX_TIMESTEPS;
  int num_iters_ = 1;  // Number of optimization iterations
  unsigned seed_ = std::chrono::system_clock::now().time_since_epoch().count();

  dim3 dynamics_rollout_dim_;
  dim3 cost_rollout_dim_;
  dim3 visualize_dim_ = dim3(32, 1, 1);
  int norm_exp_kernel_parallelization_ = 64;

  Eigen::Matrix<float, C_DIM, MAX_TIMESTEPS> init_control_traj_ = Eigen::Matrix<float, C_DIM, MAX_TIMESTEPS>::Zero();
  Eigen::Matrix<float, C_DIM, 1> slide_control_scale_ = Eigen::Matrix<float, C_DIM, 1>::Zero();
};

template <class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS,
          class SAMPLING_T = ::mppi::sampling_distributions::GaussianDistribution<typename DYN_T::DYN_PARAMS_T>,
          class PARAMS_T = ControllerParams<DYN_T::STATE_DIM, DYN_T::CONTROL_DIM, MAX_TIMESTEPS>>
class Controller
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * typedefs for access to templated class from outside classes
   */
  typedef DYN_T TEMPLATED_DYNAMICS;
  typedef COST_T TEMPLATED_COSTS;
  typedef PARAMS_T TEMPLATED_PARAMS;
  typedef SAMPLING_T TEMPLATED_SAMPLING;
  using TEMPLATED_SAMPLING_PARAMS = typename SAMPLING_T::SAMPLING_PARAMS_T;

  /**
   * Aliases
   */
  // Control typedefs
  using control_array = typename DYN_T::control_array;
  typedef Eigen::Matrix<float, DYN_T::CONTROL_DIM, MAX_TIMESTEPS> control_trajectory;  // A control trajectory

  // State typedefs
  using state_array = typename DYN_T::state_array;
  typedef Eigen::Matrix<float, DYN_T::STATE_DIM, MAX_TIMESTEPS> state_trajectory;  // A state trajectory

  // Output typedefs
  using output_array = typename DYN_T::output_array;
  typedef Eigen::Matrix<float, DYN_T::OUTPUT_DIM, MAX_TIMESTEPS> output_trajectory;  // An output trajectory

  // Cost typedefs
  typedef Eigen::Matrix<float, MAX_TIMESTEPS + 1, 1> cost_trajectory;  // +1 for terminal cost
  typedef Eigen::Matrix<float, NUM_ROLLOUTS, 1> sampled_cost_traj;
  typedef Eigen::Matrix<int, MAX_TIMESTEPS, 1> crash_status_trajectory;
  typedef Eigen::Matrix<int, NUM_ROLLOUTS, 1> rollout_crash_status_traj;

  Controller(DYN_T* model, COST_T* cost, SAMPLING_T* sampler, float dt, int max_iter, float lambda, float alpha,
             int num_timesteps = MAX_TIMESTEPS,
             const Eigen::Ref<const control_trajectory>& init_control_traj = control_trajectory::Zero(),
             cudaStream_t stream = nullptr)
  {
    // Create the random number generator
    createAndSeedCUDARandomNumberGen();
    model_ = model;
    cost_ = cost;
    sampler_ = sampler;
    sampler_->setNumRollouts(NUM_ROLLOUTS);
    sampler_->setNumDistributions(1);
    TEMPLATED_PARAMS params;
    params.dt_ = dt;
    params.num_iters_ = max_iter;
    params.lambda_ = lambda;
    params.alpha_ = alpha;
    params.num_timesteps_ = num_timesteps;
    params.init_control_traj_ = init_control_traj;
    setNumTimesteps(params.num_timesteps_);
    setParams(params);

    control_ = init_control_traj;
    control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

    // Bind the model and control to the given stream
    setCUDAStream(stream);
    // Create new stream for visualization purposes
    HANDLE_ERROR(cudaStreamCreate(&vis_stream_));

    GPUSetup();

    auto logger = std::make_shared<mppi::util::MPPILogger>();
    setLogger(logger);

    // Allocate CUDA memory and pick the fastest rollout kernel for this GPU
    allocateCUDAMemory();
    chooseAppropriateKernel();
  }

  Controller(DYN_T* model, COST_T* cost, SAMPLING_T* sampler, PARAMS_T& params, cudaStream_t stream = nullptr)
  {
    model_ = model;
    cost_ = cost;
    sampler_ = sampler;
    sampler_->setNumRollouts(NUM_ROLLOUTS);
    sampler_->setNumDistributions(1);
    setNumTimesteps(params_.num_timesteps_);
    // Create the random number generator
    createAndSeedCUDARandomNumberGen();
    setParams(params);
    control_ = params_.init_control_traj_;
    control_history_ = Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>::Zero();

    // Bind the model and control to the given stream
    setCUDAStream(stream);
    // Create new stream for visualization purposes
    HANDLE_ERROR(cudaStreamCreate(&vis_stream_));

    GPUSetup();

    auto logger = std::make_shared<mppi::util::MPPILogger>();
    setLogger(logger);

    // Allocate CUDA memory and pick the fastest rollout kernel for this GPU
    allocateCUDAMemory();
    chooseAppropriateKernel();
  }

  // TODO should be private with test as a friend to ensure it is only used in testing
  Controller() = default;

  ~Controller()
  {
    // Free the CUDA memory of every object
    if (model_)
    {
      model_->freeCudaMem();
    }
    if (cost_)
    {
      cost_->freeCudaMem();
    }
    if (sampler_)
    {
      sampler_->freeCudaMem();
    }

    // Free the CUDA memory of the controller
    deallocateCUDAMemory();
  }

  std::string getControllerName() const
  {
    return "MPPI";
  };

  std::string getCostFunctionName() const
  {
    return cost_->getCostFunctionName();
  }

  std::string getDynamicsModelName() const
  {
    return model_->getDynamicsModelName();
  }

  std::string getSamplingDistributionName() const
  {
    return sampler_->getSamplingDistributionName();
  }

  std::string getFullName() const
  {
    return getControllerName() + "(" + getDynamicsModelName() + ", " + getCostFunctionName() + ", " +
           getSamplingDistributionName() + ")";
  }

  void GPUSetup()
  {
    // Call the GPU setup functions of the model, cost, and sampling distribution
    model_->GPUSetup();
    cost_->GPUSetup();
    sampler_->setVisStream(vis_stream_);
    sampler_->GPUSetup();
  }

  std::vector<output_trajectory> getSampledOutputTrajectories() const
  {
    return sampled_trajectories_;
  }

  std::vector<cost_trajectory> getSampledCostTrajectories() const
  {
    return sampled_costs_;
  }

  std::vector<crash_status_trajectory> getSampledCrashStatusTrajectories() const
  {
    return sampled_crash_status_;
  }

  std::vector<float> getTopTransformedCosts() const
  {
    return top_n_costs_;
  }

  /**
   * Times the single-kernel vs split-kernel rollout implementations on the
   * current GPU and picks whichever is faster (falling back to whichever one
   * fits in shared memory if only one of them does).
   */
  void chooseAppropriateKernel();

  /**
   * Given a state, calculates the optimal control sequence using MPPI according
   * to the cost function used as part of the template
   * @param state - the current state from which we would like to calculate
   * a control sequence
   */
  void computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride = 1);

  /**
   * Call a kernel to evaluate the sampled state trajectories for visualization
   * and debugging.
   */
  void calculateSampledStateTrajectories();

  void setPercentageSampledControlTrajectories(float new_perc)
  {
    this->setPercentageSampledControlTrajectoriesHelper(new_perc, 1);
  }

  /**
   * Used to update the importance sampler
   * @param nominal_control the new nominal control sequence to sample around
   */
  void updateImportanceSampler(const Eigen::Ref<const control_trajectory>& nominal_control)
  {
    // TODO copy to device new control sequence
    control_ = nominal_control;
  }

  void setRegenerateNoise(bool regenerate_noise)
  {
    regenerate_noise_ = regenerate_noise;
  }

  void setRetryAttemptLimit(int retry_attempt_limit)
  {
    retry_attempt_limit_ = retry_attempt_limit;
  }

  /**
   * determines the control that should be applied given the current relative time,
   * by interpolating the optimized (feedforward) MPPI control trajectory
   * @param rel_time
   * @return
   */
  control_array getCurrentControl(double rel_time, control_trajectory& c_traj)
  {
    control_array result = interpolateControls(rel_time, c_traj);

    state_array empty_state = model_->getZeroState();
    model_->enforceConstraints(empty_state, result);

    return result;
  }

  /**
   * @param steps - number of dt's to slide control sequence forward
   * Slide the control sequence forwards by 'steps'
   */
  void slideControlSequence(int steps)
  {
    // Save the control history
    this->saveControlHistoryHelper(steps, this->control_, this->control_history_);

    this->slideControlSequenceHelper(steps, this->control_);
  }

  /**
   * determines the interpolated control from control_seq_, linear interpolation
   * @param rel_time time since the solution was calculated
   * @return
   */
  control_array interpolateControls(double rel_time, control_trajectory& c_traj)
  {
    int lower_idx = (int)(rel_time / getDt());
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * getDt()) / getDt();

    control_array interpolated_control;
    control_array prev_cmd = c_traj.col(lower_idx);
    control_array next_cmd = c_traj.col(upper_idx);
    interpolated_control = (1 - alpha) * prev_cmd + alpha * next_cmd;

    return interpolated_control;
  }

  state_array interpolateState(state_trajectory& s_traj, double rel_time)
  {
    int lower_idx = (int)(rel_time / getDt());
    int upper_idx = lower_idx + 1;
    double alpha = (rel_time - lower_idx * getDt()) / getDt();

    return model_->interpolateState(s_traj.col(lower_idx), s_traj.col(upper_idx), alpha);
  }

  curandGenerator_t getGenerator() const
  {
    return gen_;
  }

  /**
   * returns the current control sequence
   */
  control_trajectory getControlSeq() const
  {
    return control_;
  };

  /**
   * Gets the state sequence of the nominal trajectory
   */
  state_trajectory getTargetStateSeq() const
  {
    return state_;
  }

  /**
   * Gets the output sequence of the nominal trajectory
   */
  output_trajectory getTargetOutputSeq() const
  {
    return output_;
  }

  /**
   * Return all the sampled costs sequences
   */
  sampled_cost_traj getSampledCostSeq() const
  {
    return trajectory_costs_;
  };

  // Indicator for algorithm health, should be between 0.01 and 0.1 anecdotally
  float getNormalizerPercent() const
  {
    return this->getNormalizerCost() / (float)NUM_ROLLOUTS;
  }

  float getBaselineCost() const
  {
    return cost_baseline_and_norm_.x;
  };
  float getNormalizerCost() const
  {
    return cost_baseline_and_norm_.y;
  };

  /**
   * returns the current state sequence
   */
  state_trajectory getActualStateSeq() const
  {
    return state_;
  };

  /**
   * returns the current output sequence
   */
  output_trajectory getActualOutputSeq() const
  {
    return output_;
  };

  void smoothControlTrajectoryHelper(Eigen::Ref<control_trajectory> u,
                                     const Eigen::Ref<Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>>& control_history)
  {
    // TODO generalize to any size filter
    // TODO does the logic of handling control history reasonable?

    // Create the filter coefficients
    Eigen::Matrix<float, 1, 5> filter_coefficients;
    filter_coefficients << -3, 12, 17, 12, -3;
    filter_coefficients /= 35.0;

    // Create and fill a control buffer that we can apply the convolution filter
    Eigen::MatrixXf control_buffer(getNumTimesteps() + 4, DYN_T::CONTROL_DIM);

    // Fill the first two timesteps with the control history
    control_buffer.topRows(2) = control_history.transpose();

    // Fill the center timesteps with the current nominal trajectory
    control_buffer.middleRows(2, getNumTimesteps()) = u.transpose();

    // Fill the last two timesteps with the end of the current nominal control trajectory
    control_buffer.row(getNumTimesteps() + 2) = u.transpose().row(getNumTimesteps() - 1);
    control_buffer.row(getNumTimesteps() + 3) = u.transpose().row(getNumTimesteps() - 1);

    // Apply convolutional filter to each timestep
    for (int i = 0; i < getNumTimesteps(); ++i)
    {
      u.col(i) = (filter_coefficients * control_buffer.middleRows(i, 5)).transpose();
    }
  }

  void slideControlSequenceHelper(int steps, Eigen::Ref<control_trajectory> u)
  {
    for (int i = 0; i < getNumTimesteps(); ++i)
    {
      int ind = std::min(i + steps, getNumTimesteps() - 1);
      u.col(i) = u.col(ind);
      if (i + steps > getNumTimesteps() - 1)
      {
        u.col(i) = (u.col(ind).array() - model_->zero_control_.array()) * params_.slide_control_scale_.array() +
                   model_->zero_control_.array();
      }
    }
  }

  void saveControlHistoryHelper(int steps, const Eigen::Ref<const control_trajectory>& u_trajectory,
                                Eigen::Ref<Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2>> u_history)
  {
    if (steps == 1)
    {  // We only moved one timestep
      u_history.col(0) = u_history.col(1);
      u_history.col(1) = u_trajectory.col(0);
    }
    else if (steps >= 2)
    {  // We have moved more than one timestep, but our history size is still only 2
      u_history.col(0) = u_trajectory.col(steps - 2);
      u_history.col(1) = u_trajectory.col(steps - 1);
    }
  }

  void computeStateTrajectoryHelper(Eigen::Ref<state_trajectory> result, const Eigen::Ref<const state_array>& x0,
                                    const Eigen::Ref<const control_trajectory>& u)
  {
    result.col(0) = x0;
    state_array xdot;
    state_array state, next_state;
    output_array output;
    model_->initializeDynamics(result.col(0), u.col(0), output, 0, getDt());
    for (int i = 0; i < getNumTimesteps() - 1; ++i)
    {
      state = result.col(i);
      control_array u_i = u.col(i);
      model_->enforceConstraints(state, u_i);
      model_->step(state, next_state, xdot, u_i, output, i, getDt());
      result.col(i + 1) = next_state;
    }
  }

  void computeOutputTrajectoryHelper(Eigen::Ref<output_trajectory> output_result,
                                     Eigen::Ref<state_trajectory> state_result,
                                     const Eigen::Ref<const state_array>& x0,
                                     const Eigen::Ref<const control_trajectory>& u)
  {
    state_result.col(0) = x0;
    state_array xdot;
    state_array state, next_state;
    output_array output;
    model_->initializeDynamics(state_result.col(0), u.col(0), output, 0, getDt());
    output_result.col(0) = output;
    for (int i = 0; i < getNumTimesteps() - 1; ++i)
    {
      state = state_result.col(i);
      control_array u_i = u.col(i);
      model_->enforceConstraints(state, u_i);
      model_->step(state, next_state, xdot, u_i, output, i, getDt());
      state_result.col(i + 1) = next_state;
      output_result.col(i + 1) = output;
    }
  }

  void setNumTimesteps(int num_timesteps)
  {
    if ((num_timesteps <= MAX_TIMESTEPS) && (num_timesteps > 0))
    {
      params_.num_timesteps_ = num_timesteps;
    }
    else
    {
      params_.num_timesteps_ = MAX_TIMESTEPS;
      printf("You must give a number of timesteps between [0, %d]\n", MAX_TIMESTEPS);
    }
    sampler_->setNumTimesteps(params_.num_timesteps_);
  }

  void setBaseline(float baseline)
  {
    cost_baseline_and_norm_.x = baseline;
  };

  void setNormalizer(float normalizer)
  {
    cost_baseline_and_norm_.y = normalizer;
  };

  int getNumTimesteps() const
  {
    return this->params_.num_timesteps_;
  }

  int getNormExpThreads() const
  {
    return this->params_.norm_exp_kernel_parallelization_;
  }

  float getPercentageSampledControlTrajectories() const
  {
    return perc_sampled_control_trajectories_;
  }
  /**
   * Set the percentage of sample control trajectories to copy
   * back from the GPU. Multiplier is an integer in case the nominal
   * control trajectories also need to be saved.
   */
  void setPercentageSampledControlTrajectoriesHelper(float new_perc, int multiplier = 1)
  {
    perc_sampled_control_trajectories_ = new_perc;
    sample_multiplier_ = multiplier;
    resizeSampledControlTrajectories(perc_sampled_control_trajectories_, sample_multiplier_,
                                     num_top_control_trajectories_);
  }

  void setTopNSampledControlTrajectoriesHelper(int new_top_num_samples)
  {
    num_top_control_trajectories_ = new_top_num_samples;
    resizeSampledControlTrajectories(perc_sampled_control_trajectories_, sample_multiplier_,
                                     num_top_control_trajectories_);
  }

  void resizeSampledControlTrajectories(float perc, int multiplier, int top_num);

  int getNumberSampledTrajectories() const
  {
    return perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  }

  int getNumberTopControlTrajectories() const
  {
    return num_top_control_trajectories_;
  }

  int getTotalSampledTrajectories() const
  {
    return getNumberSampledTrajectories() + getNumberTopControlTrajectories();
  }

  void setSlideControlScale(const Eigen::Ref<const control_array>& slide_control_scale)
  {
    params_.slide_control_scale_ = slide_control_scale;
  }

  /**
   * Return the most recent free energy calculation
   */
  MPPIFreeEnergyStatistics getFreeEnergyStatistics() const
  {
    return free_energy_statistics_;
  }

  std::vector<float> getSampledNoise();

  /**
   * Public data members
   */
  DYN_T* model_ = nullptr;
  COST_T* cost_ = nullptr;
  SAMPLING_T* sampler_ = nullptr;
  cudaStream_t stream_;
  cudaStream_t vis_stream_;

  float getDt() const
  {
    return params_.dt_;
  }
  void setDt(float dt)
  {
    params_.dt_ = dt;
  }

  float getLambda() const
  {
    return params_.lambda_;
  }
  void setLambda(float lambda)
  {
    params_.lambda_ = lambda;
  }

  float getAlpha() const
  {
    return params_.alpha_;
  }
  void setAlpha(float alpha)
  {
    params_.alpha_ = alpha;
  }

  PARAMS_T getParams() const
  {
    return params_;
  }

  const TEMPLATED_SAMPLING_PARAMS getSamplingParams() const
  {
    return sampler_->getParams();
  }

  void setSamplingParams(const TEMPLATED_SAMPLING_PARAMS& params, bool synchronize = true)
  {
    sampler_->setParams(params, synchronize);
  }

  void setParams(const PARAMS_T& p)
  {
    bool change_seed = p.seed_ != params_.seed_;
    bool change_num_timesteps = p.num_timesteps_ != params_.num_timesteps_;
    params_ = p;
    if (change_num_timesteps)
    {
      setNumTimesteps(p.num_timesteps_);
    }
    if (change_seed)
    {
      setSeedCUDARandomNumberGen(params_.seed_);
    }
  }

  int getNumIters() const
  {
    return params_.num_iters_;
  }
  void setNumIters(int num_iter)
  {
    params_.num_iters_ = num_iter;
  }

  float getDebug() const
  {
    return debug_;
  }
  void setDebug(bool debug)
  {
    debug_ = debug;
    setLogLevel(mppi::util::LOG_LEVEL::DEBUG);
  }

  int getKernelChoiceAsInt() const
  {
    return static_cast<int>(use_kernel_);
  }

  kernelType getKernelChoiceAsEnum() const
  {
    return use_kernel_;
  }

  int getNumKernelEvaluations() const
  {
    return num_kernel_evaluations_;
  }

  void setKernelChoice(const int& kernel_type)
  {
    use_kernel_ = static_cast<kernelType>(kernel_type);
  }

  void setKernelChoice(const kernelType& kernel_type)
  {
    use_kernel_ = kernel_type;
  }

  void setNumKernelEvaluations(const int& num_kernel_evaluations)
  {
    num_kernel_evaluations_ = num_kernel_evaluations;
  }

  void setCUDAStream(cudaStream_t stream);

  void setLogLevel(const mppi::util::LOG_LEVEL& level)
  {
    logger_->setLogLevel(level);
    model_->setLogLevel(level);
    cost_->setLogLevel(level);
    sampler_->setLogLevel(level);
  }

  void setLogger(const mppi::util::MPPILoggerPtr& logger)
  {
    logger_ = logger;
    model_->setLogger(logger);
    cost_->setLogger(logger);
    sampler_->setLogger(logger);
  }

  mppi::util::MPPILoggerPtr getLogger() const
  {
    return logger_;
  }

  mppi::util::MPPILoggerPtr getLogger()
  {
    return logger_;
  }

protected:
  void deallocateCUDAMemory();
  void allocateCUDAMemory();

  PARAMS_T params_;
  mppi::util::MPPILoggerPtr logger_ = nullptr;

  bool debug_ = false;
  bool regenerate_noise_ = true;
  int retry_attempt_limit_ = 1;
  int num_kernel_evaluations_ = 10;  // number of kernel calls to do to determine which kernel to use
  kernelType use_kernel_ = kernelType::USE_SPLIT_KERNELS;  // default is to use the split kernels

  // Free energy variables
  MPPIFreeEnergyStatistics free_energy_statistics_;

  float perc_sampled_control_trajectories_ = 0;  // Percentage of sampled trajectories to return
  int sample_multiplier_ = 1;
  int num_top_control_trajectories_ = 0;  // Top n sampled trajectories to visualize
  std::vector<float> top_n_costs_;

  curandGenerator_t gen_;
  float* initial_state_d_;      // Array of size DYN_T::STATE_DIM
  float* vis_initial_state_d_;  // Array of size DYN_T::STATE_DIM

  Eigen::Matrix<float, DYN_T::CONTROL_DIM, 2> control_history_;

  float* control_d_;           // Array of size DYN_T::CONTROL_DIM*MAX_TIMESTEPS
  float* output_d_;            // Array of size DYN_T::OUTPUT_DIM*MAX_TIMESTEPS*NUM_ROLLOUTS
  float* trajectory_costs_d_;  // Array of size NUM_ROLLOUTS
  int* crash_status_d_;        // Array of size NUM_ROLLOUTS -- did rollout i collide anywhere
  float2* cost_baseline_and_norm_d_;
  control_trajectory control_ = control_trajectory::Zero();
  state_trajectory state_ = state_trajectory::Zero();
  output_trajectory output_ = output_trajectory::Zero();
  sampled_cost_traj trajectory_costs_ = sampled_cost_traj::Zero();
  rollout_crash_status_traj crash_status_ = rollout_crash_status_traj::Zero();
  float2 cost_baseline_and_norm_ = make_float2(0.0, 0.0);
  bool CUDA_mem_init_ = false;

  bool sampled_states_CUDA_mem_init_ = false;  // cudaMalloc, cudaFree boolean
  float* sampled_outputs_d_;                   // result of states that have been sampled from state trajectory kernel
  float* sampled_costs_d_;       // result of cost that have been sampled from state and cost trajectory kernel
  int* sampled_crash_status_d_;  // result of crash_status that have been sampled
  std::vector<output_trajectory> sampled_trajectories_;  // sampled state trajectories from state trajectory kernel
  std::vector<cost_trajectory> sampled_costs_;
  std::vector<crash_status_trajectory> sampled_crash_status_;

  void copyNominalControlToDevice(bool synchronize = true);

  /**
   * Saves the sampled controls from the GPU back to the CPU
   * Must be called after the rolloutKernel as that is when
   * du_d becomes the sampled controls
   */
  void copySampledControlFromDevice(bool synchronize = true);

  std::pair<int, float> findMinIndexAndValue(std::vector<int>& temp_list);
  void copyTopControlFromDevice(bool synchronize = true);

  void createAndSeedCUDARandomNumberGen();

  void setSeedCUDARandomNumberGen(unsigned seed);

  void computeStateTrajectory(const Eigen::Ref<const state_array>& x0)
  {
    computeOutputTrajectoryHelper(output_, state_, x0, control_);
  }

  void smoothControlTrajectory()
  {
    smoothControlTrajectoryHelper(control_, control_history_);
  }
};


#endif  // MPPI_CONTROLLER_CUH
