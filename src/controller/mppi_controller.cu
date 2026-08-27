#include <atomic>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mppi_controller_cuda/controller/mppi_controller.cuh"
#include "mppi_controller_cuda/costs/motion_model_cost.cuh"
#include "mppi_controller_cuda/dynamics/motion_model.cuh"

__global__ void print_from_gpu(void) {
    printf("Hello World! from thread [%d,%d] From device\n", threadIdx.x, blockIdx.x); 
}

void cudamain() 
{
    cudaDeviceProp  prop;
    int dev;

    HANDLE_ERROR( cudaGetDevice( &dev ) );
    printf( "ID of current CUDA device:  %d\n", dev );

    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = 1;
    prop.minor = 3;
    HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
    printf( "ID of CUDA device closest to revision 1.3:  %d\n", dev );

    HANDLE_ERROR( cudaSetDevice( dev ) );
}

#define CONTROLLER_TEMPLATE                                                                                          \
  template <class DYN_T, class COST_T, int MAX_TIMESTEPS, int NUM_ROLLOUTS, class SAMPLING_T, class PARAMS_T>
#define CONTROLLER Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T, PARAMS_T>

CONTROLLER_TEMPLATE
void CONTROLLER::deallocateCUDAMemory()
{
  if (CUDA_mem_init_)
  {
    HANDLE_ERROR(cudaFree(initial_state_d_));
    HANDLE_ERROR(cudaFree(vis_initial_state_d_));
    HANDLE_ERROR(cudaFree(control_d_));
    HANDLE_ERROR(cudaFree(output_d_));
    HANDLE_ERROR(cudaFree(trajectory_costs_d_));
    HANDLE_ERROR(cudaFree(crash_status_d_));
    HANDLE_ERROR(cudaFree(cost_baseline_and_norm_d_));
    CUDA_mem_init_ = false;
  }
  if (sampled_states_CUDA_mem_init_)
  {
    HANDLE_ERROR(cudaFree(sampled_outputs_d_));
    HANDLE_ERROR(cudaFree(sampled_costs_d_));
    sampled_states_CUDA_mem_init_ = false;
  }
}

CONTROLLER_TEMPLATE
void CONTROLLER::copyNominalControlToDevice(bool synchronize)
{
  if (!CUDA_mem_init_)
  {
    return;
  }
  this->sampler_->copyImportanceSamplerToDevice(control_.data(), 0, synchronize);
}

CONTROLLER_TEMPLATE
void CONTROLLER::copySampledControlFromDevice(bool synchronize)
{
  // if mem is not inited don't use it
  if (!sampled_states_CUDA_mem_init_)
  {
    return;
  }

  int num_sampled_trajectories = perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  std::vector<int> samples(num_sampled_trajectories);
  if (perc_sampled_control_trajectories_ > 0.98)
  {
    // if above threshold just do everything
    std::iota(samples.begin(), samples.end(), 0);
  }
  else
  {
    // Create sample list without replacement
    // removes the top 2% since top 1% are complete noise
    samples = mppi::math::sample_without_replacement(num_sampled_trajectories, NUM_ROLLOUTS * 0.98);
  }

  // this explicitly adds the optimized control sequence
  HANDLE_ERROR(cudaMemcpyAsync(this->sampled_outputs_d_, this->output_.data(),
                               sizeof(float) * getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyHostToDevice,
                               this->vis_stream_));
  HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(0, 0, 0), this->control_d_,
                               sizeof(float) * getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                               this->vis_stream_));

  for (int i = 1; i < num_sampled_trajectories; i++)
  {
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_outputs_d_ + i * getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 this->output_d_ + samples[i] * getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 sizeof(float) * getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(i, 0, 0),
                                 this->sampler_->getControlSample(samples[i], 0, 0),
                                 sizeof(float) * getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
  }
}

CONTROLLER_TEMPLATE
std::pair<int, float> CONTROLLER::findMinIndexAndValue(std::vector<int>& temp_list)
{
  if (temp_list.size() == 0)
  {
    return std::make_pair(0, 0.0);
  }
  int min_sample_index = 0;
  float min_sample_value = this->trajectory_costs_[temp_list[min_sample_index]];

  for (int index = 1; index < temp_list.size(); index++)
  {
    if (this->trajectory_costs_[temp_list[index]] < min_sample_value)
    {
      min_sample_value = this->trajectory_costs_[temp_list[index]];
      min_sample_index = index;
    }
  }
  return std::make_pair(min_sample_index, min_sample_value);
}

CONTROLLER_TEMPLATE
void CONTROLLER::copyTopControlFromDevice(bool synchronize)
{
  // if mem is not inited don't use it
  if (!sampled_states_CUDA_mem_init_ || num_top_control_trajectories_ <= 0)
  {
    return;
  }

  // Important note: Highest weighted trajectories are the ones with the lowest cost
  int start_top_control_traj_index = perc_sampled_control_trajectories_ * NUM_ROLLOUTS;
  std::vector<int> samples(num_top_control_trajectories_);
  // Start by filling in the top samples list with the first n in the trajectory
  for (int i = 0; i < num_top_control_trajectories_; i++)
  {
    samples[i] = i;
  }

  // Calculate min weight in the current top samples list
  int min_sample_index = 0;
  float min_sample_value = 0;
  std::tie(min_sample_index, min_sample_value) = findMinIndexAndValue(samples);

  // find top n samples by removing the smallest weights from the list
  for (int i = num_top_control_trajectories_; i < NUM_ROLLOUTS; i++)
  {
    if (trajectory_costs_[i] > min_sample_value)
    {  // Remove the smallest weight in the current list and add the new index
      samples[min_sample_index] = i;
      // recalculate min weight in the current list
      std::tie(min_sample_index, min_sample_value) = findMinIndexAndValue(samples);
    }
  }

  // Copy top n samples to the visualization buffer after the randomly sampled trajectories
  top_n_costs_.resize(num_top_control_trajectories_);
  for (int i = 0; i < num_top_control_trajectories_; i++)
  {
    top_n_costs_[i] = trajectory_costs_[samples[i]] / getNormalizerCost();
    HANDLE_ERROR(cudaMemcpyAsync(
        this->sampled_outputs_d_ + (start_top_control_traj_index + i) * getNumTimesteps() * DYN_T::OUTPUT_DIM,
        this->output_d_ + samples[i] * getNumTimesteps() * DYN_T::OUTPUT_DIM,
        sizeof(float) * getNumTimesteps() * DYN_T::OUTPUT_DIM, cudaMemcpyDeviceToDevice, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampler_->getVisControlSample(start_top_control_traj_index + i, 0, 0),
                                 this->sampler_->getControlSample(samples[i], 0, 0),
                                 sizeof(float) * getNumTimesteps() * DYN_T::CONTROL_DIM, cudaMemcpyDeviceToDevice,
                                 this->vis_stream_));
  }
  if (synchronize)
  {
    HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
  }
}

CONTROLLER_TEMPLATE
void CONTROLLER::setCUDAStream(cudaStream_t stream)
{
  stream_ = stream;
  model_->bindToStream(stream);
  cost_->bindToStream(stream);
  sampler_->bindToStream(stream);
  curandSetStream(gen_, stream);  // requires the generator to be created!
}

CONTROLLER_TEMPLATE
void CONTROLLER::createAndSeedCUDARandomNumberGen()
{
  // Seed the PseudoRandomGenerator with the CPU time.
  curandCreateGenerator(&gen_, CURAND_RNG_PSEUDO_DEFAULT);
  setSeedCUDARandomNumberGen(this->params_.seed_);
}

CONTROLLER_TEMPLATE
void CONTROLLER::setSeedCUDARandomNumberGen(unsigned seed)
{
  // Seed the PseudoRandomGenerator with the CPU time.
  curandSetPseudoRandomGeneratorSeed(gen_, seed);
  // Reset the offset so setting the seed multiple times returns the same samples
  curandSetGeneratorOffset(gen_, 0);
}

CONTROLLER_TEMPLATE
void CONTROLLER::allocateCUDAMemory()
{
  HANDLE_ERROR(cudaMalloc((void**)&initial_state_d_, sizeof(float) * DYN_T::STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&vis_initial_state_d_, sizeof(float) * DYN_T::STATE_DIM));
  HANDLE_ERROR(cudaMalloc((void**)&control_d_, sizeof(float) * DYN_T::CONTROL_DIM * MAX_TIMESTEPS));
  HANDLE_ERROR(cudaMalloc((void**)&output_d_, sizeof(float) * DYN_T::OUTPUT_DIM * MAX_TIMESTEPS * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMalloc((void**)&trajectory_costs_d_, sizeof(float) * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMalloc((void**)&crash_status_d_, sizeof(int) * NUM_ROLLOUTS));
  HANDLE_ERROR(cudaMalloc((void**)&cost_baseline_and_norm_d_, sizeof(float2)));
  cost_baseline_and_norm_ = make_float2(0.0, 0.0);
  CUDA_mem_init_ = true;
}

CONTROLLER_TEMPLATE
void CONTROLLER::resizeSampledControlTrajectories(float perc, int multiplier, int top_num)
{
  int num_sampled_trajectories = perc * NUM_ROLLOUTS + top_num;

  if (sampled_states_CUDA_mem_init_)
  {
    cudaFree(sampled_outputs_d_);
    cudaFree(sampled_costs_d_);
    cudaFree(sampled_crash_status_d_);
    sampled_states_CUDA_mem_init_ = false;
  }
  sampled_trajectories_.resize(num_sampled_trajectories * multiplier, output_trajectory::Zero());
  sampled_costs_.resize(num_sampled_trajectories * multiplier, cost_trajectory::Zero());
  sampled_crash_status_.resize(num_sampled_trajectories * multiplier, crash_status_trajectory::Zero());
  sampler_->setNumVisRollouts(num_sampled_trajectories);
  if (num_sampled_trajectories <= 0)
  {
    return;
  }

  HANDLE_ERROR(cudaMalloc((void**)&sampled_outputs_d_,
                          sizeof(float) * DYN_T::OUTPUT_DIM * MAX_TIMESTEPS * num_sampled_trajectories * multiplier));
  // +1 for terminal cost
  HANDLE_ERROR(cudaMalloc((void**)&sampled_costs_d_,
                          sizeof(float) * (MAX_TIMESTEPS + 1) * num_sampled_trajectories * multiplier));
  HANDLE_ERROR(cudaMalloc((void**)&sampled_crash_status_d_,
                          sizeof(int) * MAX_TIMESTEPS * num_sampled_trajectories * multiplier));
  sampled_states_CUDA_mem_init_ = true;
}

CONTROLLER_TEMPLATE
std::vector<float> CONTROLLER::getSampledNoise()
{
  std::vector<float> vector = std::vector<float>(NUM_ROLLOUTS * getNumTimesteps() * DYN_T::CONTROL_DIM, FLT_MIN);

  HANDLE_ERROR(cudaMemcpyAsync(vector.data(), this->sampler_->getControlSample(0, 0, 0),
                               sizeof(float) * NUM_ROLLOUTS * getNumTimesteps() * DYN_T::CONTROL_DIM,
                               cudaMemcpyDeviceToHost, stream_));
  HANDLE_ERROR(cudaStreamSynchronize(stream_));
  return vector;
}

CONTROLLER_TEMPLATE
void CONTROLLER::chooseAppropriateKernel()
{
  cudaDeviceProp deviceProp;
  HANDLE_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
  unsigned single_kernel_byte_size = mppi::kernels::calcRolloutCombinedKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_dyn_kernel_byte_size = mppi::kernels::calcRolloutDynamicsKernelSharedMemSize(
      this->model_, this->sampler_, this->params_.dynamics_rollout_dim_);
  unsigned split_cost_kernel_byte_size =
      mppi::kernels::calcRolloutCostKernelSharedMemSize(this->cost_, this->sampler_, this->params_.cost_rollout_dim_);
  unsigned vis_single_kernel_byte_size = mppi::kernels::calcVisualizeKernelSharedMemSize(
      this->model_, this->cost_, this->sampler_, this->getNumTimesteps(), this->params_.visualize_dim_);

  bool too_much_mem_single_kernel = single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_vis_kernel = vis_single_kernel_byte_size > deviceProp.sharedMemPerBlock;
  bool too_much_mem_split_kernel = split_dyn_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_split_kernel = too_much_mem_split_kernel || split_cost_kernel_byte_size > deviceProp.sharedMemPerBlock;
  too_much_mem_single_kernel = too_much_mem_single_kernel || too_much_mem_vis_kernel;

  if (too_much_mem_split_kernel && too_much_mem_single_kernel)
  {
    std::string error_msg =
        "There is not enough shared memory on the GPU for either rollout kernel option. The combined rollout kernel "
        "takes " +
        std::to_string(single_kernel_byte_size) + " bytes, the cost rollout kernel takes " +
        std::to_string(split_cost_kernel_byte_size) + " bytes, the dynamics rollout kernel takes " +
        std::to_string(split_dyn_kernel_byte_size) + " bytes, the combined visualization kernel takes " +
        std::to_string(vis_single_kernel_byte_size) + " bytes, and the max is " +
        std::to_string(deviceProp.sharedMemPerBlock) +
        " bytes. Considering lowering the corresponding thread block sizes.";
    throw std::runtime_error(error_msg);
  }
  else if (too_much_mem_single_kernel)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    return;
  }
  else if (too_much_mem_split_kernel)
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    return;
  }

  // Send the nominal control to the device
  this->copyNominalControlToDevice(false);
  state_array zero_state = this->model_->getZeroState();
  // Send zero state to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, zero_state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));
  // Generate noise data
  this->sampler_->generateSamples(1, 0, this->gen_, true);

  float single_kernel_time_ms = std::numeric_limits<float>::infinity();
  float split_kernel_time_ms = std::numeric_limits<float>::infinity();

  // Evaluate each kernel that is applicable
  auto start_single_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_single_kernel; i++)
  {
    mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
        this->crash_status_d_, this->params_.dynamics_rollout_dim_, this->stream_, true);
  }
  auto end_single_kernel_time = std::chrono::steady_clock::now();
  auto start_split_kernel_time = std::chrono::steady_clock::now();
  for (int i = 0; i < this->getNumKernelEvaluations() && !too_much_mem_split_kernel; i++)
  {
    mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
        this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
        this->crash_status_d_, this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_, this->stream_,
        true);
  }
  auto end_split_kernel_time = std::chrono::steady_clock::now();

  // calc times
  if (!too_much_mem_single_kernel)
  {
    single_kernel_time_ms = mppi::math::timeDiffms(end_single_kernel_time, start_single_kernel_time);
  }
  if (!too_much_mem_split_kernel)
  {
    split_kernel_time_ms = mppi::math::timeDiffms(end_split_kernel_time, start_split_kernel_time);
  }
  std::string kernel_choice = "";
  if (split_kernel_time_ms < single_kernel_time_ms)
  {
    this->setKernelChoice(kernelType::USE_SPLIT_KERNELS);
    kernel_choice = "split ";
  }
  else
  {
    this->setKernelChoice(kernelType::USE_SINGLE_KERNEL);
    kernel_choice = "single";
  }
  this->logger_->info("Choosing %s kernel based on split taking %f ms and single taking %f ms after %d iterations\n",
                     kernel_choice.c_str(), split_kernel_time_ms, single_kernel_time_ms,
                     this->getNumKernelEvaluations());
}

CONTROLLER_TEMPLATE
void CONTROLLER::computeControl(const Eigen::Ref<const state_array>& state, int optimization_stride)
{
  this->free_energy_statistics_.real_sys.previousBaseline = this->getBaselineCost();

  // Send the initial condition to the device
  HANDLE_ERROR(cudaMemcpyAsync(this->initial_state_d_, state.data(), DYN_T::STATE_DIM * sizeof(float),
                               cudaMemcpyHostToDevice, this->stream_));

  bool all_collide = false;
  bool force_regenerate_noise = false;
  for (int retry = 0; retry <= this->retry_attempt_limit_; retry++)
  {
    float baseline_prev = 1e8;

    for (int opt_iter = 0; opt_iter < this->getNumIters(); opt_iter++)
    {
      // Send the nominal control to the device
      this->copyNominalControlToDevice(false);

      this->sampler_->generateSamples(optimization_stride, opt_iter, this->gen_, false,
                                      regenerate_noise_ || force_regenerate_noise);

      // Launch the rollout kernel
      if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
      {
        mppi::kernels::launchSplitRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
            this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
            this->getLambda(), this->getAlpha(), this->initial_state_d_, this->output_d_, this->trajectory_costs_d_,
            this->crash_status_d_, this->params_.dynamics_rollout_dim_, this->params_.cost_rollout_dim_,
            this->stream_, false);
      }
      else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
      {
        mppi::kernels::launchRolloutKernel<DYN_T, COST_T, SAMPLING_T>(
            this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), NUM_ROLLOUTS,
            this->getLambda(), this->getAlpha(), this->initial_state_d_, this->trajectory_costs_d_,
            this->crash_status_d_, this->params_.dynamics_rollout_dim_, this->stream_, false);
      }

      // Copy the costs back to the host
      HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                   NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
      HANDLE_ERROR(cudaMemcpyAsync(this->crash_status_.data(), this->crash_status_d_, NUM_ROLLOUTS * sizeof(int),
                                   cudaMemcpyDeviceToHost, this->stream_));
      HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

      all_collide = true;
      for (int i = 0; i < NUM_ROLLOUTS; i++)
      {
        if (!this->crash_status_[i])
        {
          all_collide = false;
          break;
        }
      }

      this->setBaseline(mppi::kernels::computeBaselineCost(this->trajectory_costs_.data(), NUM_ROLLOUTS));

      if (this->getBaselineCost() > baseline_prev + 1)
      {
        this->logger_->debug("Previous Baseline: %f\n         Baseline: %f\n", baseline_prev, this->getBaselineCost());
      }

      baseline_prev = this->getBaselineCost();

      // Launch the norm exponential kernel
      mppi::kernels::launchNormExpKernel(NUM_ROLLOUTS, this->getNormExpThreads(), this->trajectory_costs_d_,
                                         1.0 / this->getLambda(), this->getBaselineCost(), this->stream_, false);
      HANDLE_ERROR(cudaMemcpyAsync(this->trajectory_costs_.data(), this->trajectory_costs_d_,
                                   NUM_ROLLOUTS * sizeof(float), cudaMemcpyDeviceToHost, this->stream_));
      HANDLE_ERROR(cudaStreamSynchronize(this->stream_));

      // Compute the normalizer
      this->setNormalizer(mppi::kernels::computeNormalizer(this->trajectory_costs_.data(), NUM_ROLLOUTS));

      mppi::kernels::computeFreeEnergy(this->free_energy_statistics_.real_sys.freeEnergyMean,
                                       this->free_energy_statistics_.real_sys.freeEnergyVariance,
                                       this->free_energy_statistics_.real_sys.freeEnergyModifiedVariance,
                                       this->trajectory_costs_.data(), NUM_ROLLOUTS, this->getBaselineCost(),
                                       this->getLambda());

      this->sampler_->updateDistributionParamsFromDevice(this->trajectory_costs_d_, this->getNormalizerCost(), 0, false);

      // Transfer the new control to the host
      this->sampler_->setHostOptimalControlSequence(this->control_.data(), 0, true);
    }

    if (!all_collide)
    {
      break;
    }
    if (retry < this->retry_attempt_limit_)
    {
      this->logger_->warning(
          "computeControl(): every sampled rollout predicted collision, wiping the warm-started nominal "
          "control and retrying with fresh noise (attempt %d/%d)\n",
          retry + 1, this->retry_attempt_limit_);
      this->updateImportanceSampler(control_trajectory::Zero());
      force_regenerate_noise = true;
    }
  }
  if (all_collide)
  {
    this->logger_->error(
        "computeControl(): every sampled rollout still predicts collision after %d retries -- proceeding with "
        "best-effort control.\n",
        this->retry_attempt_limit_);
  }

  this->free_energy_statistics_.real_sys.normalizerPercent = this->getNormalizerCost() / NUM_ROLLOUTS;
  this->free_energy_statistics_.real_sys.increase =
      this->getBaselineCost() - this->free_energy_statistics_.real_sys.previousBaseline;
  smoothControlTrajectory();
  {
    const auto cost_params = this->cost_->getParams();
    for (int t = 0; t < this->getNumTimesteps(); t++)
    {
      float& vx = this->control_(C_IND_CLASS(MotionModelParams, VX), t);
      vx = fminf(fmaxf(vx, cost_params.vx_min), cost_params.vx_max);
      float& vy = this->control_(C_IND_CLASS(MotionModelParams, VY), t);
      vy = fminf(fmaxf(vy, -cost_params.vy_max), cost_params.vy_max);
      float& wz = this->control_(C_IND_CLASS(MotionModelParams, WZ), t);
      wz = fminf(fmaxf(wz, -cost_params.wz_max), cost_params.wz_max);
    }
  }
  computeStateTrajectory(state);
  state_array zero_state = this->model_->getZeroState();
  for (int i = 0; i < this->getNumTimesteps(); i++)
  {
    this->model_->enforceConstraints(zero_state, this->control_.col(i));
  }

  // Copy back sampled trajectories
  this->copySampledControlFromDevice(false);
  if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {  // copy initial state to vis initial state for use with visualizeKernel
    HANDLE_ERROR(cudaMemcpyAsync(this->vis_initial_state_d_, this->initial_state_d_, sizeof(float) * DYN_T::STATE_DIM,
                                 cudaMemcpyDeviceToDevice, this->vis_stream_));
  }
  this->copyTopControlFromDevice(true);
}

CONTROLLER_TEMPLATE
void CONTROLLER::calculateSampledStateTrajectories()
{
  int num_sampled_trajectories = this->getTotalSampledTrajectories();

  // control already copied in compute control, so run kernel
  if (this->getKernelChoiceAsEnum() == kernelType::USE_SPLIT_KERNELS)
  {
    mppi::kernels::launchVisualizeCostKernel<COST_T, SAMPLING_T>(
        this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->sampled_outputs_d_, this->sampled_crash_status_d_,
        this->sampled_costs_d_, this->params_.cost_rollout_dim_, this->stream_, false);
  }
  else if (this->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
  {
    mppi::kernels::launchVisualizeKernel<DYN_T, COST_T, SAMPLING_T>(
        this->model_, this->cost_, this->sampler_, this->getDt(), this->getNumTimesteps(), num_sampled_trajectories,
        this->getLambda(), this->getAlpha(), this->vis_initial_state_d_, this->sampled_outputs_d_,
        this->sampled_costs_d_, this->sampled_crash_status_d_, this->params_.visualize_dim_, this->stream_, false);
  }

  for (int i = 0; i < num_sampled_trajectories; i++)
  {
    // set initial state to the first location
    // shifted by one since we do not save the initial state
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_trajectories_[i].data(),
                                 this->sampled_outputs_d_ + i * this->getNumTimesteps() * DYN_T::OUTPUT_DIM,
                                 (this->getNumTimesteps() - 1) * DYN_T::OUTPUT_DIM * sizeof(float),
                                 cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(
        cudaMemcpyAsync(this->sampled_costs_[i].data(), this->sampled_costs_d_ + (i * (this->getNumTimesteps() + 1)),
                        (this->getNumTimesteps() + 1) * sizeof(float), cudaMemcpyDeviceToHost, this->vis_stream_));
    HANDLE_ERROR(cudaMemcpyAsync(this->sampled_crash_status_[i].data(),
                                 this->sampled_crash_status_d_ + (i * this->getNumTimesteps()),
                                 this->getNumTimesteps() * sizeof(int), cudaMemcpyDeviceToHost, this->vis_stream_));
  }
  HANDLE_ERROR(cudaStreamSynchronize(this->vis_stream_));
}

#undef CONTROLLER_TEMPLATE
#undef CONTROLLER

#ifndef MPPI_NUM_ROLLOUTS
#define MPPI_NUM_ROLLOUTS 2048
#endif
template class Controller<MotionModel, MotionModelCost, 100, MPPI_NUM_ROLLOUTS,
    mppi::sampling_distributions::ColoredNoiseDistribution<MotionModel::DYN_PARAMS_T>>;
