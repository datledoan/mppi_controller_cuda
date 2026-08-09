#include <stdint.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tf2/utils.h>
#include <angles/angles.h>
#include <nav2_costmap_2d/cost_values.hpp>
#include <nav2_costmap_2d/costmap_filters/filter_values.hpp>
#include "mppi_controller_cuda/controller.hpp"
#include "mppi_controller_cuda/controller/mppi_controller.cuh"
#include "mppi_controller_cuda/costs/motion_model_cost.cuh"
#include "mppi_controller_cuda/dynamics/motion_model.cuh"

namespace mppi_controller_cuda
{

void MPPIControllerCUDA::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  if (!isInitialized())
  {
    auto node = parent.lock();
    parent_ = parent;
    name_ = name;
    logger_ = node->get_logger();
    clock_ = node->get_clock();
    costmap_ros_ = costmap_ros;
    costmap_ = costmap_ros_->getCostmap();
    tf_buffer_ = tf;

    // Parameters handler
    parameters_handler_ = std::make_unique<ParametersHandler>(parent_, name_);
    params_ = parameters_handler_->getParams();

    path_handler_.initialize(parent_, name_, costmap_ros_, tf_buffer_, params_);
    trajectory_visualizer_.onConfigure(parent_, name_, costmap_ros_->getGlobalFrameID(), params_);

    const int DYN_BLOCK_X = 32;
    const int DYN_BLOCK_Y = DYN_T::STATE_DIM;
    dynamics_ = new DYN_T();
    {
      // DiffDrive/Ackermann/Omni
      MotionModelParams dyn_params = dynamics_->getParams();
      dyn_params.min_turning_radius =
        (params_->optimizer_motion_model == "Ackermann") ? params_->ackermann_constraints_min_turning_r : 0.0f;
      dyn_params.is_holonomic = (params_->optimizer_motion_model == "Omni");
      dynamics_->setParams(dyn_params);
    }

    cost_gpu_ = new COST_T();
    {
      auto cost_params = cost_gpu_->getParams();
      refreshCostParams(cost_params);
      cost_params.is_holonomic = (params_->optimizer_motion_model == "Omni");
      cost_params.mppi_num_timesteps = std::min(params_->optimizer_time_steps, NUM_TIMESTEPS);
      cost_gpu_->setParams(cost_params);
    }
    updateObstacleMap(costmap_);

    sampler_ = new SAMPLING_T();
    auto sampler_params = sampler_->getParams();
    refreshSamplerParams(sampler_params);
    sampler_->setParams(sampler_params);

    CONTROLLER_T::TEMPLATED_PARAMS controller_params;
    controller_params.dt_ = params_->optimizer_model_dt;
    controller_params.lambda_ = params_->optimizer_temperature;
    controller_params.num_iters_ = params_->optimizer_iteration_count;
    controller_params.num_timesteps_ = std::min(params_->optimizer_time_steps, NUM_TIMESTEPS);
    controller_params.dynamics_rollout_dim_ = dim3(DYN_BLOCK_X, DYN_BLOCK_Y, 1);
    controller_params.cost_rollout_dim_ = dim3(32, 1, 1);
    controller_params.slide_control_scale_ = CONTROLLER_T::control_array::Ones();
    mppi_controller_ = new CONTROLLER_T(dynamics_, cost_gpu_, sampler_, controller_params);
    mppi_controller_->setRetryAttemptLimit(params_->optimizer_retry_attempt_limit);

    // Force split-kernel mode
    mppi_controller_->setKernelChoice(kernelType::USE_SPLIT_KERNELS);

    mppi_controller_->setPercentageSampledControlTrajectories(params_->trajectory_visualizer_samples_percentage);

    initialized_ = true;
    RCLCPP_INFO(logger_, "Configured MPPI Controller: %s", name_.c_str());
    RCLCPP_INFO(logger_, "[MPPI] Noise generator: %s (noise_generator_use_colored_noise=%s)",
             params_->noise_generator_use_colored_noise ? "ColoredNoiseDistribution (FFT-based)" : "plain Gaussian (i.i.d.)",
             params_->noise_generator_use_colored_noise ? "true" : "false");
  }
  else
  {
    RCLCPP_WARN(logger_, "[MPPI] MPPIControllerCUDA already configured.");
  }
}

MPPIControllerCUDA::~MPPIControllerCUDA()
{
  delete mppi_controller_;
  delete sampler_;
  delete cost_gpu_;
  delete dynamics_;
}

void MPPIControllerCUDA::cleanup()
{
  trajectory_visualizer_.onCleanup();
  parameters_handler_.reset();
  RCLCPP_INFO(logger_, "Cleaned up MPPI Controller: %s", name_.c_str());
}

void MPPIControllerCUDA::activate()
{
  trajectory_visualizer_.onActivate();
  parameters_handler_->start();
  RCLCPP_INFO(logger_, "Activated MPPI Controller: %s", name_.c_str());
}

void MPPIControllerCUDA::deactivate()
{
  trajectory_visualizer_.onDeactivate();
  RCLCPP_INFO(logger_, "Deactivated MPPI Controller: %s", name_.c_str());
}

float MPPIControllerCUDA::effectiveVxMax() const
{
  if (speed_limit_ == nav2_costmap_2d::NO_SPEED_LIMIT)
  {
    return params_->vx_max;
  }
  return speed_limit_is_percentage_ ?
    static_cast<float>(params_->vx_max * speed_limit_ / 100.0) :
    static_cast<float>(speed_limit_);
}

void MPPIControllerCUDA::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  speed_limit_ = speed_limit;
  speed_limit_is_percentage_ = percentage;
}

void MPPIControllerCUDA::updateObstacleMap(nav2_costmap_2d::Costmap2D * costmap)
{
  const int size_x = costmap->getSizeInCellsX();
  const int size_y = costmap->getSizeInCellsY();
  std::vector<float> costmap_data(size_x * size_y);
  const unsigned char* char_map = costmap->getCharMap();
  for (int i = 0; i < size_x * size_y; i++)
  {
    costmap_data[i] = static_cast<float>(char_map[i]);
  }
  cost_gpu_->updateObstacleMap(costmap_data, size_x, size_y, static_cast<float>(costmap->getOriginX()),
                               static_cast<float>(costmap->getOriginY()),
                               static_cast<float>(costmap->getResolution()));
}

void MPPIControllerCUDA::refreshCostParams(MotionModelCostParams& cost_params)
{
  cost_params.goal_weight = params_->goal_cost_weight;
  cost_params.goal_power = params_->goal_cost_power;
  cost_params.goal_angle_weight = params_->goal_angle_cost_weight;
  cost_params.goal_angle_power = params_->goal_angle_cost_power;
  cost_params.prefer_forward_weight = params_->prefer_forward_cost_weight;
  cost_params.prefer_forward_power = params_->prefer_forward_cost_power;
  cost_params.twirling_weight = params_->twirling_cost_weight;
  cost_params.twirling_power = params_->twirling_cost_power;
  cost_params.near_goal_twirling_weight = params_->near_goal_twirling_cost_weight;
  cost_params.near_goal_twirling_power = params_->near_goal_twirling_cost_power;
  cost_params.velocity_deadband_weight = params_->velocity_deadband_cost_weight;
  cost_params.velocity_deadband_power = params_->velocity_deadband_cost_power;
  cost_params.deadband_vx = params_->velocity_deadband_velocities_1;
  cost_params.deadband_vy = params_->velocity_deadband_velocities_2;
  cost_params.deadband_wz = params_->velocity_deadband_velocities_3;
  cost_params.constraint_weight = params_->constrain_cost_weight;
  cost_params.constraint_power = params_->constrain_cost_power;
  cost_params.vx_max = effectiveVxMax();
  cost_params.vx_min = params_->vx_min;
  cost_params.vy_max = params_->vy_max;
  cost_params.wz_max = params_->wz_max;
  cost_params.obstacle_scale_factor = params_->obstacles_inflation_scale_factor;
  cost_params.obstacle_inflation_radius = params_->obstacles_inflation_radius;
  cost_params.obstacle_collision_margin = params_->obstacles_collision_margin_distance;
  cost_params.obstacle_collision_cost = params_->obstacles_collision_cost;
  cost_params.obstacle_traj_weight = params_->obstacles_critical_weight;
  cost_params.obstacle_repulsion_weight = params_->obstacles_repulsion_weight;
  cost_params.obstacle_power = params_->obstacles_cost_power;
  cost_params.cost_weight = params_->cost_weight;
  cost_params.cost_power = params_->cost_power;
  cost_params.cost_critical_cost = params_->cost_critical_cost;
  cost_params.cost_collision_cost = params_->cost_collision_cost;
  cost_params.path_align_offset_from_furthest = params_->path_align_offset_from_furthest;
  cost_params.path_align_weight = params_->path_align_cost_weight;
  cost_params.path_align_power = params_->path_align_cost_power;
  cost_params.path_angle_weight = params_->path_angle_cost_weight;
  cost_params.path_angle_power = params_->path_angle_cost_power;
}

void MPPIControllerCUDA::refreshDynamicsParams(MotionModelParams& dyn_params)
{
  dyn_params.ax_max = params_->ax_max;
  dyn_params.ax_min = params_->ax_min;
  dyn_params.ay_max = params_->ay_max;
  dyn_params.az_max = params_->az_max;
}

void MPPIControllerCUDA::refreshSamplerParams(SAMPLING_T::SAMPLING_PARAMS_T& sampler_params, bool near_goal)
{
  (void)near_goal;
  sampler_params.std_dev[C_IND_CLASS(MotionModelParams, VX)] = params_->vx_std;
  sampler_params.std_dev[C_IND_CLASS(MotionModelParams, VY)] = params_->vy_std;
  sampler_params.std_dev[C_IND_CLASS(MotionModelParams, WZ)] = params_->wz_std;

  const float control_cost_coeff = -params_->optimizer_gamma / params_->optimizer_temperature;
  sampler_params.control_cost_coeff[C_IND_CLASS(MotionModelParams, VX)] = control_cost_coeff;
  sampler_params.control_cost_coeff[C_IND_CLASS(MotionModelParams, VY)] = control_cost_coeff;
  sampler_params.control_cost_coeff[C_IND_CLASS(MotionModelParams, WZ)] = control_cost_coeff;

  sampler_params.pure_noise_trajectories_percentage = params_->pure_noise_trajectories_percentage;

  sampler_params.min_control[C_IND_CLASS(MotionModelParams, VX)] = -std::numeric_limits<float>::max();
  sampler_params.max_control[C_IND_CLASS(MotionModelParams, VX)] = std::numeric_limits<float>::max();
  sampler_params.min_control[C_IND_CLASS(MotionModelParams, VY)] = -std::numeric_limits<float>::max();
  sampler_params.max_control[C_IND_CLASS(MotionModelParams, VY)] = std::numeric_limits<float>::max();
  sampler_params.min_control[C_IND_CLASS(MotionModelParams, WZ)] = -std::numeric_limits<float>::max();
  sampler_params.max_control[C_IND_CLASS(MotionModelParams, WZ)] = std::numeric_limits<float>::max();

  // Runtime noise-generator switch
  sampler_params.use_colored_noise = params_->noise_generator_use_colored_noise;

  // Colored-noise spectral exponents
  sampler_params.exponents[C_IND_CLASS(MotionModelParams, VX)] = params_->vx_noise_color_exponent;
  sampler_params.exponents[C_IND_CLASS(MotionModelParams, VY)] = params_->vy_noise_color_exponent;
  sampler_params.exponents[C_IND_CLASS(MotionModelParams, WZ)] = params_->wz_noise_color_exponent;
}

void MPPIControllerCUDA::updateFootprint(MotionModelCostParams& cost_params)
{
  if (!params_->obstacles_consider_footprint)
  {
    cost_params.footprint_length = 0;
    return;
  }
  const std::vector<geometry_msgs::msg::Point>& footprint = costmap_ros_->getRobotFootprint();
  const size_t max_points = static_cast<size_t>(MotionModelCostParams::MAX_FOOTPRINT_POINTS);
  const size_t stride = (footprint.size() > max_points) ?
    (footprint.size() + max_points - 1) / max_points : 1;
  if (footprint.size() > max_points)
  {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
      "[MPPI] Robot footprint has %zu points, subsampling to fit MotionModelCostParams::MAX_FOOTPRINT_POINTS (%d)",
      footprint.size(), MotionModelCostParams::MAX_FOOTPRINT_POINTS);
  }
  int footprint_length = 0;
  for (size_t i = 0; i < footprint.size() && footprint_length < MotionModelCostParams::MAX_FOOTPRINT_POINTS; i += stride)
  {
    cost_params.footprint_x[footprint_length] = footprint[i].x;
    cost_params.footprint_y[footprint_length] = footprint[i].y;
    ++footprint_length;
  }
  cost_params.footprint_length = footprint_length;
}

void MPPIControllerCUDA::visualize(nav_msgs::msg::Path transformed_plan, const builtin_interfaces::msg::Time& cmd_stamp)
{
  const int num_timesteps = mppi_controller_->getNumTimesteps();

  if (params_->trajectory_visualizer_visualize_optimal)
  {
    CONTROLLER_T::output_trajectory output_seq = mppi_controller_->getTargetOutputSeq();
    Eigen::ArrayXXf optimal_arr(num_timesteps, 3);
    for (int t = 0; t < num_timesteps; t++)
    {
      optimal_arr(t, 0) = output_seq(O_IND_CLASS(MotionModelParams, X), t);
      optimal_arr(t, 1) = output_seq(O_IND_CLASS(MotionModelParams, Y), t);
      optimal_arr(t, 2) = output_seq(O_IND_CLASS(MotionModelParams, YAW), t);
    }
    trajectory_visualizer_.add(optimal_arr, "Optimal Trajectory", cmd_stamp);
  }

  if (params_->trajectory_visualizer_visualize_samples)
  {
    if (mppi_controller_->getKernelChoiceAsEnum() == kernelType::USE_SINGLE_KERNEL)
    {
      RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
        "[MPPI] trajectory_visualizer_visualize_samples is enabled but this GPU chose single-kernel rollout mode "
        "(sample trajectories aren't available in that mode) -- skipping /trajectories.");
    }
    else
    {
      mppi_controller_->calculateSampledStateTrajectories();
      std::vector<CONTROLLER_T::output_trajectory> sampled = mppi_controller_->getSampledOutputTrajectories();
      if (!sampled.empty())
      {
        const int valid_timesteps = num_timesteps - 1;
        Eigen::ArrayXXf sample_x(sampled.size(), valid_timesteps);
        Eigen::ArrayXXf sample_y(sampled.size(), valid_timesteps);
        for (size_t i = 0; i < sampled.size(); i++)
        {
          for (int t = 0; t < valid_timesteps; t++)
          {
            sample_x(i, t) = sampled[i](O_IND_CLASS(MotionModelParams, X), t);
            sample_y(i, t) = sampled[i](O_IND_CLASS(MotionModelParams, Y), t);
          }
        }
        trajectory_visualizer_.add(sample_x, sample_y, "Candidate Trajectories");
      }
    }
  }

  trajectory_visualizer_.visualize(transformed_plan);
}

geometry_msgs::msg::TwistStamped MPPIControllerCUDA::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped& pose,
  const geometry_msgs::msg::Twist& velocity,
  nav2_core::GoalChecker* /*goal_checker*/)
{
  if(!initialized_)
  {
    throw std::runtime_error("[MPPI] MPPIControllerCUDA has not been configured");
  }
  std::lock_guard<std::mutex> param_lock(*parameters_handler_->getLock());
  geometry_msgs::msg::Pose goal = path_handler_.getTransformedGoal(pose.header.stamp).pose;
  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));
  nav_msgs::msg::Path transformed_plan = path_handler_.transformPath(pose);
  const geometry_msgs::msg::Twist & speed = velocity;

  DYN_T::state_array x = DYN_T::state_array::Zero();
  x[S_IND_CLASS(MotionModelParams, X)] = pose.pose.position.x;
  x[S_IND_CLASS(MotionModelParams, Y)] = pose.pose.position.y;
  x[S_IND_CLASS(MotionModelParams, YAW)] = tf2::getYaw(pose.pose.orientation);

  auto cost_params = cost_gpu_->getParams();
  const size_t num_path_poses = transformed_plan.poses.size();
  const size_t max_points = static_cast<size_t>(MotionModelCostParams::MAX_PATH_POINTS);
  const size_t stride = (num_path_poses > max_points) ?
    (num_path_poses + max_points - 1) / max_points : 1;
  if (num_path_poses > max_points)
  {
    RCLCPP_WARN_THROTTLE(logger_, *clock_, 5000,
      "[MPPI] Transformed path has %zu poses, subsampling to fit MotionModelCostParams::MAX_PATH_POINTS (%d)",
      num_path_poses, MotionModelCostParams::MAX_PATH_POINTS);
  }
  int path_length = 0;
  float cumulative_path_dist = 0.0f;
  for (size_t i = 0; i < num_path_poses && path_length < MotionModelCostParams::MAX_PATH_POINTS; i += stride)
  {
    const double wx = transformed_plan.poses[i].pose.position.x;
    const double wy = transformed_plan.poses[i].pose.position.y;
    if (path_length > 0)
    {
      const double ddx = wx - cost_params.path_x[path_length - 1];
      const double ddy = wy - cost_params.path_y[path_length - 1];
      cumulative_path_dist += static_cast<float>(std::sqrt(ddx * ddx + ddy * ddy));
    }
    cost_params.path_x[path_length] = wx;
    cost_params.path_y[path_length] = wy;
    cost_params.path_integrated_distances[path_length] = cumulative_path_dist;
    unsigned int mx = 0, my = 0;
    bool valid = true;
    if (costmap->worldToMap(wx, wy, mx, my))
    {
      valid = costmap->getCost(mx, my) < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
    }
    cost_params.path_valid[path_length] = valid;
    ++path_length;
  }
  cost_params.path_length = path_length;
  cost_params.path_follow_weight = params_->path_follow_cost_weight;
  cost_params.path_follow_power = params_->path_follow_cost_power;

  refreshCostParams(cost_params);

  // Rate-limit params
  auto dyn_params = dynamics_->getParams();
  refreshDynamicsParams(dyn_params);
  dyn_params.current_vx = speed.linear.x;
  dyn_params.current_vy = speed.linear.y;
  dyn_params.current_wz = speed.angular.z;
  dynamics_->setParams(dyn_params);

  double goal_dx = pose.pose.position.x - goal.position.x;
  double goal_dy = pose.pose.position.y - goal.position.y;
  const double dist_to_goal = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy);
  const bool near_goal = dist_to_goal < params_->goal_threshold_to_consider;
  const bool path_align_near_goal = dist_to_goal < params_->path_align_threshold_to_consider;
  const bool path_angle_near_goal = dist_to_goal < params_->path_angle_threshold_to_consider;
  const bool prefer_forward_near_goal = dist_to_goal < params_->prefer_forward_threshold_to_consider;
  const bool obstacles_near_goal = dist_to_goal < params_->obstacles_near_goal_distance;

  auto sampler_params = sampler_->getParams();
  refreshSamplerParams(sampler_params, near_goal);
  sampler_->setParams(sampler_params);

  int furthest_reached_idx = path_length > 0 ? std::min(last_true_furthest_reached_idx_, path_length - 1) : 0;
  if (path_length > 1)
  {
    const bool is_holonomic = dynamics_->getParams().is_holonomic;
    CONTROLLER_T::control_trajectory nominal_controls = mppi_controller_->getControlSeq();
    float sim_x = x[S_IND_CLASS(MotionModelParams, X)];
    float sim_y = x[S_IND_CLASS(MotionModelParams, Y)];
    float sim_yaw = x[S_IND_CLASS(MotionModelParams, YAW)];
    const float dt = params_->optimizer_model_dt;
    const int horizon = std::min(params_->optimizer_time_steps, static_cast<int>(NUM_TIMESTEPS));
    for (int t = 0; t < horizon; t++)
    {
      float vx = params_->vx_max;
      float vy = nominal_controls.col(t)(C_IND_CLASS(MotionModelParams, VY));
      float wz = nominal_controls.col(t)(C_IND_CLASS(MotionModelParams, WZ));
      sim_x += vx * std::cos(sim_yaw) * dt;
      sim_y += vx * std::sin(sim_yaw) * dt;
      if (is_holonomic)
      {
        sim_x -= vy * std::sin(sim_yaw) * dt;
        sim_y += vy * std::cos(sim_yaw) * dt;
      }
      sim_yaw += wz * dt;
      // Nearest path point searched forward-only from the last match, mirroring
      // findClosestPathPt's monotonic search (Euclidean here, not arc-length).
      float best_d = std::numeric_limits<float>::max();
      int best_idx = furthest_reached_idx;
      for (int j = furthest_reached_idx; j < path_length; j++)
      {
        float dxp = cost_params.path_x[j] - sim_x;
        float dyp = cost_params.path_y[j] - sim_y;
        float d = dxp * dxp + dyp * dyp;
        if (d < best_d)
        {
          best_d = d;
          best_idx = j;
        }
      }
      if (!cost_params.path_valid[best_idx])
      {
        break;
      }
      furthest_reached_idx = best_idx;
    }
  }
  cost_params.furthest_reached_idx = furthest_reached_idx;

  if (furthest_reached_idx > 0)
  {
    int invalid_ctr = 0;
    for (int i = 0; i < furthest_reached_idx; i++)
    {
      if (!cost_params.path_valid[i])
      {
        ++invalid_ctr;
      }
    }
    float occupancy_ratio = static_cast<float>(invalid_ctr) / static_cast<float>(furthest_reached_idx);
    if (invalid_ctr > 2 && occupancy_ratio > params_->path_align_max_path_occupancy_ratio)
    {
      cost_params.path_align_weight = 0.0f;
    }
  }

  if (path_length > 0)
  {
    int pf_idx = std::min(furthest_reached_idx + params_->path_follow_offset_from_furthest, path_length - 1);
    while (!cost_params.path_valid[pf_idx] && pf_idx < path_length - 1)
    {
      ++pf_idx;
    }
    cost_params.path_follow_target_x = cost_params.path_x[pf_idx];
    cost_params.path_follow_target_y = cost_params.path_y[pf_idx];
  }

  cost_params.path_angle_active = false;
  if (path_length > 0)
  {
    int offsetted_idx = std::min(furthest_reached_idx + params_->path_angle_offset_from_furthest, path_length - 1);
    float target_x = cost_params.path_x[offsetted_idx];
    float target_y = cost_params.path_y[offsetted_idx];
    float yaw_to_target = std::atan2(target_y - pose.pose.position.y, target_x - pose.pose.position.x);
    float angle_err = std::fabs(
      angles::shortest_angular_distance(tf2::getYaw(pose.pose.orientation), yaw_to_target));
    cost_params.path_angle_active = angle_err >= params_->path_angle_max_angle_to_furthest;
    cost_params.path_angle_target_x = target_x;
    cost_params.path_angle_target_y = target_y;
  }

  cost_params.near_goal = near_goal;
  cost_params.path_align_near_goal = path_align_near_goal;
  cost_params.path_angle_near_goal = path_angle_near_goal;
  cost_params.prefer_forward_near_goal = prefer_forward_near_goal;
  cost_params.obstacles_near_goal = obstacles_near_goal;
  cost_params.goal_x = goal.position.x;
  cost_params.goal_y = goal.position.y;
  cost_params.goal_yaw = tf2::getYaw(goal.orientation);

  updateObstacleMap(costmap);
  updateFootprint(cost_params);

  cost_gpu_->setParams(cost_params);

  mppi_controller_->setRegenerateNoise(!noise_generated_once_ || params_->noise_generator_regenerate_noises);
  noise_generated_once_ = true;

  // Calculate control and apply the first timestep of the resulting sequence
  mppi_controller_->computeControl(x, 1);

  last_true_furthest_reached_idx_ =
      std::max(last_true_furthest_reached_idx_, cost_gpu_->getDeviceFurthestReachedIdx());

  {
    const int horizon = std::min(params_->optimizer_time_steps, static_cast<int>(NUM_TIMESTEPS));
    CONTROLLER_T::control_trajectory nominal = mppi_controller_->getControlSeq();
    const float max_delta_vx = params_->optimizer_model_dt * params_->ax_max;
    const float min_delta_vx = params_->optimizer_model_dt * params_->ax_min;
    const float max_delta_wz = params_->optimizer_model_dt * params_->az_max;
    float vx_last = std::clamp(nominal(C_IND_CLASS(MotionModelParams, VX), 0), params_->vx_min, effectiveVxMax());
    float wz_last = std::clamp(nominal(C_IND_CLASS(MotionModelParams, WZ), 0), -params_->wz_max, params_->wz_max);
    nominal(C_IND_CLASS(MotionModelParams, VX), 0) = vx_last;
    nominal(C_IND_CLASS(MotionModelParams, WZ), 0) = wz_last;
    for (int t = 1; t < horizon; t++)
    {
      float vx_curr = std::clamp(nominal(C_IND_CLASS(MotionModelParams, VX), t), params_->vx_min, effectiveVxMax());
      vx_curr = std::clamp(vx_curr, vx_last + min_delta_vx, vx_last + max_delta_vx);
      nominal(C_IND_CLASS(MotionModelParams, VX), t) = vx_curr;
      vx_last = vx_curr;

      float wz_curr = std::clamp(nominal(C_IND_CLASS(MotionModelParams, WZ), t), -params_->wz_max, params_->wz_max);
      wz_curr = std::clamp(wz_curr, wz_last - max_delta_wz, wz_last + max_delta_wz);
      nominal(C_IND_CLASS(MotionModelParams, WZ), t) = wz_curr;
      wz_last = wz_curr;
    }
    mppi_controller_->updateImportanceSampler(nominal);
  }

  CONTROLLER_T::control_trajectory control_sequence = mppi_controller_->getControlSeq();
  CONTROLLER_T::control_array u0 = control_sequence.col(0);

  // Warm-start the next cycle
  mppi_controller_->slideControlSequence(1);

  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.stamp = pose.header.stamp;
  cmd_vel.header.frame_id = pose.header.frame_id;
  // Hard-clamp
  cmd_vel.twist.linear.x = std::clamp(u0[C_IND_CLASS(MotionModelParams, VX)], params_->vx_min, effectiveVxMax());
  // Always set linear.y for non-holonomic modes
  cmd_vel.twist.linear.y = std::clamp(u0[C_IND_CLASS(MotionModelParams, VY)], -params_->vy_max, params_->vy_max);
  cmd_vel.twist.angular.z = std::clamp(u0[C_IND_CLASS(MotionModelParams, WZ)], -params_->wz_max, params_->wz_max);

  // Publish debug visualization
  visualize(transformed_plan, cmd_vel.header.stamp);

  return cmd_vel;
}

void MPPIControllerCUDA::setPlan(const nav_msgs::msg::Path& path)
{
  if(!initialized_)
  {
    RCLCPP_ERROR(logger_, "[MPPI] MPPIControllerCUDA has not been configured, please call configure() before using this planner");
    return;
  }
  path_handler_.setPath(path);
  last_true_furthest_reached_idx_ = 0;
}

}  // namespace mppi_controller_cuda

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(mppi_controller_cuda::MPPIControllerCUDA, nav2_core::Controller)
