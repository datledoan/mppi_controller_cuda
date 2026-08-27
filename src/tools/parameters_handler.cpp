#include "mppi_controller_cuda/tools/parameters_handler.hpp"

namespace mppi
{

ParametersHandler::ParametersHandler(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  const std::string & plugin_name)
{
  node_ = parent;
  plugin_name_ = plugin_name;
  auto node = node_.lock();
  logger_ = node->get_logger();

  initializeParams();
}

ParametersHandler::~ParametersHandler()
{
  auto node = node_.lock();
  if (on_set_param_handler_ && node) {
    node->remove_on_set_parameters_callback(on_set_param_handler_.get());
  }
  on_set_param_handler_.reset();
}

void ParametersHandler::start()
{
  auto node = node_.lock();
  on_set_param_handler_ = node->add_on_set_parameters_callback(
    std::bind(
      &ParametersHandler::dynamicParamsCallback, this,
      std::placeholders::_1));

  auto get_param = getParamGetter(plugin_name_);
  get_param(verbose_, "verbose", false, ParameterType::Static);
}

rcl_interfaces::msg::SetParametersResult
ParametersHandler::dynamicParamsCallback(
  std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  std::lock_guard<std::mutex> lock(parameters_change_mutex_);

  for (auto & pre_cb : pre_callbacks_) {
    pre_cb();
  }

  for (auto & param : parameters) {
    const std::string & param_name = param.get_name();

    if (auto callback = get_param_callbacks_.find(param_name);
      callback != get_param_callbacks_.end())
    {
      callback->second(param);
    } else {
      RCLCPP_WARN(logger_, "Parameter %s not found", param_name.c_str());
    }
  }

  for (auto & post_cb : post_callbacks_) {
    post_cb();
  }

  result.successful = true;
  return result;
}

void ParametersHandler::initializeParams()
{
  auto getParam = getParamGetter(plugin_name_);

  getParam(params_.vx_max, "vx_max", 0.5f);
  getParam(params_.vy_max, "vy_max", 0.0f);
  getParam(params_.vx_min, "vx_min", -0.35f);
  getParam(params_.vx_std, "vx_std", 0.2f);
  getParam(params_.vy_std, "vy_std", 0.2f);
  getParam(params_.wz_std, "wz_std", 0.4f);
  getParam(params_.optimizer_gamma, "optimizer_gamma", 0.015f);
  getParam(
    params_.pure_noise_trajectories_percentage,
    "pure_noise_trajectories_percentage", 0.0f);
  getParam(params_.vx_noise_color_exponent, "vx_noise_color_exponent", 0.0f);
  getParam(params_.vy_noise_color_exponent, "vy_noise_color_exponent", 0.0f);
  getParam(params_.wz_noise_color_exponent, "wz_noise_color_exponent", 1.0f);
  getParam(params_.ax_max, "ax_max", 3.0f);
  getParam(params_.ax_min, "ax_min", -3.0f);
  getParam(params_.ay_max, "ay_max", 3.0f);
  getParam(params_.az_max, "az_max", 3.5f);

  // Static controller parameters
  getParam(params_.wz_max, "wz_max", 1.9f, ParameterType::Static);
  getParam(
    params_.ackermann_constraints_min_turning_r,
    "ackermann_constraints_min_turning_r", 0.2f, ParameterType::Static);

  // Constrain critic parameters
  getParam(params_.constrain_cost_power, "constrain_cost_power", 1);
  getParam(params_.constrain_cost_weight, "constrain_cost_weight", 4.0f);

  // Goal angle critic parameters
  getParam(params_.goal_angle_cost_power, "goal_angle_cost_power", 1);
  getParam(params_.goal_angle_cost_weight, "goal_angle_cost_weight", 3.0f);

  // Goal critic parameters
  getParam(params_.goal_cost_power, "goal_cost_power", 1);
  getParam(params_.goal_cost_weight, "goal_cost_weight", 5.0f);
  getParam(params_.goal_threshold_to_consider, "goal_threshold_to_consider", 1.4f);
  getParam(
    params_.path_align_threshold_to_consider,
    "path_align_threshold_to_consider", 0.5f);
  getParam(
    params_.path_angle_threshold_to_consider,
    "path_angle_threshold_to_consider", 0.5f);
  getParam(
    params_.prefer_forward_threshold_to_consider,
    "prefer_forward_threshold_to_consider", 0.5f);

  // Obstacles critic parameters
  getParam(params_.obstacles_consider_footprint, "obstacles_consider_footprint", false);
  getParam(params_.obstacles_cost_power, "obstacles_cost_power", 1);
  getParam(params_.obstacles_repulsion_weight, "obstacles_repulsion_weight", 1.5f);
  getParam(params_.obstacles_critical_weight, "obstacles_critical_weight", 20.0f);
  getParam(params_.obstacles_collision_cost, "obstacles_collision_cost", 100000.0f);
  getParam(
    params_.obstacles_collision_margin_distance,
    "obstacles_collision_margin_distance", 0.20f);
  getParam(params_.obstacles_near_goal_distance, "obstacles_near_goal_distance", 0.5f);
  getParam(
    params_.obstacles_inflation_radius, "obstacles_inflation_radius", 0.5f,
    ParameterType::Static);
  getParam(
    params_.obstacles_inflation_scale_factor, "obstacles_inflation_scale_factor", 1.0f,
    ParameterType::Static);

  // Cost critic parameters
  getParam(params_.cost_power, "cost_power", 1);
  getParam(params_.cost_weight, "cost_weight", 0.0f);
  getParam(params_.cost_critical_cost, "cost_critical_cost", 300.0f);
  getParam(params_.cost_collision_cost, "cost_collision_cost", 1000000.0f);

  // Path align critic parameters
  getParam(params_.path_align_offset_from_furthest, "path_align_offset_from_furthest", 20);
  getParam(
    params_.path_align_max_path_occupancy_ratio,
    "path_align_max_path_occupancy_ratio", 0.07f);
  getParam(params_.path_align_cost_power, "path_align_cost_power", 1);
  getParam(params_.path_align_cost_weight, "path_align_cost_weight", 10.0f);

  // Path angle critic parameters
  getParam(params_.path_angle_offset_from_furthest, "path_angle_offset_from_furthest", 4);
  getParam(params_.path_angle_cost_power, "path_angle_cost_power", 1);
  getParam(params_.path_angle_cost_weight, "path_angle_cost_weight", 2.2f);
  getParam(
    params_.path_angle_max_angle_to_furthest,
    "path_angle_max_angle_to_furthest", 0.785398f);

  // Path follow critic parameters
  getParam(params_.path_follow_offset_from_furthest, "path_follow_offset_from_furthest", 6);
  getParam(params_.path_follow_cost_power, "path_follow_cost_power", 1);
  getParam(params_.path_follow_cost_weight, "path_follow_cost_weight", 5.0f);

  // Prefer forward / twirling critic parameters
  getParam(params_.prefer_forward_cost_power, "prefer_forward_cost_power", 1);
  getParam(params_.prefer_forward_cost_weight, "prefer_forward_cost_weight", 0.3f);

  getParam(params_.twirling_cost_power, "twirling_cost_power", 1);
  getParam(params_.twirling_cost_weight, "twirling_cost_weight", 0.0f);
  getParam(params_.near_goal_twirling_cost_power, "near_goal_twirling_cost_power", 1);
  getParam(params_.near_goal_twirling_cost_weight, "near_goal_twirling_cost_weight", 0.5f);

  // Velocity deadband critic parameters
  getParam(params_.velocity_deadband_cost_power, "velocity_deadband_cost_power", 1);
  getParam(params_.velocity_deadband_cost_weight, "velocity_deadband_cost_weight", 35.0f);
  getParam(params_.velocity_deadband_velocities_1, "velocity_deadband_velocities_1", 0.0);
  getParam(params_.velocity_deadband_velocities_2, "velocity_deadband_velocities_2", 0.0);
  getParam(params_.velocity_deadband_velocities_3, "velocity_deadband_velocities_3", 0.0);

  // Path handler parameters
  getParam(
    params_.path_handler_max_robot_pose_search_dist,
    "path_handler_max_robot_pose_search_dist", 1.5, ParameterType::Static);
  getParam(
    params_.path_handler_prune_distance, "path_handler_prune_distance", 1.5,
    ParameterType::Static);
  getParam(
    params_.path_handler_transform_tolerance, "path_handler_transform_tolerance", 0.1,
    ParameterType::Static);
  getParam(
    params_.path_handler_enforce_path_inversion, "path_handler_enforce_path_inversion", false,
    ParameterType::Static);
  getParam(
    params_.path_handler_inversion_xy_tolerance, "path_handler_inversion_xy_tolerance", 0.2,
    ParameterType::Static);
  getParam(
    params_.path_handler_inversion_yaw_tolerance, "path_handler_inversion_yaw_tolerance", 0.4,
    ParameterType::Static);

  // Noise generator
  getParam(params_.noise_generator_regenerate_noises, "noise_generator_regenerate_noises", false);

  // Optimizer parameters
  getParam(
    params_.optimizer_iteration_count, "optimizer_iteration_count", 1,
    ParameterType::Static);
  getParam(
    params_.optimizer_retry_attempt_limit, "optimizer_retry_attempt_limit", 1,
    ParameterType::Static);
  getParam(params_.optimizer_model_dt, "optimizer_model_dt", 0.05f, ParameterType::Static);
  getParam(params_.optimizer_time_steps, "optimizer_time_steps", 56, ParameterType::Static);
  getParam(
    params_.optimizer_temperature, "optimizer_temperature", 0.3f,
    ParameterType::Static);
  getParam(
    params_.optimizer_motion_model, "optimizer_motion_model", std::string("DiffDrive"),
    ParameterType::Static);
  getParam(
    params_.noise_generator_use_colored_noise, "noise_generator_use_colored_noise", true,
    ParameterType::Static);

  // Trajectory visualizer parameters.
  getParam(
    params_.trajectory_visualizer_visualize_optimal,
    "trajectory_visualizer_visualize_optimal", true);
  getParam(
    params_.trajectory_visualizer_visualize_samples,
    "trajectory_visualizer_visualize_samples", false);
  getParam(params_.trajectory_visualizer_time_step, "trajectory_visualizer_time_step", 3);
  getParam(
    params_.trajectory_visualizer_samples_percentage,
    "trajectory_visualizer_samples_percentage", 0.05f, ParameterType::Static);
}

}  // namespace mppi
