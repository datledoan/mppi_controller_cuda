#include "mppi_controller_cuda/tools/parameters_handler.hpp"

namespace mppi
{

ParametersHandler::ParametersHandler(std::shared_ptr<ros::NodeHandle> node, std::shared_ptr<ros::NodeHandle> private_node)
{
  node_ = node;
  private_node_ = private_node;

  // Controller parameters
  private_node_->param<float>("vx_max", params_.vx_max, 0.5f);
  private_node_->param<float>("vy_max", params_.vy_max, 0.0f);
  private_node_->param<float>("vx_min", params_.vx_min, -0.35f);
  private_node_->param<float>("wz_max", params_.wz_max, 1.9f);
  private_node_->param<float>("xy_goal_tolerance", params_.xy_goal_tolerance, 0.25f);
  private_node_->param<float>("yaw_goal_tolerance", params_.yaw_goal_tolerance, 0.25f);
  private_node_->param<double>("theta_stopped_vel", params_.theta_stopped_vel, 0.1);
  private_node_->param<double>("trans_stopped_vel", params_.trans_stopped_vel, 0.1);
  node_->param<double>("controller_frequency", params_.controller_frequency, 20.0f);
  private_node_->param<float>("ackermann_constraints_min_turning_r", params_.ackermann_constraints_min_turning_r, 0.2f);

  // Acceleration/rate-of-change limits
  private_node_->param<float>("ax_max", params_.ax_max, 3.0f);
  private_node_->param<float>("ax_min", params_.ax_min, -3.0f);
  private_node_->param<float>("ay_max", params_.ay_max, 3.0f);
  private_node_->param<float>("az_max", params_.az_max, 3.5f);

  // Sampler noise std_dev
  private_node_->param<float>("vx_std", params_.vx_std, 0.2f);
  private_node_->param<float>("vy_std", params_.vy_std, 0.2f);
  private_node_->param<float>("wz_std", params_.wz_std, 0.4f);
  private_node_->param<float>("optimizer_gamma", params_.optimizer_gamma, 0.015f);
  private_node_->param<float>("pure_noise_trajectories_percentage", params_.pure_noise_trajectories_percentage, 0.0f);
  private_node_->param<float>("vx_noise_color_exponent", params_.vx_noise_color_exponent, 0.0f);
  private_node_->param<float>("vy_noise_color_exponent", params_.vy_noise_color_exponent, 0.0f);
  private_node_->param<float>("wz_noise_color_exponent", params_.wz_noise_color_exponent, 1.0f);

  // Constrain critic parameters
  private_node_->param<int>("constrain_cost_power", params_.constrain_cost_power, 1);
  private_node_->param<float>("constrain_cost_weight", params_.constrain_cost_weight, 4.0f);

  // Goal angle critic parameters
  private_node_->param<int>("goal_angle_cost_power", params_.goal_angle_cost_power, 1);
  private_node_->param<float>("goal_angle_cost_weight", params_.goal_angle_cost_weight, 3.0f);

  // Goal critic parameters
  private_node_->param<int>("goal_cost_power", params_.goal_cost_power, 1);
  private_node_->param<float>("goal_cost_weight", params_.goal_cost_weight, 5.0f);
  private_node_->param<float>("goal_threshold_to_consider", params_.goal_threshold_to_consider, 1.4f);
  private_node_->param<float>("path_align_threshold_to_consider", params_.path_align_threshold_to_consider, 0.5f);
  private_node_->param<float>("path_angle_threshold_to_consider", params_.path_angle_threshold_to_consider, 0.5f);
  private_node_->param<float>("prefer_forward_threshold_to_consider", params_.prefer_forward_threshold_to_consider,
                              0.5f);

  // Obstacles critic parameters
  private_node_->param<bool>("obstacles_consider_footprint", params_.obstacles_consider_footprint, false);
  private_node_->param<int>("obstacles_cost_power", params_.obstacles_cost_power, 1);
  private_node_->param<float>("obstacles_repulsion_weight", params_.obstacles_repulsion_weight, 1.5f);
  private_node_->param<float>("obstacles_critical_weight", params_.obstacles_critical_weight, 20.0f);
  private_node_->param<float>("obstacles_collision_cost", params_.obstacles_collision_cost, 100000.0f);
  private_node_->param<float>("obstacles_collision_margin_distance", params_.obstacles_collision_margin_distance, 0.20f);
  if (!node_->getParam("local_costmap/inflation_layer/inflation_radius", params_.obstacles_inflation_radius) &&
      !node_->getParam("local_costmap/inflation_radius", params_.obstacles_inflation_radius))
  {
    params_.obstacles_inflation_radius = 0.5f;
  }
  if (!node_->getParam("local_costmap/inflation_layer/cost_scaling_factor", params_.obstacles_inflation_scale_factor) &&
      !node_->getParam("local_costmap/cost_scaling_factor", params_.obstacles_inflation_scale_factor))
  {
    params_.obstacles_inflation_scale_factor = 1.0f;
  }
  private_node_->param<float>("obstacles_near_goal_distance", params_.obstacles_near_goal_distance, 0.5f);

  // Cost critic parameters
  private_node_->param<int>("cost_power", params_.cost_power, 1);
  private_node_->param<float>("cost_weight", params_.cost_weight, 0.0f);
  private_node_->param<float>("cost_critical_cost", params_.cost_critical_cost, 300.0f);
  private_node_->param<float>("cost_collision_cost", params_.cost_collision_cost, 1000000.0f);

  // Path align critic parameters
  private_node_->param<int>("path_align_offset_from_furthest", params_.path_align_offset_from_furthest, 20);
  private_node_->param<float>("path_align_max_path_occupancy_ratio", params_.path_align_max_path_occupancy_ratio, 0.07f);
  private_node_->param<int>("path_align_cost_power", params_.path_align_cost_power, 1);
  private_node_->param<float>("path_align_cost_weight", params_.path_align_cost_weight, 10.0f);

  // Path angle critic parameters
  private_node_->param<int>("path_angle_offset_from_furthest", params_.path_angle_offset_from_furthest, 4);
  private_node_->param<int>("path_angle_cost_power", params_.path_angle_cost_power, 1);
  private_node_->param<float>("path_angle_cost_weight", params_.path_angle_cost_weight, 2.2f);
  private_node_->param<float>("path_angle_max_angle_to_furthest", params_.path_angle_max_angle_to_furthest, 0.785398f);

  // Path follow critic parameters
  private_node_->param<int>("path_follow_offset_from_furthest", params_.path_follow_offset_from_furthest, 6);
  private_node_->param<int>("path_follow_cost_power", params_.path_follow_cost_power, 1);
  private_node_->param<float>("path_follow_cost_weight", params_.path_follow_cost_weight, 5.0f);

  private_node_->param<int>("prefer_forward_cost_power", params_.prefer_forward_cost_power, 1);
  private_node_->param<float>("prefer_forward_cost_weight", params_.prefer_forward_cost_weight, 0.3f);

  private_node_->param<int>("twirling_cost_power", params_.twirling_cost_power, 1);
  private_node_->param<float>("twirling_cost_weight", params_.twirling_cost_weight, 0.0f);
  private_node_->param<int>("near_goal_twirling_cost_power", params_.near_goal_twirling_cost_power, 1);
  private_node_->param<float>("near_goal_twirling_cost_weight", params_.near_goal_twirling_cost_weight, 0.5f);

  // Velocity deadband critic parameters
  private_node_->param<int>("velocity_deadband_cost_power", params_.velocity_deadband_cost_power, 1);
  private_node_->param<float>("velocity_deadband_cost_weight", params_.velocity_deadband_cost_weight, 35.0f);
  private_node_->param<double>("velocity_deadband_velocities_1", params_.velocity_deadband_velocities_1, 0.0);
  private_node_->param<double>("velocity_deadband_velocities_2", params_.velocity_deadband_velocities_2, 0.0);
  private_node_->param<double>("velocity_deadband_velocities_3", params_.velocity_deadband_velocities_3, 0.0);

  // Path handler parameters
  double map_height, map_width;
  node_->param<double>("local_costmap/height", map_height, 3.0);
  node_->param<double>("local_costmap/width", map_width, 3.0);
  double max_costmap_dist = std::max(map_height, map_width) * 0.5;
  private_node_->param<double>("path_handler_max_robot_pose_search_dist", params_.path_handler_max_robot_pose_search_dist, max_costmap_dist);
  private_node_->param<double>("path_handler_prune_distance", params_.path_handler_prune_distance, 1.5);
  private_node_->param<double>("path_handler_transform_tolerance", params_.path_handler_transform_tolerance, 0.1);
  private_node_->param<bool>("path_handler_enforce_path_inversion", params_.path_handler_enforce_path_inversion, false);
  private_node_->param<double>("path_handler_inversion_xy_tolerance", params_.path_handler_inversion_xy_tolerance, 0.2);
  private_node_->param<double>("path_handler_inversion_yaw_tolerance", params_.path_handler_inversion_yaw_tolerance, 0.4);

  // Noise generator
  private_node_->param<bool>("noise_generator_regenerate_noises", params_.noise_generator_regenerate_noises, false);

  // Optimizer parameters
  private_node_->param<int>("optimizer_iteration_count", params_.optimizer_iteration_count, 1);
  private_node_->param<int>("optimizer_retry_attempt_limit", params_.optimizer_retry_attempt_limit, 1);
  private_node_->param<float>("optimizer_model_dt", params_.optimizer_model_dt, 0.05f);
  private_node_->param<int>("optimizer_time_steps", params_.optimizer_time_steps, 56);
  private_node_->param<float>("optimizer_temperature", params_.optimizer_temperature, 0.3f);
  private_node_->param<std::string>("optimizer_motion_model", params_.optimizer_motion_model, "DiffDrive");
  private_node_->param<bool>("noise_generator_use_colored_noise", params_.noise_generator_use_colored_noise, true);

  private_node_->param<bool>("trajectory_visualizer_visualize_optimal", params_.trajectory_visualizer_visualize_optimal, true);
  private_node_->param<bool>("trajectory_visualizer_visualize_samples", params_.trajectory_visualizer_visualize_samples, false);
  private_node_->param<int>("trajectory_visualizer_time_step", params_.trajectory_visualizer_time_step, 3);
  private_node_->param<float>("trajectory_visualizer_samples_percentage", params_.trajectory_visualizer_samples_percentage, 0.05f);

  // dynamic reconfigure server
  dynamic_reconfigure_server_ = std::make_shared<dynamic_reconfigure::Server<MPPIControllerCUDAConfig>>(*private_node_);
  dynamic_reconfigure_callback_ = boost::bind(&ParametersHandler::dynamicParamsCallback, this, _1, _2);
  dynamic_reconfigure_server_->setCallback(dynamic_reconfigure_callback_);
}

ParametersHandler::~ParametersHandler()
{}

void ParametersHandler::dynamicParamsCallback(const MPPIControllerCUDAConfig& cfg, uint32_t level)
{
  std::lock_guard<std::mutex> lock(mutex_);

  // Controller parameters
  params_.vx_max = cfg.vx_max;
  params_.vy_max = cfg.vy_max;
  params_.vx_min = cfg.vx_min;

  // Acceleration/rate-of-change limits
  params_.ax_max = cfg.ax_max;
  params_.ax_min = cfg.ax_min;
  params_.ay_max = cfg.ay_max;
  params_.az_max = cfg.az_max;

  // Sampler noise std_dev
  params_.vx_std = cfg.vx_std;
  params_.vy_std = cfg.vy_std;
  params_.wz_std = cfg.wz_std;
  params_.optimizer_gamma = cfg.optimizer_gamma;
  params_.pure_noise_trajectories_percentage = cfg.pure_noise_trajectories_percentage;
  params_.vx_noise_color_exponent = cfg.vx_noise_color_exponent;
  params_.vy_noise_color_exponent = cfg.vy_noise_color_exponent;
  params_.wz_noise_color_exponent = cfg.wz_noise_color_exponent;

  // Constrain critic parameters
  params_.constrain_cost_power = cfg.constrain_cost_power;
  params_.constrain_cost_weight = cfg.constrain_cost_weight;

  // Goal angle critic parameters
  params_.goal_angle_cost_power = cfg.goal_angle_cost_power;
  params_.goal_angle_cost_weight = cfg.goal_angle_cost_weight;

  // Goal critic parameters
  params_.goal_cost_power = cfg.goal_cost_power;
  params_.goal_cost_weight = cfg.goal_cost_weight;
  params_.goal_threshold_to_consider = cfg.goal_threshold_to_consider;
  params_.path_align_threshold_to_consider = cfg.path_align_threshold_to_consider;
  params_.path_angle_threshold_to_consider = cfg.path_angle_threshold_to_consider;
  params_.prefer_forward_threshold_to_consider = cfg.prefer_forward_threshold_to_consider;

  // Obstacles critic parameters
  params_.obstacles_consider_footprint = cfg.obstacles_consider_footprint;
  params_.obstacles_cost_power = cfg.obstacles_cost_power;
  params_.obstacles_repulsion_weight = cfg.obstacles_repulsion_weight;
  params_.obstacles_critical_weight = cfg.obstacles_critical_weight;
  params_.obstacles_collision_cost = cfg.obstacles_collision_cost;
  params_.obstacles_collision_margin_distance = cfg.obstacles_collision_margin_distance;
  params_.obstacles_near_goal_distance = cfg.obstacles_near_goal_distance;

  // Cost critic parameters
  params_.cost_power = cfg.cost_power;
  params_.cost_weight = cfg.cost_weight;
  params_.cost_critical_cost = cfg.cost_critical_cost;
  params_.cost_collision_cost = cfg.cost_collision_cost;

  // Path align critic parameters
  params_.path_align_offset_from_furthest = cfg.path_align_offset_from_furthest;
  params_.path_align_max_path_occupancy_ratio = cfg.path_align_max_path_occupancy_ratio;
  params_.path_align_cost_power = cfg.path_align_cost_power;
  params_.path_align_cost_weight = cfg.path_align_cost_weight;

  // Path angle critic parameters
  params_.path_angle_offset_from_furthest = cfg.path_angle_offset_from_furthest;
  params_.path_angle_cost_power = cfg.path_angle_cost_power;
  params_.path_angle_cost_weight = cfg.path_angle_cost_weight;
  params_.path_angle_max_angle_to_furthest = cfg.path_angle_max_angle_to_furthest;

  // Path follow critic parameters
  params_.path_follow_offset_from_furthest = cfg.path_follow_offset_from_furthest;
  params_.path_follow_cost_power = cfg.path_follow_cost_power;
  params_.path_follow_cost_weight = cfg.path_follow_cost_weight;

  // Prefer forward critic parameters
  params_.prefer_forward_cost_power = cfg.prefer_forward_cost_power;
  params_.prefer_forward_cost_weight = cfg.prefer_forward_cost_weight;

  // Twirling critic parameters
  params_.twirling_cost_power = cfg.twirling_cost_power;
  params_.twirling_cost_weight = cfg.twirling_cost_weight;
  params_.near_goal_twirling_cost_power = cfg.near_goal_twirling_cost_power;
  params_.near_goal_twirling_cost_weight = cfg.near_goal_twirling_cost_weight;

  // Velocity deadband critic parameters
  params_.velocity_deadband_cost_power = cfg.velocity_deadband_cost_power;
  params_.velocity_deadband_cost_weight = cfg.velocity_deadband_cost_weight;
  params_.velocity_deadband_velocities_1 = cfg.velocity_deadband_velocities_1;
  params_.velocity_deadband_velocities_2 = cfg.velocity_deadband_velocities_2;
  params_.velocity_deadband_velocities_3 = cfg.velocity_deadband_velocities_3;

  // Noise generator parameters
  params_.noise_generator_regenerate_noises = cfg.noise_generator_regenerate_noises;

  // Trajectory visualizer parameters
  params_.trajectory_visualizer_visualize_optimal = cfg.trajectory_visualizer_visualize_optimal;
  params_.trajectory_visualizer_visualize_samples = cfg.trajectory_visualizer_visualize_samples;
  params_.trajectory_visualizer_time_step = cfg.trajectory_visualizer_time_step;
}

}  // namespace mppi
