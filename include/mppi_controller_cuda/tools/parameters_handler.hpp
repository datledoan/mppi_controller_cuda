#ifndef MPPI_CONTROLLER_CUDA__TOOLS__PARAMETERS_HANDLER_HPP_
#define MPPI_CONTROLLER_CUDA__TOOLS__PARAMETERS_HANDLER_HPP_

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include "mppi_controller_cuda/MPPIControllerCUDAConfig.h"

namespace mppi
{

using namespace mppi_controller_cuda; 
struct Parameters
{
  // controller (goal-reached checks; read once at startup, not reactive to
  // dynamic_reconfigure -- see parameters_handler.cpp)
  float vx_max;
  float vy_max;
  float vx_min;
  float wz_max;
  float xy_goal_tolerance;
  float yaw_goal_tolerance;
  double trans_stopped_vel;
  double theta_stopped_vel;
  double controller_frequency;
  float vx_std;
  float vy_std;
  float wz_std;
  float optimizer_gamma;
  float pure_noise_trajectories_percentage;
  float vx_noise_color_exponent;
  float vy_noise_color_exponent;
  float wz_noise_color_exponent;
  float ackermann_constraints_min_turning_r;

  float ax_max;
  float ax_min;
  float ay_max;
  float az_max;

  //constrain critics
  int constrain_cost_power;
  float constrain_cost_weight;

  // goal angle critic
  int goal_angle_cost_power;
  float goal_angle_cost_weight;

  // goal critic
  int goal_cost_power;
  float goal_cost_weight;
  float goal_threshold_to_consider;
  float path_align_threshold_to_consider;
  float path_angle_threshold_to_consider;
  float prefer_forward_threshold_to_consider;

  // obstacles critic
  bool obstacles_consider_footprint;
  int obstacles_cost_power;
  float obstacles_repulsion_weight;
  float obstacles_critical_weight;
  float obstacles_collision_cost;
  float obstacles_collision_margin_distance;
  float obstacles_inflation_scale_factor;
  float obstacles_inflation_radius;
  float obstacles_near_goal_distance;

  // cost critic
  int cost_power;
  float cost_weight;
  float cost_critical_cost;
  float cost_collision_cost;

  // path align critics
  int path_align_offset_from_furthest;
  float path_align_max_path_occupancy_ratio;
  int path_align_cost_power;
  float path_align_cost_weight;

  // path angle critic
  int path_angle_offset_from_furthest;
  int path_angle_cost_power;
  float path_angle_cost_weight;
  float path_angle_max_angle_to_furthest;

  // path follow critic
  int path_follow_offset_from_furthest;
  int path_follow_cost_power;
  float path_follow_cost_weight;

  // prefer forward critic
  int prefer_forward_cost_power;
  float prefer_forward_cost_weight;

  // twirling critic
  int twirling_cost_power;
  float twirling_cost_weight;

  // near-goal twirling critic
  int near_goal_twirling_cost_power;
  float near_goal_twirling_cost_weight;

  // velocity deadband critic
  int velocity_deadband_cost_power;
  float velocity_deadband_cost_weight;
  double velocity_deadband_velocities_1;
  double velocity_deadband_velocities_2;
  double velocity_deadband_velocities_3;

  // path handler
  double path_handler_max_robot_pose_search_dist;
  double path_handler_prune_distance;
  double path_handler_transform_tolerance;
  bool path_handler_enforce_path_inversion;
  double path_handler_inversion_xy_tolerance;
  double path_handler_inversion_yaw_tolerance;

  // Noise generator
  bool noise_generator_regenerate_noises;

  // Optimizer
  float optimizer_model_dt;
  int optimizer_time_steps;
  int optimizer_iteration_count;
  int optimizer_retry_attempt_limit;
  float optimizer_temperature;
  std::string optimizer_motion_model;
  bool noise_generator_use_colored_noise;

  // Trajectory visualizer
  bool trajectory_visualizer_visualize_optimal;
  bool trajectory_visualizer_visualize_samples;
  int trajectory_visualizer_time_step;
  float trajectory_visualizer_samples_percentage;
};

/**
 * @class mppi::ParametersHandler
 * @brief Handles getting parameters
 */
class ParametersHandler
{
public:
  /**
   * @brief Constructor
   */
  ParametersHandler(std::shared_ptr<ros::NodeHandle> node, std::shared_ptr<ros::NodeHandle> private_node);

  /**
   * @brief Destructor
   */
  ~ParametersHandler();

  std::mutex * getLock() {return &mutex_;}

  Parameters * getParams() {return &params_;}

protected:
  std::shared_ptr<ros::NodeHandle> node_;
  std::shared_ptr<ros::NodeHandle> private_node_;
  Parameters params_;
  int map_height_;
  int map_width_;
  double map_resolution_;
  std::mutex mutex_;

  void dynamicParamsCallback(const MPPIControllerCUDAConfig& cfg, uint32_t level);
  std::shared_ptr<dynamic_reconfigure::Server<MPPIControllerCUDAConfig> > dynamic_reconfigure_server_;
  dynamic_reconfigure::Server<MPPIControllerCUDAConfig>::CallbackType dynamic_reconfigure_callback_;
};

}  // namespace mppi

#endif  // MPPI_CONTROLLER_CUDA__TOOLS__PARAMETERS_HANDLER_HPP_
