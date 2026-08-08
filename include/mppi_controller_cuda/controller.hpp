#ifndef MPPI_CONTROLLER_CUDA__CONTROLLER_HPP_
#define MPPI_CONTROLLER_CUDA__CONTROLLER_HPP_

#include <string>
#include <memory>

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <base_local_planner/odometry_helper_ros.h>
#include <base_local_planner/goal_functions.h>
#include <base_local_planner/costmap_model.h>

#include <mbf_costmap_core/costmap_controller.h>
#include <mbf_msgs/ExePathResult.h>

#include "mppi_controller_cuda/tools/path_handler.hpp"
#include "mppi_controller_cuda/tools/trajectory_visualizer.hpp"
#include "mppi_controller_cuda/controller/mppi_controller.cuh"
#include "mppi_controller_cuda/costs/motion_model_cost.cuh"
#include "mppi_controller_cuda/dynamics/motion_model.cuh"

namespace mppi_controller_cuda
{

using namespace mppi;  // NOLINT

/**
 * @class mppi::MPPIControllerCUDA
 * @brief Main plugin controller for MPPI Controller
 */
class MPPIControllerCUDA : public nav_core::BaseLocalPlanner, public mbf_costmap_core::CostmapController
{
public:
  static constexpr int NUM_TIMESTEPS = 100;
  static constexpr int NUM_ROLLOUTS = 2048;
  using DYN_T = MotionModel;
  using COST_T = MotionModelCost;
  using SAMPLING_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>;
  using CONTROLLER_T = Controller<DYN_T, COST_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

  /**
    * @brief Constructor for mppi::MPPIControllerCUDA
    */
  MPPIControllerCUDA() = default;
  /**
    * @brief Destructor for mppi::MPPIControllerCUDA
    */
  ~MPPIControllerCUDA();

  /**
    * @brief Configure controller on bringup
    * @param name Name of plugin
    * @param tf TF buffer to use
    * @param costmap_ros Costmap2DROS object of environment
    */
  void initialize(
    std::string name, tf2_ros::Buffer* tf,
    costmap_2d::Costmap2DROS* costmap_ros) override;

  /**
   * @brief move_base_flex api compute the best command given the current pose and velocity, with possible debug information
   * @param pose      Current robot pose
   * @param velocity  Current robot velocity
   * @param cmd_vel   Best command
   * @param message   Debug information
   * @return          move_base_flex result code
   */
  uint32_t computeVelocityCommands(const geometry_msgs::PoseStamped& pose,
                                  const geometry_msgs::TwistStamped& velocity,
                                  geometry_msgs::TwistStamped& cmd_vel,
                                  std::string& message) override;
  
  /**
   * @brief move_base api compute the best command given the current pose and velocity
   * @param pose      Current robot pose
   * @param velocity  Current robot velocity
   * @param cmd_vel   Best command
   * @return          true if a valid command was found, false otherwise
   */
  bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;

  bool cancel() { 
    return false; 
  };

  /**
   * @brief Sets the global plan
   */
  bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan);

protected:
  /**
    * @brief Visualize trajectories
    * @param transformed_plan Transformed input plan
    */
  void visualize(
    nav_msgs::Path transformed_plan,
    const ros::Time & cmd_stamp);

    /**
   * @brief move_base api whether the goal has been reached
   * @return Whether the goal has been reached
   */
  bool isGoalReached();

  /**
   * @brief move_base_flex api whether the goal has been reached
   * @return Whether the goal has been reached
   */
  bool isGoalReached(double xy_tolerance, double yaw_tolerance); 
  bool isThetaGoalReached(double dtheta, double angle_tolerance, double max_angular_vel, double dt);
  
    /** @brief Create a nav_msgs::Path message from a vector of PoseStamped messages
   * @param plan The input vector of PoseStamped messages
   * @param path The output nav_msgs::Path message
   */
  void createPathMsg(const std::vector<geometry_msgs::PoseStamped>& plan, nav_msgs::Path& path);

  /**
   * @brief Push the current costmap (obstacle) data into cost_gpu_'s GPU texture.
   * Caller must hold costmap->getMutex() for the duration of this call.
   */
  void updateObstacleMap(costmap_2d::Costmap2D* costmap);

  /**
   * @brief Copy costmap_ros_'s current robot footprint (robot-local frame,
   * can change at runtime) into cost_params for computeObstacleCost, when
   * Parameters::obstacles_consider_footprint is enabled.
   */
  void updateFootprint(MotionModelCostParams& cost_params);

  /**
   * @brief Re-read every cost-critic weight/power (+ vx_max/vx_min/vy_max)
   * from params_ into cost_params. Called both once in initialize() (to seed
   * sane values before the first rollout / kernel benchmark) and every cycle
   * in computeVelocityCommands(), so dynamic_reconfigure changes to these
   * take effect immediately instead of only at the next node restart.
   */
  void refreshCostParams(MotionModelCostParams& cost_params);

  /**
   * @brief Re-read vx_std/vy_std/wz_std from params_ into sampler_params.
   * Called both once in initialize() and every cycle in
   * computeVelocityCommands(), same reactivity pattern as refreshCostParams().
   */
  void refreshSamplerParams(SAMPLING_T::SAMPLING_PARAMS_T& sampler_params, bool near_goal = false);

  /**
   * @brief Re-read ax_max/ax_min/ay_max/az_max from params_ into dyn_params.
   * Called every cycle in computeVelocityCommands(), same reactivity pattern
   * as refreshCostParams()/refreshSamplerParams() -- current_vx/vy/wz are set
   * by the caller separately from the robot's actual measured velocity, not
   * by this method.
   */
  void refreshDynamicsParams(MotionModelParams& dyn_params);

  const bool& isInitialized()
  {
    return initialized_;
  }

  bool goal_reached_ = false;
  int last_true_furthest_reached_idx_ = 0;
  bool initialized_ = false;
  bool noise_generated_once_ = false;
  double xy_goal_tolerance_;
  double yaw_goal_tolerance_;
  double max_angular_vel_;
  double theta_stopped_vel_;
  double trans_stopped_vel_;
  double control_duration_;

  std::string name_;
  std::shared_ptr<ros::NodeHandle> node_;
  std::shared_ptr<ros::NodeHandle> private_node_;
  std::shared_ptr<costmap_2d::Costmap2DROS> costmap_ros_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  costmap_2d::Costmap2D * costmap_;
  base_local_planner::OdometryHelperRos odom_helper_;
  std::vector<geometry_msgs::PoseStamped> global_plan_;
  // TODO Fix hardcoded odom topic
  std::string odom_topic_{"odom"};
  std::unique_ptr<ParametersHandler> parameters_handler_;
  Parameters * params_;
  PathHandler path_handler_;
  TrajectoryVisualizer trajectory_visualizer_;

  DYN_T * dynamics_ = nullptr;
  COST_T * cost_gpu_ = nullptr;
  SAMPLING_T * sampler_ = nullptr;
  CONTROLLER_T * mppi_controller_ = nullptr;
};

}  // namespace mppi_controller_cuda

#endif  // MPPI_CONTROLLER_CUDA__CONTROLLER_HPP_
