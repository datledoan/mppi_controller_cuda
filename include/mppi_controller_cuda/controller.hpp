#ifndef MPPI_CONTROLLER_CUDA__CONTROLLER_HPP_
#define MPPI_CONTROLLER_CUDA__CONTROLLER_HPP_

#include <string>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <tf2_ros/buffer.h>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_core/controller.hpp>
#include <nav2_core/goal_checker.hpp>

#include "mppi_controller_cuda/tools/path_handler.hpp"
#include "mppi_controller_cuda/tools/trajectory_visualizer.hpp"
#include "mppi_controller_cuda/tools/parameters_handler.hpp"
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
class MPPIControllerCUDA : public nav2_core::Controller
{
public:
  static constexpr int NUM_TIMESTEPS = 100;
  static constexpr int NUM_ROLLOUTS = 2048;
  using DYN_T = MotionModel;
  using COST_T = MotionModelCost;
  using SAMPLING_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>;
  using CONTROLLER_T = ::Controller<DYN_T, COST_T, NUM_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

  /**
    * @brief Constructor for mppi::MPPIControllerCUDA
    */
  MPPIControllerCUDA() = default;
  /**
    * @brief Destructor for mppi::MPPIControllerCUDA
    */
  ~MPPIControllerCUDA() override;

  /**
    * @brief Configure controller on bringup
    * @param parent WeakPtr to node
    * @param name Name of plugin
    * @param tf TF buffer to use
    * @param costmap_ros Costmap2DROS object of environment
    */
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, const std::shared_ptr<tf2_ros::Buffer> tf,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  /**
    * @brief Cleanup resources
    */
  void cleanup() override;

  /**
    * @brief Activate controller
    */
  void activate() override;

  /**
    * @brief Deactivate controller
    */
  void deactivate() override;

  /**
   * @brief nav2_core::Controller api: compute the best command given the
   * current pose and velocity
   * @param pose      Current robot pose
   * @param velocity  Current robot velocity
   * @param goal_checker Pointer to the goal checker (unused -- goal-reached
   * detection is owned by the controller_server's own GoalChecker plugin)
   * @return          Best command for the robot to drive
   */
  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

  /**
   * @brief Sets the global plan
   */
  void setPlan(const nav_msgs::msg::Path & path) override;

  /**
    * @brief Set new speed limit from callback
    * @param speed_limit Speed limit to use
    * @param percentage Bool if the speed limit is absolute or relative
    */
  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

protected:
  /**
    * @brief Visualize trajectories
    * @param transformed_plan Transformed input plan
    * @param cmd_stamp Command stamp
    */
  void visualize(
    nav_msgs::msg::Path transformed_plan,
    const builtin_interfaces::msg::Time & cmd_stamp);

  /**
   * @brief Push the current costmap (obstacle) data into cost_gpu_'s GPU texture.
   * Caller must hold costmap->getMutex() for the duration of this call.
   */
  void updateObstacleMap(nav2_costmap_2d::Costmap2D * costmap);

  /**
   * @brief Copy costmap_ros_'s current robot footprint (robot-local frame,
   * can change at runtime) into cost_params for computeObstacleCost, when
   * Parameters::obstacles_consider_footprint is enabled.
   */
  void updateFootprint(MotionModelCostParams & cost_params);

  /**
   * @brief Re-read every cost-critic weight/power (+ vx_max/vx_min/vy_max)
   * from params_ into cost_params. Called both once in configure() (to seed
   * sane values before the first rollout / kernel benchmark) and every cycle
   * in computeVelocityCommands(), so dynamic parameter changes to these take
   * effect immediately instead of only at the next node restart.
   */
  void refreshCostParams(MotionModelCostParams & cost_params);

  /**
   * @brief Re-read vx_std/vy_std/wz_std from params_ into sampler_params.
   * Called both once in configure() and every cycle in
   * computeVelocityCommands(), same reactivity pattern as refreshCostParams().
   */
  void refreshSamplerParams(SAMPLING_T::SAMPLING_PARAMS_T & sampler_params, bool near_goal = false);

  /**
   * @brief Re-read ax_max/ax_min/ay_max/az_max from params_ into dyn_params.
   * Called every cycle in computeVelocityCommands(), same reactivity pattern
   * as refreshCostParams()/refreshSamplerParams() -- current_vx/vy/wz are set
   * by the caller separately from the robot's actual measured velocity, not
   * by this method.
   */
  void refreshDynamicsParams(MotionModelParams & dyn_params);

  /**
   * @brief vx_max after applying the current speed limit (see setSpeedLimit()).
   * Applied at use-time rather than written back into params_->vx_max so it
   * never fights with dynamic parameter reconfiguration of vx_max itself.
   */
  float effectiveVxMax() const;

  const bool & isInitialized()
  {
    return initialized_;
  }

  int last_true_furthest_reached_idx_ = 0;
  bool initialized_ = false;
  bool noise_generated_once_ = false;

  std::string name_;
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  rclcpp::Logger logger_{rclcpp::get_logger("MPPIControllerCUDA")};
  rclcpp::Clock::SharedPtr clock_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  nav2_costmap_2d::Costmap2D * costmap_;
  double speed_limit_{0.0};
  bool speed_limit_is_percentage_{false};
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
