#ifndef MPPI_CONTROLLER_CUDA__TOOLS__TRAJECTORY_VISUALIZER_HPP_
#define MPPI_CONTROLLER_CUDA__TOOLS__TRAJECTORY_VISUALIZER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "mppi_controller_cuda/tools/parameters_handler.hpp"

namespace mppi
{

/**
 * @class mppi::TrajectoryVisualizer
 * @brief Visualizes the optimal and sampled candidate trajectories for debugging
 */
class TrajectoryVisualizer
{
public:
  TrajectoryVisualizer() = default;

  /**
   * @brief Configure trajectory visualizer
   * @param parent WeakPtr to node
   * @param name Name of plugin
   * @param frame_id Frame to publish trajectories in
   * @param parameters Parameter handler object
   */
  void onConfigure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, const std::string & name,
    const std::string & frame_id, Parameters * parameters);

  /**
   * @brief Cleanup object on shutdown
   */
  void onCleanup();

  /**
   * @brief Activate publishers
   */
  void onActivate();

  /**
   * @brief Deactivate publishers
   */
  void onDeactivate();

  /**
   * @brief Add the optimal trajectory to visualize
   * @param trajectory Optimal trajectory, rows=timesteps, cols={x, y, yaw}
   * @param marker_namespace Marker namespace
   * @param cmd_stamp Timestamp for the published path
   */
  void add(
    const Eigen::ArrayXXf & trajectory, const std::string & marker_namespace,
    const builtin_interfaces::msg::Time & cmd_stamp);

  /**
   * @brief Add candidate sample trajectories to visualize
   * @param x Sample x positions, rows=samples, cols=timesteps
   * @param y Sample y positions, rows=samples, cols=timesteps
   * @param marker_namespace Marker namespace
   */
  void add(const Eigen::ArrayXXf & x, const Eigen::ArrayXXf & y, const std::string & marker_namespace);

  /**
   * @brief Publish everything added since the last call, plus the transformed plan
   * @param plan Transformed global plan to visualize
   */
  void visualize(const nav_msgs::msg::Path & plan);

  /**
   * @brief Reset internal marker/path buffers
   */
  void reset();

protected:
  std::string frame_id_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>>
  trajectories_publisher_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> transformed_path_pub_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> optimal_path_pub_;

  nav_msgs::msg::Path optimal_path_;
  visualization_msgs::msg::MarkerArray points_;
  int marker_id_ = 0;

  Parameters * parameters_;
  rclcpp::Logger logger_{rclcpp::get_logger("MPPIControllerCUDA")};

  size_t time_step_{1};
};

}  // namespace mppi

#endif  // MPPI_CONTROLLER_CUDA__TOOLS__TRAJECTORY_VISUALIZER_HPP_
