#ifndef MPPI_CONTROLLER_CUDA__TOOLS__TRAJECTORY_VISUALIZER_HPP_
#define MPPI_CONTROLLER_CUDA__TOOLS__TRAJECTORY_VISUALIZER_HPP_

#include <Eigen/Dense>

#include <memory>
#include <string>

#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <visualization_msgs/MarkerArray.h>

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
   * @param node Node handle
   * @param private_node Private node handle
   * @param frame_id Frame to publish trajectories in
   * @param parameters Parameter handler object
   */
  void onConfigure(
    std::shared_ptr<ros::NodeHandle> node, std::shared_ptr<ros::NodeHandle> private_node,
    const std::string & frame_id, Parameters * parameters);

  /**
   * @brief Add the optimal trajectory to visualize
   * @param trajectory Optimal trajectory, rows=timesteps, cols={x, y, yaw}
   * @param marker_namespace Marker namespace
   * @param cmd_stamp Timestamp for the published path
   */
  void add(
    const Eigen::ArrayXXf & trajectory, const std::string & marker_namespace,
    const ros::Time & cmd_stamp);

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
  void visualize(const nav_msgs::Path & plan);

  /**
   * @brief Reset internal marker/path buffers
   */
  void reset();

protected:
  std::string frame_id_;
  ros::Publisher trajectories_publisher_;
  ros::Publisher transformed_path_pub_;
  ros::Publisher optimal_path_pub_;

  nav_msgs::Path optimal_path_;
  visualization_msgs::MarkerArray points_;
  int marker_id_ = 0;

  Parameters * parameters_;

  size_t time_step_{1};
};

}  // namespace mppi

#endif  // MPPI_CONTROLLER_CUDA__TOOLS__TRAJECTORY_VISUALIZER_HPP_
