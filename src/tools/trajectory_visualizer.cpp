#include <memory>
#include <algorithm>
#include "mppi_controller_cuda/tools/trajectory_visualizer.hpp"
#include "mppi_controller_cuda/tools/utils.hpp"

namespace mppi
{

void TrajectoryVisualizer::onConfigure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent, const std::string & /*name*/,
  const std::string & frame_id, Parameters * parameters)
{
  auto node = parent.lock();
  logger_ = node->get_logger();
  frame_id_ = frame_id;
  trajectories_publisher_ = node->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/trajectories", 1);
  transformed_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("transformed_global_plan", 1);
  optimal_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("optimal_trajectory", 1);
  parameters_ = parameters;

  time_step_ = static_cast<size_t>(std::max(parameters_->trajectory_visualizer_time_step, 1));

  reset();
}

void TrajectoryVisualizer::onCleanup()
{
  trajectories_publisher_.reset();
  transformed_path_pub_.reset();
  optimal_path_pub_.reset();
}

void TrajectoryVisualizer::onActivate()
{
  trajectories_publisher_->on_activate();
  transformed_path_pub_->on_activate();
  optimal_path_pub_->on_activate();
}

void TrajectoryVisualizer::onDeactivate()
{
  trajectories_publisher_->on_deactivate();
  transformed_path_pub_->on_deactivate();
  optimal_path_pub_->on_deactivate();
}

void TrajectoryVisualizer::add(
  const Eigen::ArrayXXf & trajectory, const std::string & marker_namespace,
  const builtin_interfaces::msg::Time & cmd_stamp)
{
  size_t size = trajectory.rows();
  if (!size)
  {
    return;
  }

  auto add_marker = [&](auto i) {
    float component = static_cast<float>(i) / static_cast<float>(size);

    auto pose = utils::createPose(trajectory(i, 0), trajectory(i, 1), 0.06);
    auto scale = i != size - 1 ? utils::createScale(0.03, 0.03, 0.07) : utils::createScale(0.07, 0.07, 0.09);
    auto color = utils::createColor(0, component, component, 1);
    auto marker = utils::createMarker(marker_id_++, pose, scale, color, frame_id_, marker_namespace);
    points_.markers.push_back(marker);

    // populate optimal path
    geometry_msgs::msg::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = frame_id_;
    pose_stamped.pose = pose;

    tf2::Quaternion quaternion_tf2;
    quaternion_tf2.setRPY(0., 0., trajectory(i, 2));
    pose_stamped.pose.orientation = tf2::toMsg(quaternion_tf2);

    optimal_path_.poses.push_back(pose_stamped);
  };

  optimal_path_.header.stamp = cmd_stamp;
  optimal_path_.header.frame_id = frame_id_;
  for (size_t i = 0; i < size; i++)
  {
    add_marker(i);
  }
}

void TrajectoryVisualizer::add(const Eigen::ArrayXXf & x, const Eigen::ArrayXXf & y, const std::string & marker_namespace)
{
  size_t n_rows = x.rows();
  size_t n_cols = x.cols();
  if (!n_rows || !n_cols)
  {
    return;
  }
  const float shape_1 = static_cast<float>(n_cols);
  points_.markers.reserve(points_.markers.size() + n_rows * (n_cols / time_step_ + 1));

  for (size_t i = 0; i < n_rows; i++)
  {
    for (size_t j = 0; j < n_cols; j += time_step_)
    {
      const float j_flt = static_cast<float>(j);
      float blue_component = 1.0f - j_flt / shape_1;
      float green_component = j_flt / shape_1;

      auto pose = utils::createPose(x(i, j), y(i, j), 0.03);
      auto scale = utils::createScale(0.03, 0.03, 0.03);
      auto color = utils::createColor(0, green_component, blue_component, 1);
      auto marker = utils::createMarker(marker_id_++, pose, scale, color, frame_id_, marker_namespace);

      points_.markers.push_back(marker);
    }
  }
}

void TrajectoryVisualizer::reset()
{
  marker_id_ = 0;
  points_ = visualization_msgs::msg::MarkerArray();
  optimal_path_ = nav_msgs::msg::Path();
}

void TrajectoryVisualizer::visualize(const nav_msgs::msg::Path & plan)
{
  if (trajectories_publisher_->get_subscription_count() > 0)
  {
    trajectories_publisher_->publish(points_);
  }

  if (optimal_path_pub_->get_subscription_count() > 0)
  {
    optimal_path_pub_->publish(optimal_path_);
  }

  reset();

  if (transformed_path_pub_->get_subscription_count() > 0)
  {
    transformed_path_pub_->publish(plan);
  }
}

}  // namespace mppi
