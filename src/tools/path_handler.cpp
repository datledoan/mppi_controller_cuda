#include "mppi_controller_cuda/tools/path_handler.hpp"
#include "mppi_controller_cuda/tools/utils.hpp"
#include <angles/angles.h>
#include <costmap_2d/costmap_2d_ros.h>

namespace mppi
{

  void PathHandler::initialize(
    std::shared_ptr<ros::NodeHandle> node, std::shared_ptr<ros::NodeHandle> private_node, const std::string & name,
    std::shared_ptr<costmap_2d::Costmap2DROS> costmap,
    std::shared_ptr<tf2_ros::Buffer> buffer, Parameters * parameters)
{
  name_ = name;
  costmap_ = costmap;
  tf_buffer_ = buffer;
  parameters_ = parameters;
  max_robot_pose_search_dist_ = parameters_->path_handler_max_robot_pose_search_dist;
  prune_distance_ = parameters_->path_handler_prune_distance;
  transform_tolerance_ = parameters_->path_handler_transform_tolerance;
  enforce_path_inversion_ = parameters_->path_handler_enforce_path_inversion;
  if (enforce_path_inversion_) {
    inversion_xy_tolerance_ = parameters_->path_handler_inversion_xy_tolerance;
    inversion_yaw_tolerance_ = parameters_->path_handler_inversion_yaw_tolerance;
    inversion_locale_ = 0u;
  }
}

std::pair<nav_msgs::Path, PathIterator> PathHandler::getGlobalPlanConsideringBoundsInCostmapFrame(const geometry_msgs::PoseStamped & global_pose)
{
  using mppi_controller_cuda::geometry_utils::euclidean_distance;
  auto begin = global_plan_up_to_inversion_.poses.begin();

  // Limit the search for the closest pose up to max_robot_pose_search_dist on the path
  auto closest_pose_upper_bound =
    mppi_controller_cuda::geometry_utils::first_after_integrated_distance(
    global_plan_up_to_inversion_.poses.begin(), global_plan_up_to_inversion_.poses.end(),
    max_robot_pose_search_dist_);

  // Find closest point to the robot
  auto closest_point = mppi_controller_cuda::geometry_utils::min_by(
    begin, closest_pose_upper_bound,
    [&global_pose](const geometry_msgs::PoseStamped & ps) {
      return euclidean_distance(global_pose, ps);
    });

  nav_msgs::Path transformed_plan;
  transformed_plan.header.frame_id = costmap_->getGlobalFrameID();
  transformed_plan.header.stamp = global_pose.header.stamp;

  auto pruned_plan_end =
    mppi_controller_cuda::geometry_utils::first_after_integrated_distance(
    closest_point, global_plan_up_to_inversion_.poses.end(), prune_distance_);

  unsigned int mx, my;
  // Find the furthest relevant pose on the path to consider within costmap
  // bounds
  // Transforming it to the costmap frame in the same loop
  for (auto global_plan_pose = closest_point; global_plan_pose != pruned_plan_end;
    ++global_plan_pose)
  {
    // Transform from global plan frame to costmap frame
    geometry_msgs::PoseStamped costmap_plan_pose;
    global_plan_pose->header.stamp = global_pose.header.stamp;
    global_plan_pose->header.frame_id = global_plan_.header.frame_id;
    transformPose(costmap_->getGlobalFrameID(), *global_plan_pose, costmap_plan_pose);

    // Check if pose is inside the costmap
    if (!costmap_->getCostmap()->worldToMap(
        costmap_plan_pose.pose.position.x, costmap_plan_pose.pose.position.y, mx, my))
    {
      return {transformed_plan, closest_point};
    }

    // Filling the transformed plan to return with the transformed pose
    transformed_plan.poses.push_back(costmap_plan_pose);
  }

  return {transformed_plan, closest_point};
}

geometry_msgs::PoseStamped PathHandler::transformToGlobalPlanFrame(const geometry_msgs::PoseStamped & pose)
{
  if (global_plan_up_to_inversion_.poses.empty()) {
    throw std::runtime_error("Received plan with zero length");
  }

  geometry_msgs::PoseStamped robot_pose;
  if (!transformPose(global_plan_up_to_inversion_.header.frame_id, pose, robot_pose)) {
    throw std::runtime_error(
            "Unable to transform robot pose into global plan's frame");
  }

  return robot_pose;
}

nav_msgs::Path PathHandler::transformPath(
  const geometry_msgs::PoseStamped & robot_pose)
{
  // Find relevant bounds of path to use
  geometry_msgs::PoseStamped global_pose =
    transformToGlobalPlanFrame(robot_pose);
  auto [transformed_plan, lower_bound] = getGlobalPlanConsideringBoundsInCostmapFrame(global_pose);

  prunePlan(global_plan_up_to_inversion_, lower_bound);

  if (enforce_path_inversion_ && inversion_locale_ != 0u) {
    if (isWithinInversionTolerances(global_pose)) {
      prunePlan(global_plan_, global_plan_.poses.begin() + inversion_locale_);
      global_plan_up_to_inversion_ = global_plan_;
      inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
    }
  }

  if (transformed_plan.poses.empty()) {
    throw std::runtime_error("Resulting plan has 0 poses in it.");
  }

  return transformed_plan;
}

bool PathHandler::transformPose(
  const std::string & frame, const geometry_msgs::PoseStamped & in_pose,
  geometry_msgs::PoseStamped & out_pose) const
{
  if (in_pose.header.frame_id == frame) {
    out_pose = in_pose;
    return true;
  }

  try {
    tf_buffer_->transform(
      in_pose, out_pose, frame,
      ros::Duration(transform_tolerance_));
    out_pose.header.frame_id = frame;
    return true;
  } catch (tf2::TransformException & ex) {
    ROS_ERROR("Exception in transformPose: %s", ex.what());
  }
  return false;
}

void PathHandler::setPath(const nav_msgs::Path & plan)
{
  global_plan_ = plan;
  global_plan_up_to_inversion_ = global_plan_;
  if (enforce_path_inversion_) {
    inversion_locale_ = utils::removePosesAfterFirstInversion(global_plan_up_to_inversion_);
  }
}

void PathHandler::prunePlan(nav_msgs::Path & plan, const PathIterator end)
{
  plan.poses.erase(plan.poses.begin(), end);
}

geometry_msgs::PoseStamped PathHandler::getTransformedGoal(const ros::Time & stamp)
{
  auto goal = global_plan_.poses.back();
  goal.header.frame_id = global_plan_.header.frame_id;
  goal.header.stamp = stamp;
  if (goal.header.frame_id.empty()) {
    throw std::runtime_error("Goal pose has an empty frame_id");
  }
  geometry_msgs::PoseStamped transformed_goal;
  if (!transformPose(costmap_->getGlobalFrameID(), goal, transformed_goal)) {
    throw std::runtime_error("Unable to transform goal pose into costmap frame");
  }
  return transformed_goal;
}

bool PathHandler::isWithinInversionTolerances(const geometry_msgs::PoseStamped & robot_pose)
{
  // Keep full path if we are within tolerance of the inversion pose
  const auto last_pose = global_plan_up_to_inversion_.poses.back();
  float distance = hypotf(
    robot_pose.pose.position.x - last_pose.pose.position.x,
    robot_pose.pose.position.y - last_pose.pose.position.y);

  float angle_distance = angles::shortest_angular_distance(
    tf2::getYaw(robot_pose.pose.orientation),
    tf2::getYaw(last_pose.pose.orientation));

  return distance <= inversion_xy_tolerance_ && fabs(angle_distance) <= inversion_yaw_tolerance_;
}

}  // namespace mppi
