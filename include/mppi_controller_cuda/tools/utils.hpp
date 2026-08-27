#ifndef MPPI_CONTROLLER_CUDA__TOOLS__UTILS_HPP_
#define MPPI_CONTROLLER_CUDA__TOOLS__UTILS_HPP_

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <string>
#include <limits>
#include <memory>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <rclcpp/rclcpp.hpp>

#define M_PIF 3.141592653589793238462643383279502884e+00F
#define M_PIF_2 1.5707963267948966e+00F

namespace mppi::utils
{
/**
 * @brief Convert data into pose
 * @param x X position
 * @param y Y position
 * @param z Z position
 * @return Pose object
 */
inline geometry_msgs::msg::Pose createPose(double x, double y, double z)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation.w = 1;
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  return pose;
}

/**
 * @brief Convert data into scale
 * @param x X scale
 * @param y Y scale
 * @param z Z scale
 * @return Scale object
 */
inline geometry_msgs::msg::Vector3 createScale(double x, double y, double z)
{
  geometry_msgs::msg::Vector3 scale;
  scale.x = x;
  scale.y = y;
  scale.z = z;
  return scale;
}

/**
 * @brief Convert data into color
 * @param r Red component
 * @param g Green component
 * @param b Blue component
 * @param a Alpha component (transparency)
 * @return Color object
 */
inline std_msgs::msg::ColorRGBA createColor(float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

/**
 * @brief Convert data into a Marker
 * @param id Marker ID
 * @param pose Marker pose
 * @param scale Marker scale
 * @param color Marker color
 * @param frame Reference frame to use
 * @return Visualization Marker
 */
inline visualization_msgs::msg::Marker createMarker(
  int id, const geometry_msgs::msg::Pose & pose, const geometry_msgs::msg::Vector3 & scale,
  const std_msgs::msg::ColorRGBA & color, const std::string & frame_id, const std::string & ns)
{
  using visualization_msgs::msg::Marker;
  Marker marker;
  marker.header.frame_id = frame_id;
  marker.header.stamp = rclcpp::Time(0, 0);
  marker.ns = ns;
  marker.id = id;
  marker.type = Marker::SPHERE;
  marker.action = Marker::ADD;

  marker.pose = pose;
  marker.scale = scale;
  marker.color = color;
  return marker;
}

/**
  * @brief normalize
  * Normalizes the angle to be -M_PIF circle to +M_PIF circle
  * It takes and returns radians.
  * @param angles Angles to normalize
  * @return normalized angles
  */
template<typename T>
auto normalize_angles(const T & angles)
{
  return (angles + M_PIF).unaryExpr([&](const float x) {
             float remainder = std::fmod(x, 2.0f * M_PIF);
             return remainder < 0.0f ? remainder + M_PIF : remainder - M_PIF;
             });
}

/**
  * @brief shortest_angular_distance
  *
  * Given 2 angles, this returns the shortest angular
  * difference.  The inputs and outputs are of course radians.
  *
  * The result
  * would always be -pi <= result <= pi.  Adding the result
  * to "from" will always get you an equivalent angle to "to".
  * @param from Start angle
  * @param to End angle
  * @return Shortest distance between angles
  */
template<typename F, typename T>
auto shortest_angular_distance(
  const F & from,
  const T & to)
{
  return normalize_angles(to - from);
}

/**
 * @brief Find the iterator of the first pose at which there is an inversion on the path,
 * @param path to check for inversion
 * @return the first point after the inversion found in the path
 */
inline unsigned int findFirstPathInversion(nav_msgs::msg::Path & path)
{
  // At least 3 poses for a possible inversion
  if (path.poses.size() < 3) {
    return path.poses.size();
  }

  // Iterating through the path to determine the position of the path inversion
  for (unsigned int idx = 1; idx < path.poses.size() - 1; ++idx) {
    // We have two vectors for the dot product OA and AB. Determining the vectors.
    float oa_x = path.poses[idx].pose.position.x -
      path.poses[idx - 1].pose.position.x;
    float oa_y = path.poses[idx].pose.position.y -
      path.poses[idx - 1].pose.position.y;
    float ab_x = path.poses[idx + 1].pose.position.x -
      path.poses[idx].pose.position.x;
    float ab_y = path.poses[idx + 1].pose.position.y -
      path.poses[idx].pose.position.y;

    // Checking for the existence of cusp, in the path, using the dot product.
    float dot_product = (oa_x * ab_x) + (oa_y * ab_y);
    if (dot_product < 0.0f) {
      return idx + 1;
    }
  }

  return path.poses.size();
}

/**
 * @brief Find and remove poses after the first inversion in the path
 * @param path to check for inversion
 * @return The location of the inversion, return 0 if none exist
 */
inline unsigned int removePosesAfterFirstInversion(nav_msgs::msg::Path & path)
{
  nav_msgs::msg::Path cropped_path = path;
  const unsigned int first_after_inversion = findFirstPathInversion(cropped_path);
  if (first_after_inversion == path.poses.size()) {
    return 0u;
  }

  cropped_path.poses.erase(
    cropped_path.poses.begin() + first_after_inversion, cropped_path.poses.end());
  path = cropped_path;
  return first_after_inversion;
}

/**
 * @brief Compare to trajectory points to find closest path point along integrated distances
 * @param vec Vect to check
 * @return dist Distance to look for
 * @return init Starting index to indec from
 */
inline unsigned int findClosestPathPt(
  const std::vector<float> & vec, const float dist, const unsigned int init = 0u)
{
  float distim1 = init != 0u ? vec[init] : 0.0f;  // First is 0, no accumulated distance yet
  float disti = 0.0f;
  const unsigned int size = vec.size();
  for (unsigned int i = init + 1; i != size; i++) {
    disti = vec[i];
    if (disti > dist) {
      if (i > 0 && dist - distim1 < disti - dist) {
        return i - 1;
      }
      return i;
    }
    distim1 = disti;
  }
  return size - 1;
}

/**
 * @brief Clamps the input between the given lower and upper bounds.
 * @param lower_bound Lower bound.
 * @param upper_bound Upper bound.
 * @return Clamped output.
 */
inline float clamp(
  const float lower_bound, const float upper_bound, const float input)
{
  return std::min(upper_bound, std::max(input, lower_bound));
}

}  // namespace mppi::utils

#endif  // MPPI_CONTROLLER_CUDA__TOOLS__UTILS_HPP_
