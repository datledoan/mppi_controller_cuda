#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <sys/resource.h>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav2_core/goal_checker.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "nav2_mppi_controller/optimizer.hpp"
#include "nav2_mppi_controller/tools/parameters_handler.hpp"

namespace
{

struct SweepPointResult
{
  double wall_mean_ms, wall_median_ms, wall_p95_ms, wall_max_ms, cpu_mean_ms;
};

unsigned char inflationCost(
  float dist_m, float inscribed_radius, float inflation_radius, float cost_scaling_factor)
{
  if (dist_m < 0.0f) {return 254;}
  if (dist_m <= inscribed_radius) {return 253;}
  if (dist_m > inflation_radius) {return 0;}
  return static_cast<unsigned char>(252.0f * expf(-cost_scaling_factor * (dist_m - inscribed_radius)));
}

std::shared_ptr<nav2_costmap_2d::Costmap2DROS> buildCostmapRos()
{
  auto costmap_ros = std::make_shared<nav2_costmap_2d::Costmap2DROS>("cpu_bench_costmap");
  costmap_ros->on_configure(rclcpp_lifecycle::State{});

  auto * costmap = costmap_ros->getCostmap();
  costmap->resizeMap(200, 200, 0.025, -2.5, -2.5);

  const float obst_wx = 1.0f, obst_wy = 0.15f;
  const float inscribed_radius = 0.105f, inflation_radius = 1.0f, cost_scaling_factor = 3.0f;
  for (unsigned int gy = 0; gy < costmap->getSizeInCellsY(); gy++) {
    for (unsigned int gx = 0; gx < costmap->getSizeInCellsX(); gx++) {
      double wx, wy;
      costmap->mapToWorld(gx, gy, wx, wy);
      float dist = std::sqrt((wx - obst_wx) * (wx - obst_wx) + (wy - obst_wy) * (wy - obst_wy)) - 0.15f;
      costmap->setCost(gx, gy, inflationCost(dist, inscribed_radius, inflation_radius, cost_scaling_factor));
    }
  }
  return costmap_ros;
}

SweepPointResult runSweepPoint(
  int batch_size, int time_steps, int warmup, int reps,
  const geometry_msgs::msg::PoseStamped & pose, const geometry_msgs::msg::Twist & speed,
  const nav_msgs::msg::Path & path)
{
  const std::string name = "mppi_cpu_sweep_bench";
  std::vector<rclcpp::Parameter> params;
  params.emplace_back(name + ".batch_size", batch_size);
  params.emplace_back(name + ".time_steps", time_steps);
  params.emplace_back(name + ".iteration_count", 1);
  params.emplace_back(name + ".motion_model", std::string("DiffDrive"));
  params.emplace_back(
    name + ".critics", std::vector<std::string>{
    "ConstraintCritic", "CostCritic", "GoalCritic", "GoalAngleCritic", "ObstaclesCritic",
    "PathAlignCritic", "PathFollowCritic", "PathAngleCritic", "PreferForwardCritic",
    "TwirlingCritic", "VelocityDeadbandCritic"});
  params.emplace_back("controller_frequency", 50.0);
  rclcpp::NodeOptions options;
  options.parameter_overrides(params);

  auto node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(name, options);
  auto costmap_ros = buildCostmapRos();

  mppi::ParametersHandler params_handler(node);
  mppi::Optimizer optimizer;
  std::weak_ptr<rclcpp_lifecycle::LifecycleNode> weak_node{node};
  optimizer.initialize(weak_node, node->get_name(), costmap_ros, &params_handler);

  nav2_core::GoalChecker * no_goal_checker{nullptr};

  for (int i = 0; i < warmup; i++) {
    optimizer.evalControl(pose, speed, path, no_goal_checker);
  }

  std::vector<double> wall_ms, cpu_ms;
  wall_ms.reserve(reps);
  cpu_ms.reserve(reps);
  for (int i = 0; i < reps; i++) {
    struct rusage ru_before, ru_after;
    getrusage(RUSAGE_SELF, &ru_before);
    auto t0 = std::chrono::steady_clock::now();
    optimizer.evalControl(pose, speed, path, no_goal_checker);
    auto t1 = std::chrono::steady_clock::now();
    getrusage(RUSAGE_SELF, &ru_after);
    wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    double cpu_delta = (ru_after.ru_utime.tv_sec - ru_before.ru_utime.tv_sec) * 1000.0 +
      (ru_after.ru_utime.tv_usec - ru_before.ru_utime.tv_usec) / 1000.0 +
      (ru_after.ru_stime.tv_sec - ru_before.ru_stime.tv_sec) * 1000.0 +
      (ru_after.ru_stime.tv_usec - ru_before.ru_stime.tv_usec) / 1000.0;
    cpu_ms.push_back(cpu_delta);
  }

  optimizer.shutdown();

  std::sort(wall_ms.begin(), wall_ms.end());
  SweepPointResult r;
  double wall_sum = 0.0;
  for (double v : wall_ms) {wall_sum += v;}
  r.wall_mean_ms = wall_sum / wall_ms.size();
  r.wall_median_ms = wall_ms[wall_ms.size() / 2];
  r.wall_p95_ms = wall_ms[static_cast<size_t>(wall_ms.size() * 0.95)];
  r.wall_max_ms = wall_ms.back();
  double cpu_sum = 0.0;
  for (double v : cpu_ms) {cpu_sum += v;}
  r.cpu_mean_ms = cpu_sum / cpu_ms.size();
  return r;
}

}  // namespace

int main(int argc, char ** argv)
{
  int warmup = argc > 1 ? std::atoi(argv[1]) : 5;
  int reps = argc > 2 ? std::atoi(argv[2]) : 30;
  if (warmup < 0 || reps <= 0) {
    fprintf(
      stderr, "Usage: %s [warmup] [reps]  (warmup >= 0, reps >= 1; defaults 5 30)\n", argv[0]);
    return 1;
  }

  rclcpp::init(0, nullptr);

  const int path_n = 50;
  const float path_step = 0.1f;
  const float heading_offset_deg = 20.0f;
  const float theta = heading_offset_deg * static_cast<float>(M_PI) / 180.0f;

  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "map";
  pose.pose.orientation.w = 1.0;

  geometry_msgs::msg::Twist speed;  // zero

  nav_msgs::msg::Path path;
  path.header.frame_id = "map";
  for (int i = 0; i < path_n; i++) {
    float s = path_step * i;
    geometry_msgs::msg::PoseStamped p;
    p.header = path.header;
    p.pose.position.x = s * std::cos(theta);
    p.pose.position.y = s * std::sin(theta);
    p.pose.orientation.w = 1.0;
    path.poses.push_back(p);
  }
  std::vector<int> batch_sizes = {256, 512, 1024, 2048, 4096, 8192};
  std::vector<int> timesteps_sweep = {32, 56, 100};  // matches sweep_bench.cu's grid

  printf("batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms\n");
  fflush(stdout);
  for (int batch_size : batch_sizes) {
    for (int time_steps : timesteps_sweep) {
      SweepPointResult r = runSweepPoint(batch_size, time_steps, warmup, reps, pose, speed, path);
      printf(
        "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", batch_size, time_steps, r.wall_mean_ms, r.wall_median_ms,
        r.wall_p95_ms, r.wall_max_ms, r.cpu_mean_ms);
      fflush(stdout);
    }
  }

  rclcpp::shutdown();
  return 0;
}
