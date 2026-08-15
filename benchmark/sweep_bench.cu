#include "mppi_controller_cuda/controller/mppi_controller.cuh"
#include "mppi_controller_cuda/costs/motion_model_cost.cuh"
#include "mppi_controller_cuda/dynamics/motion_model.cuh"
#include "mppi_controller_cuda/sampling_distribution/colored_noise.cuh"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <sys/resource.h>
#include <string>

using namespace mppi;
using DYN_T = MotionModel;
using COST_T = MotionModelCost;
using SAMPLING_T = mppi::sampling_distributions::ColoredNoiseDistribution<DYN_T::DYN_PARAMS_T>;
static constexpr int MAX_TIMESTEPS = 100;
#ifndef MPPI_NUM_ROLLOUTS
#define MPPI_NUM_ROLLOUTS 2048
#endif
static constexpr int NUM_ROLLOUTS = MPPI_NUM_ROLLOUTS;
using CONTROLLER_T = Controller<DYN_T, COST_T, MAX_TIMESTEPS, NUM_ROLLOUTS, SAMPLING_T>;

static constexpr float VX_MAX = 0.5f, VX_MIN = -0.35f, WZ_MAX = 1.9f;
static constexpr float VX_STD = 0.2f, WZ_STD = 0.4f;
static constexpr float OPT_GAMMA = 0.015f;
static constexpr float AX_MAX = 3.0f, AX_MIN = -3.0f, AY_MAX = 3.0f, AZ_MAX = 3.5f;
static constexpr float GOAL_W = 5.0f, GOAL_ANGLE_W = 3.0f;
static constexpr float PATH_ALIGN_W = 10.0f, PATH_ANGLE_W = 2.2f, PATH_FOLLOW_W = 5.0f;
static constexpr float NEAR_GOAL_TWIRL_W = 0.5f;
static constexpr float CONSTRAIN_W = 4.0f;
static constexpr float OBSTACLE_REPULSION_W = 1.5f;
static constexpr float MODEL_DT = 0.05f;
static constexpr float GOAL_THRESH = 1.4f, PATH_ALIGN_THRESH = 0.5f, PATH_ANGLE_THRESH = 0.5f,
                        PREFER_FORWARD_THRESH = 0.5f;
static constexpr int PATH_ALIGN_OFFSET = 20, PATH_ANGLE_OFFSET = 4, PATH_FOLLOW_OFFSET = 6;
static constexpr float PATH_ANGLE_MAX_ANGLE = 0.785398f;
static constexpr float PREFER_FORWARD_W = 0.3f, TWIRL_W = 0.0f;
static constexpr float LAMBDA = 0.3f;

static unsigned char inflationCost(float dist_m, float inscribed_radius, float inflation_radius,
                                    float cost_scaling_factor)
{
  if (dist_m < 0.0f)
    return 254;
  if (dist_m <= inscribed_radius)
    return 253;
  if (dist_m > inflation_radius)
    return 0;
  return static_cast<unsigned char>(252.0f * expf(-cost_scaling_factor * (dist_m - inscribed_radius)));
}

int main(int argc, char** argv)
{
  int warmup = argc > 1 ? std::atoi(argv[1]) : 5;
  int reps = argc > 2 ? std::atoi(argv[2]) : 30;
  if (warmup < 0 || reps <= 0)
  {
    fprintf(stderr, "Usage: %s [warmup] [reps]  (warmup >= 0, reps >= 1; defaults 5 30)\n", argv[0]);
    fprintf(stderr, "Got warmup=%d reps=%d from argv -- these must be plain numbers, not the literal\n",
            warmup, reps);
    fprintf(stderr, "\"[warmup]\"/\"[reps]\" brackets shown in the README (those just mean \"optional\").\n");
    return 1;
  }

  std::vector<int> timesteps_sweep = { 32, 56, 100 };
  const int path_n = 50;
  const float path_step = 0.1f;
  const float heading_offset_deg = 20.0f;
  std::vector<float> path_x(path_n), path_y(path_n);
  {
    float theta = heading_offset_deg * static_cast<float>(M_PI) / 180.0f;
    for (int i = 0; i < path_n; i++)
    {
      float s = path_step * i;
      path_x[i] = s * std::cos(theta);
      path_y[i] = s * std::sin(theta);
    }
  }
  const float goal_x = path_x.back(), goal_y = path_y.back(), goal_yaw = 0.0f;

  DYN_T* dynamics = new DYN_T();
  {
    MotionModelParams p = dynamics->getParams();
    p.min_turning_radius = 0.0f;
    p.is_holonomic = false;
    p.ax_max = AX_MAX;
    p.ax_min = AX_MIN;
    p.ay_max = AY_MAX;
    p.az_max = AZ_MAX;
    dynamics->setParams(p);
  }

  COST_T* cost_gpu = new COST_T();
  const int costmap_w = 200, costmap_h = 200;
  const float costmap_res = 0.025f;
  const float costmap_origin_x = -2.5f, costmap_origin_y = -2.5f;
  {
    std::vector<float> obstacle_texture(costmap_w * costmap_h, 0.0f);
    const float obst_wx = 1.0f, obst_wy = 0.15f;
    const float inscribed_radius = 0.105f, inflation_radius = 1.0f, cost_scaling_factor = 3.0f;
    for (int gy = 0; gy < costmap_h; gy++)
    {
      for (int gx = 0; gx < costmap_w; gx++)
      {
        float wx = costmap_origin_x + (gx + 0.5f) * costmap_res;
        float wy = costmap_origin_y + (gy + 0.5f) * costmap_res;
        float dist = std::sqrt((wx - obst_wx) * (wx - obst_wx) + (wy - obst_wy) * (wy - obst_wy)) -
                     0.15f;  // shrink by a small obstacle radius
        obstacle_texture[gy * costmap_w + gx] =
            inflationCost(dist, inscribed_radius, inflation_radius, cost_scaling_factor);
      }
    }
    cost_gpu->updateObstacleMap(obstacle_texture, costmap_w, costmap_h, costmap_origin_x, costmap_origin_y,
                                 costmap_res);

    auto cp = cost_gpu->getParams();
    cp.goal_weight = GOAL_W;
    cp.goal_power = 1;
    cp.goal_angle_weight = GOAL_ANGLE_W;
    cp.goal_angle_power = 1;
    cp.prefer_forward_weight = PREFER_FORWARD_W;
    cp.prefer_forward_power = 1;
    cp.twirling_weight = TWIRL_W;
    cp.twirling_power = 1;
    cp.near_goal_twirling_weight = NEAR_GOAL_TWIRL_W;
    cp.near_goal_twirling_power = 1;
    cp.velocity_deadband_weight = 35.0f;
    cp.constraint_weight = CONSTRAIN_W;
    cp.constraint_power = 1;
    cp.vx_max = VX_MAX;
    cp.wz_max = WZ_MAX;
    cp.vx_min = VX_MIN;
    cp.vy_max = 0.0f;
    cp.obstacle_repulsion_weight = OBSTACLE_REPULSION_W;
    cp.obstacle_scale_factor = 3.0f;
    cp.obstacle_inflation_radius = 1.0f;
    cp.obstacle_collision_margin = 0.1f;
    cp.cost_weight = 0.0f;
    cp.path_align_offset_from_furthest = PATH_ALIGN_OFFSET;
    cp.path_align_weight = PATH_ALIGN_W;
    cp.path_align_power = 1;
    cp.path_angle_weight = PATH_ANGLE_W;
    cp.path_angle_power = 1;
    cp.path_follow_weight = PATH_FOLLOW_W;
    cp.path_follow_power = 1;
    cp.is_holonomic = false;
    cp.path_length = path_n;
    float cumulative = 0.0f;
    for (int i = 0; i < path_n; i++)
    {
      cp.path_x[i] = path_x[i];
      cp.path_y[i] = path_y[i];
      cp.path_valid[i] = true;
      if (i > 0)
      {
        float ddx = path_x[i] - path_x[i - 1];
        float ddy = path_y[i] - path_y[i - 1];
        cumulative += std::sqrt(ddx * ddx + ddy * ddy);
      }
      cp.path_integrated_distances[i] = cumulative;
    }
    cp.furthest_reached_idx = 0;
    int pf_idx = std::min(PATH_FOLLOW_OFFSET, path_n - 1);
    cp.path_follow_target_x = path_x[pf_idx];
    cp.path_follow_target_y = path_y[pf_idx];
    int pa_idx = std::min(PATH_ANGLE_OFFSET, path_n - 1);
    cp.path_angle_active = true;
    cp.path_angle_target_x = path_x[pa_idx];
    cp.path_angle_target_y = path_y[pa_idx];
    cp.near_goal = false;
    cp.path_align_near_goal = false;
    cp.path_angle_near_goal = false;
    cp.prefer_forward_near_goal = false;
    cp.goal_x = goal_x;
    cp.goal_y = goal_y;
    cp.goal_yaw = goal_yaw;
    cost_gpu->setParams(cp);
  }

  SAMPLING_T* sampler = new SAMPLING_T();
  {
    auto sp = sampler->getParams();
    sp.std_dev[C_IND_CLASS(MotionModelParams, VX)] = VX_STD;
    sp.std_dev[C_IND_CLASS(MotionModelParams, VY)] = 0.0f;
    sp.std_dev[C_IND_CLASS(MotionModelParams, WZ)] = WZ_STD;
    const float control_cost_coeff = -OPT_GAMMA / LAMBDA;
    sp.control_cost_coeff[C_IND_CLASS(MotionModelParams, VX)] = control_cost_coeff;
    sp.control_cost_coeff[C_IND_CLASS(MotionModelParams, VY)] = control_cost_coeff;
    sp.control_cost_coeff[C_IND_CLASS(MotionModelParams, WZ)] = control_cost_coeff;
    sp.pure_noise_trajectories_percentage = 0.0f;
    sp.min_control[C_IND_CLASS(MotionModelParams, VX)] = VX_MIN;
    sp.max_control[C_IND_CLASS(MotionModelParams, VX)] = VX_MAX;
    sp.min_control[C_IND_CLASS(MotionModelParams, VY)] = 0.0f;
    sp.max_control[C_IND_CLASS(MotionModelParams, VY)] = 0.0f;
    sp.min_control[C_IND_CLASS(MotionModelParams, WZ)] = -WZ_MAX;
    sp.max_control[C_IND_CLASS(MotionModelParams, WZ)] = WZ_MAX;
    sp.exponents[C_IND_CLASS(MotionModelParams, VX)] = 1.0f;
    sp.exponents[C_IND_CLASS(MotionModelParams, VY)] = 0.0f;
    sp.exponents[C_IND_CLASS(MotionModelParams, WZ)] = 1.0f;
    sp.use_colored_noise = true;
    sampler->setParams(sp);
  }

  DYN_T::state_array x = DYN_T::state_array::Zero();

  printf("batch_size,time_steps,wall_mean_ms,wall_median_ms,wall_p95_ms,wall_max_ms,cpu_mean_ms\n");

  for (int time_steps : timesteps_sweep)
  {
    CONTROLLER_T::TEMPLATED_PARAMS controller_params;
    controller_params.dt_ = MODEL_DT;
    controller_params.lambda_ = LAMBDA;
    controller_params.num_iters_ = 1;
    controller_params.num_timesteps_ = time_steps;
    controller_params.dynamics_rollout_dim_ = dim3(32, DYN_T::STATE_DIM, 1);
    controller_params.cost_rollout_dim_ = dim3(32, 1, 1);
    controller_params.slide_control_scale_ = CONTROLLER_T::control_array::Ones();
    CONTROLLER_T* controller = new CONTROLLER_T(dynamics, cost_gpu, sampler, controller_params);
    controller->setRetryAttemptLimit(1);
    controller->setKernelChoice(kernelType::USE_SPLIT_KERNELS);

    {
      auto cp = cost_gpu->getParams();
      cp.mppi_num_timesteps = time_steps;
      cost_gpu->setParams(cp);
    }

    for (int i = 0; i < warmup; i++)
    {
      controller->computeControl(x, 1);
    }

    std::vector<double> wall_ms;
    std::vector<double> cpu_ms;
    wall_ms.reserve(reps);
    cpu_ms.reserve(reps);
    for (int i = 0; i < reps; i++)
    {
      struct rusage ru_before, ru_after;
      getrusage(RUSAGE_SELF, &ru_before);
      auto t0 = std::chrono::steady_clock::now();
      controller->computeControl(x, 1);
      auto t1 = std::chrono::steady_clock::now();
      getrusage(RUSAGE_SELF, &ru_after);
      wall_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      double cpu_delta = (ru_after.ru_utime.tv_sec - ru_before.ru_utime.tv_sec) * 1000.0 +
                          (ru_after.ru_utime.tv_usec - ru_before.ru_utime.tv_usec) / 1000.0 +
                          (ru_after.ru_stime.tv_sec - ru_before.ru_stime.tv_sec) * 1000.0 +
                          (ru_after.ru_stime.tv_usec - ru_before.ru_stime.tv_usec) / 1000.0;
      cpu_ms.push_back(cpu_delta);
    }

    std::sort(wall_ms.begin(), wall_ms.end());
    double wall_mean = 0.0;
    for (double v : wall_ms)
      wall_mean += v;
    wall_mean /= wall_ms.size();
    double wall_median = wall_ms[wall_ms.size() / 2];
    double wall_p95 = wall_ms[static_cast<size_t>(wall_ms.size() * 0.95)];
    double wall_max = wall_ms.back();
    double cpu_mean = 0.0;
    for (double v : cpu_ms)
      cpu_mean += v;
    cpu_mean /= cpu_ms.size();

    printf("%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f\n", NUM_ROLLOUTS, time_steps, wall_mean, wall_median, wall_p95, wall_max,
           cpu_mean);
    fflush(stdout);

    delete controller;
  }

  delete sampler;
  delete cost_gpu;
  delete dynamics;
  return 0;
}
