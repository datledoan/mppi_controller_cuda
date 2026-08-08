#ifndef MOTION_MODEL_COST_CUH_
#define MOTION_MODEL_COST_CUH_

#include <cfloat>
#include "mppi_controller_cuda/costs/cost.cuh"
#include "mppi_controller_cuda/dynamics/motion_model.cuh"
#include "mppi_controller_cuda/utils/texture_helpers/two_d_texture_helper.cuh"

struct MotionModelCostParams : public CostParams<MotionModel::CONTROL_DIM>
{
  static constexpr int MAX_PATH_POINTS = 200;
  float path_x[MAX_PATH_POINTS] = { 0 };
  float path_y[MAX_PATH_POINTS] = { 0 };
  int path_length = 0;

  float path_integrated_distances[MAX_PATH_POINTS] = { 0 };

  float path_follow_target_x = 0.0f;
  float path_follow_target_y = 0.0f;
  float path_follow_weight = 1.0f;
  int path_follow_power = 1;

  bool path_valid[MAX_PATH_POINTS] = { false };

  bool near_goal = false;
  float goal_x = 0.0f;
  float goal_y = 0.0f;
  float goal_yaw = 0.0f;
  float goal_weight = 1.0f;
  int goal_power = 1;
  float goal_threshold = 0.0f;
  float goal_angle_weight = 1.0f;
  int goal_angle_power = 1;
  float goal_angle_threshold = 0.0f;

  bool prefer_forward_near_goal = false;
  float prefer_forward_weight = 0.0f;
  int prefer_forward_power = 1;
  float twirling_weight = 0.0f;
  int twirling_power = 1;

  float near_goal_twirling_weight = 0.0f;
  int near_goal_twirling_power = 1;

  float velocity_deadband_weight = 0.0f;
  int velocity_deadband_power = 1;
  float deadband_vx = 0.0f;
  float deadband_vy = 0.0f;
  float deadband_wz = 0.0f;

  float constraint_weight = 0.0f;
  int constraint_power = 1;
  float vx_max = 0.0f;
  float vx_min = 0.0f;
  float vy_max = 0.0f;
  float wz_max = 0.0f;

  bool is_holonomic = false;

  float obstacle_scale_factor = 10.0f;    // matches costmap's cost_scaling_factor
  float obstacle_inflation_radius = 0.55f;  // meters
  float obstacle_collision_margin = 0.1f;  // meters
  float obstacle_collision_cost = 10000.0f;
  float obstacle_traj_weight = 1.0f;
  float obstacle_repulsion_weight = 0.0f;
  int obstacle_power = 1;
  bool obstacles_near_goal = false;

  float cost_weight = 0.0f;
  int cost_power = 1;
  float cost_critical_cost = 300.0f;  // unreachable in point-robot mode, see computeObstacleCost
  float cost_collision_cost = 1000000.0f;

  static constexpr int MAX_FOOTPRINT_POINTS = 16;
  float footprint_x[MAX_FOOTPRINT_POINTS] = { 0 };
  float footprint_y[MAX_FOOTPRINT_POINTS] = { 0 };
  int footprint_length = 0;

  int furthest_reached_idx = 0;
  int mppi_num_timesteps = 1;
  bool path_align_near_goal = false;
  int path_align_offset_from_furthest = 0;
  float path_align_weight = 0.0f;
  int path_align_power = 1;
  bool path_angle_near_goal = false;
  int path_angle_offset_from_furthest = 0;
  float path_angle_weight = 0.0f;
  int path_angle_power = 1;
  bool path_angle_active = false;
  float path_angle_target_x = 0.0f;
  float path_angle_target_y = 0.0f;
};

class MotionModelCost : public Cost<MotionModelCost, MotionModelCostParams, MotionModel::DYN_PARAMS_T>
{
public:
  using PARENT_CLASS = Cost<MotionModelCost, MotionModelCostParams, MotionModel::DYN_PARAMS_T>;
  using DYN_P = PARENT_CLASS::TEMPLATED_DYN_PARAMS;

  MotionModelCost(cudaStream_t stream = nullptr);

  ~MotionModelCost();

  void bindToStream(cudaStream_t stream);

  void GPUSetup();

  void paramsToDevice();

  __host__ int getDeviceFurthestReachedIdx() const;

  void freeCudaMem();

  std::string getCostFunctionName() const override
  {
    return "MotionModel Cost";
  }

  void updateObstacleMap(std::vector<float>& costmap_data, int width, int height, float origin_x, float origin_y,
                         float resolution);

  float computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status);

  float terminalCost(const Eigen::Ref<const output_array> y);

  float computeControlCost(const Eigen::Ref<const control_array> u, int timestep, int* crash);

  __host__ __device__ float computeObstacleCost(const float* y, int* crash_status);

  __device__ float computeStateCost(float* y, int timestep, float* theta_c, int* crash_status);

  __device__ float terminalCost(float* y, float* theta_c);

  __device__ float computeControlCost(float* u, int timestep, float* theta_c, int* crash);

  __device__ void resetFurthestReached()
  {
    params_.furthest_reached_idx = 0;
  }

  __device__ void updateFurthestReachedFromEndpoint(float end_x, float end_y)
  {
#ifdef __CUDA_ARCH__
    if (params_.path_length <= 1)
    {
      return;
    }
    float best_d = FLT_MAX;
    int best_idx = 0;
    for (int j = 0; j < params_.path_length; j++)
    {
      float dx = params_.path_x[j] - end_x;
      float dy = params_.path_y[j] - end_y;
      float d = dx * dx + dy * dy;
      if (d < best_d)
      {
        best_d = d;
        best_idx = j;
      }
    }
    atomicMax(&params_.furthest_reached_idx, best_idx);
#endif
  }

  __host__ __device__ static int findClosestPathPt(const float* path_integrated_distances, int count, float dist)
  {
    if (count <= 1)
    {
      return 0;
    }
    float distim1 = 0.0f;
    for (int i = 1; i < count; i++)
    {
      float disti = path_integrated_distances[i];
      if (disti > dist)
      {
        return (dist - distim1 < disti - dist) ? i - 1 : i;
      }
      distim1 = disti;
    }
    return count - 1;
  }

  TwoDTextureHelper<float>* obstacle_tex_ = nullptr;
};

#endif  // MOTION_MODEL_COST_CUH_
