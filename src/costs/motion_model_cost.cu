#include "mppi_controller_cuda/costs/motion_model_cost.cuh"

MotionModelCost::MotionModelCost(cudaStream_t stream)
{
  obstacle_tex_ = new TwoDTextureHelper<float>(1, stream);
  this->bindToStream(stream);
}

MotionModelCost::~MotionModelCost()
{
  delete obstacle_tex_;
}

void MotionModelCost::bindToStream(cudaStream_t stream)
{
  if (obstacle_tex_)
  {
    obstacle_tex_->bindToStream(stream);
  }
  PARENT_CLASS::bindToStream(stream);
}

void MotionModelCost::GPUSetup()
{
  obstacle_tex_->GPUSetup();
  PARENT_CLASS::GPUSetup();
  HANDLE_ERROR(cudaMemcpyAsync(&(this->cost_d_->obstacle_tex_), &(obstacle_tex_->ptr_d_),
                               sizeof(TwoDTextureHelper<float>*), cudaMemcpyHostToDevice, this->stream_));
}

void MotionModelCost::paramsToDevice()
{
  if (this->GPUMemStatus_)
  {
    obstacle_tex_->copyToDevice();
  }
  PARENT_CLASS::paramsToDevice();
}

int MotionModelCost::getDeviceFurthestReachedIdx() const
{
  int val = 0;
  if (this->GPUMemStatus_)
  {
    HANDLE_ERROR(cudaMemcpy(&val, &(this->cost_d_->params_.furthest_reached_idx), sizeof(int),
                            cudaMemcpyDeviceToHost));
  }
  return val;
}

void MotionModelCost::freeCudaMem()
{
  if (this->GPUMemStatus_)
  {
    obstacle_tex_->freeCudaMem();
  }
  PARENT_CLASS::freeCudaMem();
}

void MotionModelCost::updateObstacleMap(std::vector<float>& costmap_data, int width, int height, float origin_x,
                                        float origin_y, float resolution)
{
  obstacle_tex_->updateOrigin(0, make_float3(origin_x, origin_y, 0.0f));
  obstacle_tex_->updateResolution(0, resolution);
  cudaExtent extent = make_cudaExtent(width, height, 0);
  obstacle_tex_->setExtent(0, extent);
  obstacle_tex_->updateTexture(0, costmap_data, false);
  if (!obstacle_tex_->checkTextureUse(0))
  {
    obstacle_tex_->enableTexture(0);
  }
}

__host__ __device__ float MotionModelCost::computeObstacleCost(const float* y, int* crash_status)
{
  if (!this->obstacle_tex_->checkTextureUse(0))
  {
    return 0.0f;
  }
  float x = y[O_IND_CLASS(DYN_P, X)];
  float yy = y[O_IND_CLASS(DYN_P, Y)];

  float tex_val = this->obstacle_tex_->queryTextureAtWorldPose(0, make_float3(x, yy, 0.0f));
  if (this->params_.footprint_length > 0)
  {
    float yaw = y[O_IND_CLASS(DYN_P, YAW)];
    float cy = cosf(yaw);
    float sy = sinf(yaw);
    for (int i = 0; i < this->params_.footprint_length; i++)
    {
      float fx = this->params_.footprint_x[i];
      float fy = this->params_.footprint_y[i];
      float3 world_pt = make_float3(x + fx * cy - fy * sy, yy + fx * sy + fy * cy, 0.0f);
      float pt_val = this->obstacle_tex_->queryTextureAtWorldPose(0, world_pt);
      if (pt_val > tex_val)
      {
        tex_val = pt_val;
      }
    }
  }

  if (tex_val <= 0.0f)
  {  // free space
    return 0.0f;
  }
  if (tex_val >= 253.0f || *crash_status)
  {
    *crash_status = 1;
    return this->params_.obstacle_collision_cost + this->params_.cost_collision_cost;
  }
  float dist_to_obj = (logf(252.0f) - logf(tex_val)) / this->params_.obstacle_scale_factor;
  float cost = 0.0f;
  if (dist_to_obj < this->params_.obstacle_collision_margin)
  {
    cost = this->params_.obstacle_traj_weight * (this->params_.obstacle_collision_margin - dist_to_obj);
  }
  else if (!this->params_.obstacles_near_goal)
  {
    cost = this->params_.obstacle_repulsion_weight * (this->params_.obstacle_inflation_radius - dist_to_obj);
  }
  cost = fmaxf(cost, 0.0f);
  cost = this->params_.obstacle_power > 1 ? powf(cost, this->params_.obstacle_power) : cost;

  float cost_critic_cost = 0.0f;
  if (!this->params_.obstacles_near_goal)
  {
    cost_critic_cost = this->params_.cost_weight * tex_val;
  }
  cost_critic_cost = this->params_.cost_power > 1 ? powf(cost_critic_cost, this->params_.cost_power) : cost_critic_cost;

  return cost + cost_critic_cost;
}

float MotionModelCost::computeStateCost(const Eigen::Ref<const output_array> y, int timestep, int* crash_status)
{
  float cost = 0;

  if (this->params_.near_goal)
  {
    float dx = y[O_IND_CLASS(DYN_P, X)] - this->params_.goal_x;
    float dy = y[O_IND_CLASS(DYN_P, Y)] - this->params_.goal_y;
    float dist = sqrtf(dx * dx + dy * dy);
    cost += this->params_.goal_weight *
            (this->params_.goal_power > 1 ? powf(dist, this->params_.goal_power) : dist);
  }

  if (this->params_.near_goal)
  {
    float dyaw = fabsf(angle_utils::shortestAngularDistance(y[O_IND_CLASS(DYN_P, YAW)], this->params_.goal_yaw));
    cost += this->params_.goal_angle_weight *
            (this->params_.goal_angle_power > 1 ? powf(dyaw, this->params_.goal_angle_power) : dyaw);
  }

  // Obstacle avoidance (obstacles_critic).
  cost += computeObstacleCost(y.data(), crash_status);

  // Path_align_critic
  if (this->params_.path_align_weight > 0.0f && !this->params_.path_align_near_goal &&
      this->params_.furthest_reached_idx >= this->params_.path_align_offset_from_furthest)
  {
    float traj_dist = y[O_IND_CLASS(DYN_P, TRAJ_DIST)];
    int path_pt = findClosestPathPt(this->params_.path_integrated_distances, this->params_.furthest_reached_idx,
                                    traj_dist);
    if (this->params_.path_valid[path_pt])
    {
      float dx = this->params_.path_x[path_pt] - y[O_IND_CLASS(DYN_P, X)];
      float dy = this->params_.path_y[path_pt] - y[O_IND_CLASS(DYN_P, Y)];
      float dist = sqrtf(dx * dx + dy * dy);
      cost += this->params_.path_align_weight *
              (this->params_.path_align_power > 1 ? powf(dist, this->params_.path_align_power) : dist);
    }
  }

  return cost;
}

float MotionModelCost::terminalCost(const Eigen::Ref<const output_array> y)
{
  float cost = 0;
  // Path_follow_critic
  if (this->params_.path_follow_weight > 0.0f && !this->params_.near_goal)
  {
    float dx = this->params_.path_follow_target_x - y[O_IND_CLASS(DYN_P, X)];
    float dy = this->params_.path_follow_target_y - y[O_IND_CLASS(DYN_P, Y)];
    float dist = sqrtf(dx * dx + dy * dy);
    cost += this->params_.path_follow_weight *
            (this->params_.path_follow_power > 1 ? powf(dist, this->params_.path_follow_power) : dist);
  }
  // path_angle_critic
  if (this->params_.path_angle_weight > 0.0f && this->params_.path_angle_active && !this->params_.path_angle_near_goal)
  {
    float dx = this->params_.path_angle_target_x - y[O_IND_CLASS(DYN_P, X)];
    float dy = this->params_.path_angle_target_y - y[O_IND_CLASS(DYN_P, Y)];
    float yaw_to_target = atan2f(dy, dx);
    float yaw_err = fabsf(angle_utils::shortestAngularDistance(y[O_IND_CLASS(DYN_P, YAW)], yaw_to_target));
    cost += this->params_.path_angle_weight *
            (this->params_.path_angle_power > 1 ? powf(yaw_err, this->params_.path_angle_power) : yaw_err);
  }
  return cost;
}

float MotionModelCost::computeControlCost(const Eigen::Ref<const control_array> u, int timestep, int* crash)
{
  float cost = 0;
  float vx = u[C_IND_CLASS(DYN_P, VX)];
  float vy = u[C_IND_CLASS(DYN_P, VY)];
  float wz = u[C_IND_CLASS(DYN_P, WZ)];

  // prefer_forward_critic / twirling_critic
  if (!this->params_.prefer_forward_near_goal)
  {
    float reverse = fmaxf(-vx, 0.0f);
    cost += this->params_.prefer_forward_weight *
            (this->params_.prefer_forward_power > 1 ? powf(reverse, this->params_.prefer_forward_power) : reverse);

    float wz_abs = fabsf(wz);
    cost += this->params_.twirling_weight *
            (this->params_.twirling_power > 1 ? powf(wz_abs, this->params_.twirling_power) : wz_abs);
  }
  else
  {
    float wz_abs = fabsf(wz);
    cost += this->params_.near_goal_twirling_weight *
            (this->params_.near_goal_twirling_power > 1 ? powf(wz_abs, this->params_.near_goal_twirling_power) :
                                                           wz_abs);
  }

  // velocity_deadband_critic
  float deadband = fmaxf(this->params_.deadband_vx - fabsf(vx), 0.0f) +
                   fmaxf(this->params_.deadband_wz - fabsf(wz), 0.0f);
  if (this->params_.is_holonomic)
  {
    deadband += fmaxf(this->params_.deadband_vy - fabsf(vy), 0.0f);
  }
  cost += this->params_.velocity_deadband_weight *
          (this->params_.velocity_deadband_power > 1 ? powf(deadband, this->params_.velocity_deadband_power) :
                                                        deadband);

  // constraint_critic
  float over;
  if (this->params_.is_holonomic)
  {
    float sgn = vx >= 0.0f ? 1.0f : -1.0f;
    float vel_total = sgn * sqrtf(vx * vx + vy * vy);
    float min_sgn = this->params_.vx_min > 0.0f ? 1.0f : -1.0f;
    float max_vel = sqrtf(this->params_.vx_max * this->params_.vx_max + this->params_.vy_max * this->params_.vy_max);
    float min_vel =
        min_sgn * sqrtf(this->params_.vx_min * this->params_.vx_min + this->params_.vy_max * this->params_.vy_max);
    over = fmaxf(vel_total - max_vel, 0.0f) + fmaxf(min_vel - vel_total, 0.0f);
  }
  else
  {
    over = fmaxf(vx - this->params_.vx_max, 0.0f) + fmaxf(this->params_.vx_min - vx, 0.0f);
  }
  cost += this->params_.constraint_weight *
          (this->params_.constraint_power > 1 ? powf(over, this->params_.constraint_power) : over);

  return cost;
}

__device__ float MotionModelCost::computeStateCost(float* y, int timestep, float* theta_c, int* crash_status)
{
  float cost = 0;

  if (this->params_.near_goal)
  {
    float dx = y[O_IND_CLASS(DYN_P, X)] - this->params_.goal_x;
    float dy = y[O_IND_CLASS(DYN_P, Y)] - this->params_.goal_y;
    float dist = sqrtf(dx * dx + dy * dy);
    cost += this->params_.goal_weight * (this->params_.goal_power > 1 ? powf(dist, this->params_.goal_power) : dist);
  }

  // goal_angle_critic
  if (this->params_.near_goal)
  {
    float dyaw = fabsf(angle_utils::shortestAngularDistance(y[O_IND_CLASS(DYN_P, YAW)], this->params_.goal_yaw));
    cost += this->params_.goal_angle_weight *
            (this->params_.goal_angle_power > 1 ? powf(dyaw, this->params_.goal_angle_power) : dyaw);
  }

  cost += computeObstacleCost(y, crash_status);

  if (this->params_.path_align_weight > 0.0f && !this->params_.path_align_near_goal &&
      this->params_.furthest_reached_idx >= this->params_.path_align_offset_from_furthest)
  {
    // Arc-length matching
    float traj_dist = y[O_IND_CLASS(DYN_P, TRAJ_DIST)];
    int path_pt = findClosestPathPt(this->params_.path_integrated_distances, this->params_.furthest_reached_idx,
                                    traj_dist);
    // path_align_critic
    if (this->params_.path_valid[path_pt])
    {
      float dx = this->params_.path_x[path_pt] - y[O_IND_CLASS(DYN_P, X)];
      float dy = this->params_.path_y[path_pt] - y[O_IND_CLASS(DYN_P, Y)];
      float dist = sqrtf(dx * dx + dy * dy);
      cost += this->params_.path_align_weight *
              (this->params_.path_align_power > 1 ? powf(dist, this->params_.path_align_power) : dist);
    }
  }

  return cost;
}

__device__ float MotionModelCost::terminalCost(float* y, float* theta_c)
{
  float cost = 0;
  if (this->params_.path_follow_weight > 0.0f && !this->params_.near_goal)
  {
    float dx = this->params_.path_follow_target_x - y[O_IND_CLASS(DYN_P, X)];
    float dy = this->params_.path_follow_target_y - y[O_IND_CLASS(DYN_P, Y)];
    float dist = sqrtf(dx * dx + dy * dy);
    cost += this->params_.path_follow_weight *
            (this->params_.path_follow_power > 1 ? powf(dist, this->params_.path_follow_power) : dist);
  }
  if (this->params_.path_angle_weight > 0.0f && this->params_.path_angle_active && !this->params_.path_angle_near_goal)
  {
    float dx = this->params_.path_angle_target_x - y[O_IND_CLASS(DYN_P, X)];
    float dy = this->params_.path_angle_target_y - y[O_IND_CLASS(DYN_P, Y)];
    float yaw_to_target = atan2f(dy, dx);
    float yaw_err = fabsf(angle_utils::shortestAngularDistance(y[O_IND_CLASS(DYN_P, YAW)], yaw_to_target));
    cost += this->params_.path_angle_weight *
            (this->params_.path_angle_power > 1 ? powf(yaw_err, this->params_.path_angle_power) : yaw_err);
  }
  return cost;
}

__device__ float MotionModelCost::computeControlCost(float* u, int timestep, float* theta_c, int* crash)
{
  float cost = 0;
  float vx = u[C_IND_CLASS(DYN_P, VX)];
  float vy = u[C_IND_CLASS(DYN_P, VY)];
  float wz = u[C_IND_CLASS(DYN_P, WZ)];

  if (!this->params_.prefer_forward_near_goal)
  {
    float reverse = fmaxf(-vx, 0.0f);
    cost += this->params_.prefer_forward_weight *
            (this->params_.prefer_forward_power > 1 ? powf(reverse, this->params_.prefer_forward_power) : reverse);

    float wz_abs = fabsf(wz);
    cost += this->params_.twirling_weight *
            (this->params_.twirling_power > 1 ? powf(wz_abs, this->params_.twirling_power) : wz_abs);
  }
  else
  {
    float wz_abs = fabsf(wz);
    cost += this->params_.near_goal_twirling_weight *
            (this->params_.near_goal_twirling_power > 1 ? powf(wz_abs, this->params_.near_goal_twirling_power) :
                                                           wz_abs);
  }

  float deadband = fmaxf(this->params_.deadband_vx - fabsf(vx), 0.0f) +
                   fmaxf(this->params_.deadband_wz - fabsf(wz), 0.0f);
  if (this->params_.is_holonomic)
  {
    deadband += fmaxf(this->params_.deadband_vy - fabsf(vy), 0.0f);
  }
  cost += this->params_.velocity_deadband_weight *
          (this->params_.velocity_deadband_power > 1 ? powf(deadband, this->params_.velocity_deadband_power) :
                                                        deadband);

  float over;
  if (this->params_.is_holonomic)
  {
    float sgn = vx >= 0.0f ? 1.0f : -1.0f;
    float vel_total = sgn * sqrtf(vx * vx + vy * vy);
    float min_sgn = this->params_.vx_min > 0.0f ? 1.0f : -1.0f;
    float max_vel = sqrtf(this->params_.vx_max * this->params_.vx_max + this->params_.vy_max * this->params_.vy_max);
    float min_vel =
        min_sgn * sqrtf(this->params_.vx_min * this->params_.vx_min + this->params_.vy_max * this->params_.vy_max);
    over = fmaxf(vel_total - max_vel, 0.0f) + fmaxf(min_vel - vel_total, 0.0f);
  }
  else
  {
    over = fmaxf(vx - this->params_.vx_max, 0.0f) + fmaxf(this->params_.vx_min - vx, 0.0f);
  }
  cost += this->params_.constraint_weight *
          (this->params_.constraint_power > 1 ? powf(over, this->params_.constraint_power) : over);

  return cost;
}
