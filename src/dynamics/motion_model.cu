#include "mppi_controller_cuda/dynamics/motion_model.cuh"
#include "mppi_controller_cuda/utils/math_utils.h"

void MotionModel::computeStateDeriv(const Eigen::Ref<const state_array>& x, const Eigen::Ref<const control_array>& u,
                                    Eigen::Ref<state_array> xdot)
{
  float vx = u[C_INDEX(VX)];
  float wz = u[C_INDEX(WZ)];
  float yaw = x[S_INDEX(YAW)];
  xdot[S_INDEX(X)] = vx * cosf(yaw);
  xdot[S_INDEX(Y)] = vx * sinf(yaw);
  if (this->params_.is_holonomic)
  {
    float vy = u[C_INDEX(VY)];
    xdot[S_INDEX(X)] -= vy * sinf(yaw);
    xdot[S_INDEX(Y)] += vy * cosf(yaw);
  }
  xdot[S_INDEX(YAW)] = wz;
}

__device__ void MotionModel::computeStateDeriv(float* x, float* u, float* xdot, float* theta_s)
{
  float vx = u[C_INDEX(VX)];
  float wz = u[C_INDEX(WZ)];
  float yaw = angle_utils::normalizeAngle(x[S_INDEX(YAW)]);
  xdot[S_INDEX(X)] = vx * __cosf(yaw);
  xdot[S_INDEX(Y)] = vx * __sinf(yaw);
  if (this->params_.is_holonomic)
  {
    float vy = u[C_INDEX(VY)];
    xdot[S_INDEX(X)] -= vy * __sinf(yaw);
    xdot[S_INDEX(Y)] += vy * __cosf(yaw);
  }
  xdot[S_INDEX(YAW)] = wz;
}

void MotionModel::enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control)
{
  PARENT_CLASS::enforceConstraints(state, control);
  if (this->params_.min_turning_radius > 0.0f)
  {
    float vx = control[C_INDEX(VX)];
    float max_wz = fabsf(vx) / this->params_.min_turning_radius;
    control[C_INDEX(WZ)] = fminf(fmaxf(control[C_INDEX(WZ)], -max_wz), max_wz);
  }
  if (!this->params_.is_holonomic)
  {
    control[C_INDEX(VY)] = 0.0f;
  }
}

__device__ void MotionModel::enforceConstraints(float* state, float* control)
{
  PARENT_CLASS::enforceConstraints(state, control);
  if (this->params_.min_turning_radius > 0.0f)
  {
    float vx = control[C_INDEX(VX)];
    float max_wz = fabsf(vx) / this->params_.min_turning_radius;
    control[C_INDEX(WZ)] = fminf(fmaxf(control[C_INDEX(WZ)], -max_wz), max_wz);
  }
  if (!this->params_.is_holonomic)
  {
    control[C_INDEX(VY)] = 0.0f;
  }
}

#ifdef FASTER_DYN_COMPUTATIONS
__device__  void MotionModel::step(float* x, float* x_next, float* xdot,
                                   float* u, float* output, float* theta_s, const float t,
                                   const float dt)
{
  float vx = u[C_INDEX(VX)];
  float vy = u[C_INDEX(VY)];
  float wz = u[C_INDEX(WZ)];
  float yaw = angle_utils::normalizeAngle(x[S_INDEX(YAW)]);
  bool holonomic = this->params_.is_holonomic;
  int p_ind, step;
  mppi::p1::getParallel1DIndex<mppi::p1::Parallel1Dir::THREAD_Y>(p_ind, step);
  for (int i = p_ind; i < STATE_DIM; i += step)
  {
    switch (i)
    {
      case S_INDEX(X):
        xdot[S_INDEX(X)] = vx * __cosf(yaw);
        if (holonomic)
        {
          xdot[S_INDEX(X)] -= vy * __sinf(yaw);
        }
        x_next[S_INDEX(X)] = x[S_INDEX(X)] + xdot[S_INDEX(X)] * dt;
        output[O_INDEX(X)] = x_next[S_INDEX(X)];
        break;
      case S_INDEX(Y):
        xdot[S_INDEX(Y)] = vx * __sinf(yaw);
        if (holonomic)
        {
          xdot[S_INDEX(Y)] += vy * __cosf(yaw);
        }
        x_next[S_INDEX(Y)] = x[S_INDEX(Y)] + xdot[S_INDEX(Y)] * dt;
        output[O_INDEX(Y)] = x_next[S_INDEX(Y)];
        break;
      case S_INDEX(YAW):
        xdot[S_INDEX(YAW)] = wz;
        x_next[S_INDEX(YAW)] = x[S_INDEX(YAW)] + xdot[S_INDEX(YAW)] * dt;
        output[O_INDEX(YAW)] = x_next[S_INDEX(YAW)];
        break;
    }
  }
}
#endif

MotionModel::state_array MotionModel::stateFromMap(const std::map<std::string, float>& map)
{
  state_array s;
  s[S_INDEX(X)] = map.at("X");
  s[S_INDEX(Y)] = map.at("POS_Y");
  s[S_INDEX(YAW)] = map.at("YAW");
  return s;
}
