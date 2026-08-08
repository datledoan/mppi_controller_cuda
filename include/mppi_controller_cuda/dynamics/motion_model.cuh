#ifndef MOTION_MODEL_CUH
#define MOTION_MODEL_CUH

#include "mppi_controller_cuda/dynamics/dynamics.cuh"

using namespace MPPI_internal;

#define FASTER_DYN_COMPUTATIONS

struct MotionModelParams : public DynamicsParams
{
  enum class StateIndex : int
  {
    X = 0,
    Y,
    YAW,
#ifdef FASTER_DYN_COMPUTATIONS
    FILLER,
#endif
    NUM_STATES
  };

  enum class ControlIndex : int
  {
    VX = 0,
    VY,
    WZ,
    NUM_CONTROLS
  };

  enum class OutputIndex : int
  {
    X = 0,
    Y,
    YAW,
    TRAJ_DIST,
    NUM_OUTPUTS
  };

  float min_turning_radius = 0.0f;
  bool is_holonomic = false;

  float ax_max = 3.0f;
  float ax_min = -3.0f;
  float ay_max = 3.0f;
  float az_max = 3.5f;
  float current_vx = 0.0f;
  float current_vy = 0.0f;
  float current_wz = 0.0f;
};

class MotionModel : public Dynamics<MotionModel, MotionModelParams>
{
public:
  using PARENT_CLASS = Dynamics<MotionModel, MotionModelParams>;
  static const int SHARED_MEM_REQUEST_GRD_BYTES = 4;

  std::string getDynamicsModelName() const override
  {
    return "MotionModel";
  }

  MotionModel(cudaStream_t stream = nullptr) : PARENT_CLASS(stream)
  {
  }

  void computeStateDeriv(const Eigen::Ref<const state_array>& x, const Eigen::Ref<const control_array>& u,
                         Eigen::Ref<state_array> x_dot);

  __device__ inline void computeStateDeriv(float* x, float* u, float* x_dot, float* theta_s);

  void enforceConstraints(Eigen::Ref<state_array> state, Eigen::Ref<control_array> control);

  __device__ void enforceConstraints(float* state, float* control);

#ifdef FASTER_DYN_COMPUTATIONS
  __device__  void step(float* x, float* x_next, float* xdot,
                        float* u, float* output, float* theta_s, const float t,
                        const float dt);

  using PARENT_CLASS::step;
#endif

  state_array stateFromMap(const std::map<std::string, float>& map) override;

  __device__ void initializeDynamics(float* state, float* control, float* output, float* theta_s, float t_0, float dt)
  {
  }
  using PARENT_CLASS::initializeDynamics;
};

#endif // MOTION_MODEL_CUH
