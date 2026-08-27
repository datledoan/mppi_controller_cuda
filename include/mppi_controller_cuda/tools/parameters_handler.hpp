#ifndef MPPI_CONTROLLER_CUDA__TOOLS__PARAMETERS_HANDLER_HPP_
#define MPPI_CONTROLLER_CUDA__TOOLS__PARAMETERS_HANDLER_HPP_

#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>

#include "nav2_util/node_utils.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/parameter_value.hpp"
#include "rclcpp_lifecycle/lifecycle_node.hpp"

namespace mppi
{

struct Parameters
{
  float vx_max;
  float vy_max;
  float vx_min;
  float vx_std;
  float vy_std;
  float wz_std;
  float optimizer_gamma;
  float pure_noise_trajectories_percentage;
  // Colored-noise spectral exponent per control dim: 0=white, 1=pink, 2=red/Brownian.
  float vx_noise_color_exponent;
  float vy_noise_color_exponent;
  float wz_noise_color_exponent;
  float ax_max;
  float ax_min;
  float ay_max;
  float az_max;
  float wz_max;
  
  float ackermann_constraints_min_turning_r;

  //constrain critics
  int constrain_cost_power;
  float constrain_cost_weight;

  // goal angle critic
  int goal_angle_cost_power;
  float goal_angle_cost_weight;

  // goal critic
  int goal_cost_power;
  float goal_cost_weight;
  float goal_threshold_to_consider;
  float path_align_threshold_to_consider;
  float path_angle_threshold_to_consider;
  float prefer_forward_threshold_to_consider;

  // obstacles critic
  bool obstacles_consider_footprint;
  int obstacles_cost_power;
  float obstacles_repulsion_weight;
  float obstacles_critical_weight;
  float obstacles_collision_cost;
  float obstacles_collision_margin_distance;
  float obstacles_inflation_scale_factor;
  float obstacles_inflation_radius;
  float obstacles_near_goal_distance;

  // cost critic
  int cost_power;
  float cost_weight;
  float cost_critical_cost;
  float cost_collision_cost;

  // path align critics
  int path_align_offset_from_furthest;
  float path_align_max_path_occupancy_ratio;
  int path_align_cost_power;
  float path_align_cost_weight;

  // path angle critic
  int path_angle_offset_from_furthest;
  int path_angle_cost_power;
  float path_angle_cost_weight;
  float path_angle_max_angle_to_furthest;

  // path follow critic
  int path_follow_offset_from_furthest;
  int path_follow_cost_power;
  float path_follow_cost_weight;

  // prefer forward critic
  int prefer_forward_cost_power;
  float prefer_forward_cost_weight;

  // twirling critic
  int twirling_cost_power;
  float twirling_cost_weight;
  // near-goal twirling critic
  int near_goal_twirling_cost_power;
  float near_goal_twirling_cost_weight;

  // velocity deadband critic
  int velocity_deadband_cost_power;
  float velocity_deadband_cost_weight;
  double velocity_deadband_velocities_1;
  double velocity_deadband_velocities_2;
  double velocity_deadband_velocities_3;

  // path handler
  double path_handler_max_robot_pose_search_dist;
  double path_handler_prune_distance;
  double path_handler_transform_tolerance;
  bool path_handler_enforce_path_inversion;
  double path_handler_inversion_xy_tolerance;
  double path_handler_inversion_yaw_tolerance;

  // Noise generator
  bool noise_generator_regenerate_noises;

  // Optimizer
  float optimizer_model_dt;
  int optimizer_time_steps;
  int optimizer_iteration_count;
  int optimizer_retry_attempt_limit;
  float optimizer_temperature;
  std::string optimizer_motion_model;
  bool noise_generator_use_colored_noise;

  // Trajectory visualizer
  bool trajectory_visualizer_visualize_optimal;
  bool trajectory_visualizer_visualize_samples;
  int trajectory_visualizer_time_step;
  float trajectory_visualizer_samples_percentage;
};

/**
 * @class Parameter Type enum
 */
enum class ParameterType { Dynamic, Static };

/**
 * @class mppi::ParametersHandler
 * @brief Handles getting parameters and dynamic parameter changes, and owns
 * this plugin's Parameters struct (populated once at construction time).
 */
class ParametersHandler
{
public:
  using get_param_func_t = void (const rclcpp::Parameter & param);
  using post_callback_t = void ();
  using pre_callback_t = void ();

  ParametersHandler() = default;

  /**
    * @brief Constructor for mppi::ParametersHandler
    * @param parent Weak ptr to node
    * @param plugin_name Name this controller plugin instance was loaded as
    * (e.g. "FollowPath") -- every parameter is declared/read under this
    * namespace, matching nav2's plugin-parameter convention.
    */
  ParametersHandler(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    const std::string & plugin_name);

  ~ParametersHandler();

  /**
    * @brief Starts processing dynamic parameter changes. Must be called from
    * activate(), after every static/dynamic param has already been declared
    * by the constructor.
    */
  void start();

  rcl_interfaces::msg::SetParametersResult dynamicParamsCallback(
    std::vector<rclcpp::Parameter> parameters);

  inline auto getParamGetter(const std::string & ns);

  template<typename T>
  void addPostCallback(T && callback);

  template<typename T>
  void addPreCallback(T && callback);

  template<typename T>
  void setDynamicParamCallback(T & setting, const std::string & name);

  std::mutex * getLock()
  {
    return &parameters_change_mutex_;
  }

  template<typename T>
  void addDynamicParamCallback(const std::string & name, T && callback);

  /**
    * @brief Get the Parameters struct populated by the constructor
    */
  Parameters * getParams() {return &params_;}

protected:
  template<typename SettingT, typename ParamT>
  void getParam(
    SettingT & setting, const std::string & name, ParamT default_value,
    ParameterType param_type = ParameterType::Dynamic);

  template<typename ParamT, typename SettingT, typename NodeT>
  void setParam(SettingT & setting, const std::string & name, NodeT node) const;

  template<typename T>
  static auto as(const rclcpp::Parameter & parameter);

  /**
    * @brief Declares/reads every field of `params_`, matching the CPU/CUDA
    * reference's parameter names and default values.
    */
  void initializeParams();

  std::mutex parameters_change_mutex_;
  rclcpp::Logger logger_{rclcpp::get_logger("MPPIControllerCUDA")};
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    on_set_param_handler_;
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::string plugin_name_;

  bool verbose_{false};

  std::unordered_map<std::string, std::function<get_param_func_t>>
  get_param_callbacks_;

  std::vector<std::function<pre_callback_t>> pre_callbacks_;
  std::vector<std::function<post_callback_t>> post_callbacks_;

  Parameters params_;
};

inline auto ParametersHandler::getParamGetter(const std::string & ns)
{
  return [this, ns](
    auto & setting, const std::string & name, auto default_value,
    ParameterType param_type = ParameterType::Dynamic) {
           getParam(
             setting, ns.empty() ? name : ns + "." + name,
             std::move(default_value), param_type);
         };
}

template<typename T>
void ParametersHandler::addDynamicParamCallback(const std::string & name, T && callback)
{
  get_param_callbacks_[name] = callback;
}

template<typename T>
void ParametersHandler::addPostCallback(T && callback)
{
  post_callbacks_.push_back(callback);
}

template<typename T>
void ParametersHandler::addPreCallback(T && callback)
{
  pre_callbacks_.push_back(callback);
}

template<typename SettingT, typename ParamT>
void ParametersHandler::getParam(
  SettingT & setting, const std::string & name,
  ParamT default_value,
  ParameterType param_type)
{
  auto node = node_.lock();

  nav2_util::declare_parameter_if_not_declared(
    node, name, rclcpp::ParameterValue(default_value));

  setParam<ParamT>(setting, name, node);

  if (param_type == ParameterType::Dynamic) {
    setDynamicParamCallback(setting, name);
  }
}

template<typename ParamT, typename SettingT, typename NodeT>
void ParametersHandler::setParam(
  SettingT & setting, const std::string & name, NodeT node) const
{
  ParamT param_in{};
  node->get_parameter(name, param_in);
  setting = static_cast<SettingT>(param_in);
}

template<typename T>
void ParametersHandler::setDynamicParamCallback(T & setting, const std::string & name)
{
  if (get_param_callbacks_.find(name) != get_param_callbacks_.end()) {
    return;
  }

  auto callback = [this, &setting, name](const rclcpp::Parameter & param) {
      setting = as<T>(param);

      if (verbose_) {
        RCLCPP_INFO(logger_, "Dynamic parameter changed: %s", std::to_string(param).c_str());
      }
    };

  addDynamicParamCallback(name, callback);

  if (verbose_) {
    RCLCPP_INFO(logger_, "Dynamic Parameter added %s", name.c_str());
  }
}

template<typename T>
auto ParametersHandler::as(const rclcpp::Parameter & parameter)
{
  if constexpr (std::is_same_v<T, bool>) {
    return parameter.as_bool();
  } else if constexpr (std::is_integral_v<T>) {
    return parameter.as_int();
  } else if constexpr (std::is_floating_point_v<T>) {
    return parameter.as_double();
  } else if constexpr (std::is_same_v<T, std::string>) {
    return parameter.as_string();
  } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
    return parameter.as_integer_array();
  } else if constexpr (std::is_same_v<T, std::vector<double>>) {
    return parameter.as_double_array();
  } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
    return parameter.as_string_array();
  } else if constexpr (std::is_same_v<T, std::vector<bool>>) {
    return parameter.as_bool_array();
  }
}

}  // namespace mppi

#endif  // MPPI_CONTROLLER_CUDA__TOOLS__PARAMETERS_HANDLER_HPP_
