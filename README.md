# Model Predictive Path Integral Controller

CUDA-accelerated MPPI controller based on [nav2_mppi_controller](https://github.com/ros-navigation/navigation2/tree/main/nav2_mppi_controller) built on the [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) CUDA framework.

>For ROS1, see the [`noetic`](https://github.com/datledoan/mppi_controller_cuda/tree/noetic) branch.

>For ROS2-Jazzy, see the [`jazzy`](https://github.com/datledoan/mppi_controller_cuda/tree/jazzy) branch.

# Installation

Requires an NVIDIA GPU + CUDA toolkit. 
Tested with ROS2 Humble, Ubuntu 22.04, CUDA 12.6, CPU AMD Ryzen 5 5500H (8 threads) + RTX 2050 (compute capability 8.6); **Jetson AGX Orin Developer Kit** (JetPack 6.2.3, Ubuntu 22.04.5, CUDA 12.6, compute capability 8.7).

* Install the CUDA toolkit (see [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)).
* Clone and build:
    ```sh
    cd your_ws/src
    git clone -b humble https://github.com/datledoan/mppi_controller_cuda.git
    cd your_ws
    colcon build --packages-select mppi_controller_cuda --parallel-workers 2
    ```
  `CMAKE_CUDA_ARCHITECTURES` defaults to `86` (desktop Ampere, e.g. RTX 2050). Override for other GPUs, e.g. `87` for Jetson Orin (AGX Orin/Orin NX/Orin Nano):
    ```sh
    colcon build --packages-select mppi_controller_cuda --parallel-workers 2 --cmake-args -DCMAKE_CUDA_ARCHITECTURES=<your-arch>
    ```

# Usage

Set the plugin type under your `controller_server`'s `FollowPath` in your Nav2 params yaml:

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "mppi_controller_cuda::MPPIControllerCUDA"
      # ... see params/mppi_controller_params.yaml for every parameter + its default
```

See [`params/mppi_controller_params.yaml`](params/mppi_controller_params.yaml) for the full list with defaults.
Example with turtlebot3 burger: [turtlebot3](https://github.com/datledoan/turtlebot3.git)

# Result
## Demo
![](media/mppi_demo.gif)
## Benchmark

Real-run `controller_server` CPU usage while navigating a goal in Gazebo
(turtlebot3 burger, see [turtlebot3](https://github.com/datledoan/turtlebot3.git)),
sampled with [`benchmark/run_live_cpu_test.sh`](benchmark/run_live_cpu_test.sh)
(10s window, 0.2s interval, 5 fresh-relaunch runs x 50 samples each):

| Controller | CPU usage (of 1 core, mean ± std across 5 runs) |
|---|---|
| nav2_mppi_controller (CPU) | 30.9% ± 0.9 |
| mppi_controller_cuda (GPU) | 9.4% ± 0.5 |

~69% less `controller_server` CPU load with the GPU plugin (RTX 2050).

Same setup on a **Jetson AGX Orin Developer Kit** (JetPack 6.2.3, Ubuntu 22.04.5,
CUDA 12.6, built with `-DCMAKE_CUDA_ARCHITECTURES=87` for Orin's compute
capability 8.7) -- Gazebo running on a separate host PC, 
`controller_server` on the Jetson over the network:

| Controller | CPU usage (of 1 core, mean ± std across 5 runs) |
|---|---|
| nav2_mppi_controller (CPU) | 65.4% ± 0.8 |
| mppi_controller_cuda (GPU) | 18.6% ± 0.1 |

~72% less `controller_server` CPU load with the GPU plugin (Jetson AGX Orin).

Per-call compute time (`Optimizer::evalControl()`/`Controller::computeControl()` -- see [`benchmark/`](benchmark/)), swept across `batch_size`/`time_steps`:

![](media/benchmark_sweep.png)

# Reference
- [nav2_mppi_controller](https://github.com/ros-navigation/navigation2/tree/main/nav2_mppi_controller)
- [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic)
```
@misc{vlahov2024mppi,
      title={MPPI-Generic: A CUDA Library for Stochastic Trajectory Optimization},
      author={Bogdan Vlahov and Jason Gibson and Manan Gandhi and Evangelos A. Theodorou},
      year={2024},
      eprint={2409.07563},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2409.07563},
}
```
