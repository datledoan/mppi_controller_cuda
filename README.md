# Model Predictive Path Integral Controller

CUDA-accelerated MPPI controller based on [nav2_mppi_controller](https://github.com/ros-navigation/navigation2/tree/main/nav2_mppi_controller) built on the [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic) CUDA framework.

>For ROS1, see the [`noetic`](https://github.com/datledoan/mppi_controller_cuda/tree/noetic) branch.

# Installation

Requires an NVIDIA GPU + CUDA toolkit. Tested with ROS2 Jazzy, Ubuntu 24.04, CUDA 12.0, an RTX 2050 (compute capability 8.6).

* Install the CUDA toolkit (see [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)).
* Clone and build:
    ```sh
    cd your_ws/src
    git clone -b jazzy https://github.com/datledoan/mppi_controller_cuda.git
    cd your_ws
    colcon build --packages-select mppi_controller_cuda
    ```
  `CMAKE_CUDA_ARCHITECTURES` defaults to `86` (this project's own dev GPU). Override for other GPUs, e.g.:
    ```sh
    colcon build --packages-select mppi_controller_cuda --cmake-args -DCMAKE_CUDA_ARCHITECTURES=<your-arch>
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

# Result
![](media/mppi_demo.gif)

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
