# Model Predictive Path Integral Controller

CUDA version of [mppi_controller_ros](https://github.com/datledoan/mppi_controller_ros), a ROS1 port of [nav2_mppi_controller](https://github.com/ros-navigation/navigation2/tree/main/nav2_mppi_controller), based on the [MPPI-Generic](https://github.com/ACDSLab/MPPI-Generic). Applicable for both [move_base](http://wiki.ros.org/move_base) and [move_base_flex](http://wiki.ros.org/move_base_flex).

# Installation

Requires NVIDIA GPU + CUDA toolkit. Tested with ROS Noetic, Ubuntu 20.04, CUDA 12.4, an RTX 2050 (compute capability 8.6).

* Install CUDA toolkit (see [NVIDIA's install guide](https://developer.nvidia.com/cuda-downloads)) and move_base_flex
    ```sh
    sudo apt install ros-noetic-mbf-costmap-nav
    ```
* Clone code and build
    ```sh
    cd your_ws/src
    git clone https://github.com/datledoan/mppi_controller_cuda.git
    catkin build mppi_controller_cuda
    ```

# Result
## Demo
![](media/mppi_demo.gif)
## Benchmark
Benchmark with CPU version [mppi_controller_ros](https://github.com/datledoan/mppi_controller_ros).
![](media/benchmark.png)

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
