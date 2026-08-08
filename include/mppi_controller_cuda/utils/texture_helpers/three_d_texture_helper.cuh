#ifndef THREE_D_TEXTURE_HELPER_CUH_
#define THREE_D_TEXTURE_HELPER_CUH_

#include "mppi_controller_cuda/utils/texture_helpers/texture_helper.cuh"


template <class DATA_T>
class ThreeDTextureHelper : public TextureHelper<ThreeDTextureHelper<DATA_T>, DATA_T>
{
public:
  ThreeDTextureHelper(int number, bool synced = false, cudaStream_t stream = 0);

  void allocateCudaTexture(int index) override;

  void updateTexture(const int index, const int z_index, std::vector<DATA_T>& data, bool column_major = false);
  void updateTexture(const int index, const int z_index,
                     const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                                      Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
                         values,
                     bool column_major = true);
  bool setExtent(int index, cudaExtent& extent) override;
  void copyDataToGPU(int index, bool sync = false) override;

  std::vector<std::vector<bool>> getLayerCopy()
  {
    return layer_copy_;
  }

  __host__ __device__ DATA_T queryTexture(const int index, const float3& point);

protected:
  // cpu values
  // layer -> 3D array -> actual texture lookup thing
  std::vector<std::vector<bool>> layer_copy_;  // indicator what 2D part of the 3D array needs to be copied over

  // if we should require every depth to be updated before sending to GPU
  bool synched_ = false;
};

#endif  // THREE_D_TEXTURE_HELPER_CUH_
