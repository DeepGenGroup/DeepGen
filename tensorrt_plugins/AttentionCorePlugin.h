#pragma once

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace deepgen {
namespace tensorrt_plugin {

struct KernelLaunchSpec {
  std::string binaryPath;
  std::string kernelName;
  std::array<int64_t, 3> grid{0, 0, 0};
  std::array<int64_t, 3> block{0, 0, 0};
  int64_t shmBytes{0};
};

struct AttentionPluginSpec {
  std::string opName;
  int64_t batchSize{0};
  int64_t numHeads{0};
  int64_t seqLen{0};
  int64_t headDim{0};
  int64_t numKernels{0};
  std::array<KernelLaunchSpec, 3> kernels{};
};

class DeepGenAttentionCorePlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  DeepGenAttentionCorePlugin(const std::string& name, const AttentionPluginSpec& spec);
  DeepGenAttentionCorePlugin(const std::string& name, const void* data, size_t length);
  ~DeepGenAttentionCorePlugin() override;

  size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  void destroy() noexcept override;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

  nvinfer1::DataType getOutputDataType(
      int index,
      const nvinfer1::DataType* inputTypes,
      int nbInputs) const noexcept override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex,
      const nvinfer1::DimsExprs* inputs,
      int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) noexcept override;

  void configurePlugin(
      const nvinfer1::DynamicPluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::DynamicPluginTensorDesc* outputs,
      int nbOutputs) noexcept override;

  bool supportsFormatCombination(
      int pos,
      const nvinfer1::PluginTensorDesc* inOut,
      int nbInputs,
      int nbOutputs) noexcept override;

  size_t getWorkspaceSize(
      const nvinfer1::PluginTensorDesc* inputs,
      int nbInputs,
      const nvinfer1::PluginTensorDesc* outputs,
      int nbOutputs) const noexcept override;

  int enqueue(
      const nvinfer1::PluginTensorDesc* inputDescs,
      const nvinfer1::PluginTensorDesc* outputDescs,
      const void* const* inputs,
      void* const* outputs,
      void* workspace,
      cudaStream_t stream) noexcept override;

 private:
  struct LoadedKernel {
    CUmodule module{nullptr};
    CUfunction function{nullptr};
    int64_t numWarps{1};
    bool loaded{false};
  };

  bool ensureKernelsLoaded(cudaStream_t stream) noexcept;
  bool loadKernel(const KernelLaunchSpec& spec, LoadedKernel& loaded) noexcept;
  bool launchKernel(
      const LoadedKernel& kernel,
      const KernelLaunchSpec& spec,
      const std::vector<uint64_t>& devicePtrs,
      cudaStream_t stream) const noexcept;
  void releaseKernels() noexcept;
  size_t workspaceSizeBytes(nvinfer1::DataType dtype) const noexcept;
  AttentionPluginSpec deserializeSpec(const void* data, size_t length) const;

 private:
  std::string name_;
  std::string nameSpace_;
  AttentionPluginSpec spec_{};
  std::array<LoadedKernel, 3> loadedKernels_{};
  mutable std::mutex loadMutex_;
  bool kernelsLoaded_{false};
};

class DeepGenAttentionCorePluginCreator final : public nvinfer1::IPluginCreator {
 public:
  DeepGenAttentionCorePluginCreator();

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const char* getPluginNamespace() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(
      const char* name,
      const nvinfer1::PluginFieldCollection* fieldCollection) noexcept override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(
      const char* name,
      const void* serialData,
      size_t serialLength) noexcept override;

 private:
  static nvinfer1::PluginFieldCollection collection_;
  static std::vector<nvinfer1::PluginField> fields_;
  std::string nameSpace_;
};

}  // namespace tensorrt_plugin
}  // namespace deepgen
