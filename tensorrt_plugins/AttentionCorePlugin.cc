#include "AttentionCorePlugin.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <new>
#include <utility>

namespace deepgen {
namespace tensorrt_plugin {

namespace {

constexpr const char* kPluginType = "DeepGenAttentionCorePlugin";
constexpr const char* kPluginVersion = "1";
constexpr const char* kDefaultNamespace = "";
constexpr size_t kWorkspaceAlignment = 256;
constexpr int kMaxKernels = 3;

bool LogCudaError(CUresult code, const char* message) {
  if (code == CUDA_SUCCESS) {
    return true;
  }
  const char* err = "unknown";
  cuGetErrorString(code, &err);
  std::cerr << "[DeepGenTRTPlugin] " << message << ": " << err << std::endl;
  return false;
}

bool LogCudaRuntimeError(cudaError_t code, const char* message) {
  if (code == cudaSuccess) {
    return true;
  }
  std::cerr << "[DeepGenTRTPlugin] " << message << ": " << cudaGetErrorString(code) << std::endl;
  return false;
}

size_t AlignSize(size_t value, size_t alignment = kWorkspaceAlignment) {
  return (value + alignment - 1) / alignment * alignment;
}

char* AlignPtr(char* ptr, size_t alignment = kWorkspaceAlignment) {
  auto value = reinterpret_cast<uintptr_t>(ptr);
  value = (value + alignment - 1) / alignment * alignment;
  return reinterpret_cast<char*>(value);
}

size_t ElementSize(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    default:
      return 0;
  }
}

template <typename T>
void WriteValue(char*& cursor, const T& value) {
  std::memcpy(cursor, &value, sizeof(T));
  cursor += sizeof(T);
}

template <typename T>
T ReadValue(const char*& cursor) {
  T value{};
  std::memcpy(&value, cursor, sizeof(T));
  cursor += sizeof(T);
  return value;
}

void WriteString(char*& cursor, const std::string& value) {
  const auto size = static_cast<uint64_t>(value.size());
  WriteValue(cursor, size);
  if (!value.empty()) {
    std::memcpy(cursor, value.data(), value.size());
    cursor += value.size();
  }
}

std::string ReadString(const char*& cursor) {
  const auto size = ReadValue<uint64_t>(cursor);
  std::string value(cursor, cursor + size);
  cursor += size;
  return value;
}

size_t SerializedSizeOfString(const std::string& value) {
  return sizeof(uint64_t) + value.size();
}

const nvinfer1::PluginField* FindField(
    const nvinfer1::PluginFieldCollection* collection,
    const char* name) {
  if (collection == nullptr || collection->fields == nullptr) {
    return nullptr;
  }
  for (int i = 0; i < collection->nbFields; ++i) {
    const auto& field = collection->fields[i];
    if (field.name != nullptr && std::strcmp(field.name, name) == 0) {
      return &field;
    }
  }
  return nullptr;
}

std::string ReadStringField(
    const nvinfer1::PluginFieldCollection* collection,
    const char* name,
    const std::string& defaultValue = "") {
  const auto* field = FindField(collection, name);
  if (field == nullptr || field->data == nullptr || field->length <= 0) {
    return defaultValue;
  }
  const auto* chars = static_cast<const char*>(field->data);
  return std::string(chars, chars + field->length);
}

int64_t ReadInt64Field(
    const nvinfer1::PluginFieldCollection* collection,
    const char* name,
    int64_t defaultValue = 0) {
  const auto* field = FindField(collection, name);
  if (field == nullptr || field->data == nullptr) {
    return defaultValue;
  }
  switch (field->type) {
    case nvinfer1::PluginFieldType::kINT64:
      return *static_cast<const int64_t*>(field->data);
    case nvinfer1::PluginFieldType::kINT32:
      return *static_cast<const int32_t*>(field->data);
    case nvinfer1::PluginFieldType::kINT16:
      return *static_cast<const int16_t*>(field->data);
    case nvinfer1::PluginFieldType::kINT8:
      return *static_cast<const int8_t*>(field->data);
    default:
      return defaultValue;
  }
}

std::vector<std::string>& FieldNameStorage() {
  static std::vector<std::string> names;
  if (!names.empty()) {
    return names;
  }

  names = {
      "op_name",
      "batch_size",
      "num_heads",
      "seq_len",
      "head_dim",
      "num_kernels",
  };
  for (int i = 1; i <= kMaxKernels; ++i) {
    const auto prefix = "k" + std::to_string(i);
    names.push_back(prefix + "_binary_path");
    names.push_back(prefix + "_kernel_name");
    names.push_back(prefix + "_grid_x");
    names.push_back(prefix + "_grid_y");
    names.push_back(prefix + "_grid_z");
    names.push_back(prefix + "_block_x");
    names.push_back(prefix + "_block_y");
    names.push_back(prefix + "_block_z");
    names.push_back(prefix + "_shm_bytes");
  }
  return names;
}

std::vector<nvinfer1::PluginField> BuildPluginFields() {
  std::vector<nvinfer1::PluginField> fields;
  auto& names = FieldNameStorage();
  auto addString = [&fields](const char* name) {
    fields.emplace_back(nvinfer1::PluginField{name, nullptr, nvinfer1::PluginFieldType::kCHAR, 1});
  };
  auto addInt = [&fields](const char* name) {
    fields.emplace_back(nvinfer1::PluginField{name, nullptr, nvinfer1::PluginFieldType::kINT64, 1});
  };

  int index = 0;
  addString(names[index++].c_str());
  addInt(names[index++].c_str());
  addInt(names[index++].c_str());
  addInt(names[index++].c_str());
  addInt(names[index++].c_str());
  addInt(names[index++].c_str());
  for (int i = 0; i < kMaxKernels; ++i) {
    addString(names[index++].c_str());
    addString(names[index++].c_str());
    addInt(names[index++].c_str());
    addInt(names[index++].c_str());
    addInt(names[index++].c_str());
    addInt(names[index++].c_str());
    addInt(names[index++].c_str());
    addInt(names[index++].c_str());
    addInt(names[index++].c_str());
  }
  return fields;
}

KernelLaunchSpec ReadKernelSpec(const nvinfer1::PluginFieldCollection* collection, int index) {
  KernelLaunchSpec spec;
  const auto prefix = "k" + std::to_string(index);
  spec.binaryPath = ReadStringField(collection, (prefix + "_binary_path").c_str());
  spec.kernelName = ReadStringField(collection, (prefix + "_kernel_name").c_str());
  spec.grid = {
      ReadInt64Field(collection, (prefix + "_grid_x").c_str()),
      ReadInt64Field(collection, (prefix + "_grid_y").c_str()),
      ReadInt64Field(collection, (prefix + "_grid_z").c_str())};
  spec.block = {
      ReadInt64Field(collection, (prefix + "_block_x").c_str()),
      ReadInt64Field(collection, (prefix + "_block_y").c_str()),
      ReadInt64Field(collection, (prefix + "_block_z").c_str())};
  spec.shmBytes = ReadInt64Field(collection, (prefix + "_shm_bytes").c_str());
  return spec;
}

AttentionPluginSpec SpecFromFields(const nvinfer1::PluginFieldCollection* collection) {
  AttentionPluginSpec spec;
  spec.opName = ReadStringField(collection, "op_name");
  spec.batchSize = ReadInt64Field(collection, "batch_size");
  spec.numHeads = ReadInt64Field(collection, "num_heads");
  spec.seqLen = ReadInt64Field(collection, "seq_len");
  spec.headDim = ReadInt64Field(collection, "head_dim");
  spec.numKernels = ReadInt64Field(collection, "num_kernels");
  for (int i = 0; i < kMaxKernels; ++i) {
    spec.kernels[i] = ReadKernelSpec(collection, i + 1);
  }
  return spec;
}

size_t SerializedSizeOfSpec(const AttentionPluginSpec& spec) {
  size_t size = 0;
  size += SerializedSizeOfString(spec.opName);
  size += sizeof(spec.batchSize) * 5;
  for (int i = 0; i < kMaxKernels; ++i) {
    size += SerializedSizeOfString(spec.kernels[i].binaryPath);
    size += SerializedSizeOfString(spec.kernels[i].kernelName);
    size += sizeof(int64_t) * 7;
  }
  return size;
}

void SerializeSpec(const AttentionPluginSpec& spec, void* buffer) {
  char* cursor = static_cast<char*>(buffer);
  WriteString(cursor, spec.opName);
  WriteValue(cursor, spec.batchSize);
  WriteValue(cursor, spec.numHeads);
  WriteValue(cursor, spec.seqLen);
  WriteValue(cursor, spec.headDim);
  WriteValue(cursor, spec.numKernels);
  for (int i = 0; i < kMaxKernels; ++i) {
    const auto& kernel = spec.kernels[i];
    WriteString(cursor, kernel.binaryPath);
    WriteString(cursor, kernel.kernelName);
    for (auto value : kernel.grid) {
      WriteValue(cursor, value);
    }
    for (auto value : kernel.block) {
      WriteValue(cursor, value);
    }
    WriteValue(cursor, kernel.shmBytes);
  }
}

}  // namespace

nvinfer1::PluginFieldCollection DeepGenAttentionCorePluginCreator::collection_{};
std::vector<nvinfer1::PluginField> DeepGenAttentionCorePluginCreator::fields_{};

REGISTER_TENSORRT_PLUGIN(DeepGenAttentionCorePluginCreator);

DeepGenAttentionCorePlugin::DeepGenAttentionCorePlugin(
    const std::string& name,
    const AttentionPluginSpec& spec)
    : name_(name), nameSpace_(kDefaultNamespace), spec_(spec) {}

DeepGenAttentionCorePlugin::DeepGenAttentionCorePlugin(
    const std::string& name,
    const void* data,
    size_t length)
    : name_(name), nameSpace_(kDefaultNamespace), spec_(deserializeSpec(data, length)) {}

DeepGenAttentionCorePlugin::~DeepGenAttentionCorePlugin() {
  releaseKernels();
}

size_t DeepGenAttentionCorePlugin::getSerializationSize() const noexcept {
  return SerializedSizeOfSpec(spec_);
}

void DeepGenAttentionCorePlugin::serialize(void* buffer) const noexcept {
  SerializeSpec(spec_, buffer);
}

const char* DeepGenAttentionCorePlugin::getPluginType() const noexcept {
  return kPluginType;
}

const char* DeepGenAttentionCorePlugin::getPluginVersion() const noexcept {
  return kPluginVersion;
}

int DeepGenAttentionCorePlugin::getNbOutputs() const noexcept {
  return 1;
}

int DeepGenAttentionCorePlugin::initialize() noexcept {
  return 0;
}

void DeepGenAttentionCorePlugin::terminate() noexcept {
  releaseKernels();
}

void DeepGenAttentionCorePlugin::destroy() noexcept {
  delete this;
}

nvinfer1::IPluginV2DynamicExt* DeepGenAttentionCorePlugin::clone() const noexcept {
  auto* plugin = new (std::nothrow) DeepGenAttentionCorePlugin(name_, spec_);
  if (plugin != nullptr) {
    plugin->setPluginNamespace(nameSpace_.c_str());
  }
  return plugin;
}

void DeepGenAttentionCorePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  nameSpace_ = pluginNamespace == nullptr ? "" : pluginNamespace;
}

const char* DeepGenAttentionCorePlugin::getPluginNamespace() const noexcept {
  return nameSpace_.c_str();
}

nvinfer1::DataType DeepGenAttentionCorePlugin::getOutputDataType(
    int index,
    const nvinfer1::DataType* inputTypes,
    int nbInputs) const noexcept {
  return (index == 0 && nbInputs > 0) ? inputTypes[0] : nvinfer1::DataType::kFLOAT;
}

nvinfer1::DimsExprs DeepGenAttentionCorePlugin::getOutputDimensions(
    int outputIndex,
    const nvinfer1::DimsExprs* inputs,
    int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) noexcept {
  (void) outputIndex;
  (void) exprBuilder;
  return nbInputs >= 3 ? inputs[2] : nvinfer1::DimsExprs{};
}

void DeepGenAttentionCorePlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs,
    int nbOutputs) noexcept {
  (void) outputs;
  if (nbInputs != 3 || nbOutputs != 1) {
    std::cerr << "[DeepGenTRTPlugin] configurePlugin expects 3 inputs and 1 output." << std::endl;
    return;
  }
  const auto& qDesc = inputs[0].desc;
  const auto& kDesc = inputs[1].desc;
  const auto& vDesc = inputs[2].desc;
  if (qDesc.dims.nbDims != 4 || kDesc.dims.nbDims != 4 || vDesc.dims.nbDims != 4) {
    std::cerr << "[DeepGenTRTPlugin] configurePlugin expects 4D tensors." << std::endl;
    return;
  }
}

bool DeepGenAttentionCorePlugin::supportsFormatCombination(
    int pos,
    const nvinfer1::PluginTensorDesc* inOut,
    int nbInputs,
    int nbOutputs) noexcept {
  if (nbInputs != 3 || nbOutputs != 1) {
    return false;
  }
  const auto& desc = inOut[pos];
  const bool isSupportedType =
      desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF;
  if (pos == 0) {
    return isSupportedType && desc.format == nvinfer1::TensorFormat::kLINEAR;
  }
  return desc.type == inOut[0].type && desc.format == inOut[0].format;
}

size_t DeepGenAttentionCorePlugin::workspaceSizeBytes(nvinfer1::DataType dtype) const noexcept {
  const auto typeWidth = ElementSize(dtype);
  if (typeWidth == 0) {
    return 0;
  }
  const auto rows = static_cast<size_t>(spec_.batchSize * spec_.numHeads * spec_.seqLen);
  size_t size = kWorkspaceAlignment;
  size += AlignSize(rows * typeWidth);
  size += AlignSize(rows * typeWidth);
  if (spec_.opName == "h2o") {
    size += AlignSize(rows * typeWidth);
  }
  return size;
}

size_t DeepGenAttentionCorePlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs,
    int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs,
    int nbOutputs) const noexcept {
  (void) outputs;
  if (nbInputs != 3 || nbOutputs != 1) {
    return 0;
  }
  return workspaceSizeBytes(inputs[0].type);
}

bool DeepGenAttentionCorePlugin::loadKernel(
    const KernelLaunchSpec& spec,
    LoadedKernel& loaded) noexcept {
  if (spec.binaryPath.empty() || spec.kernelName.empty()) {
    std::cerr << "[DeepGenTRTPlugin] Missing kernel path or kernel name." << std::endl;
    return false;
  }

  if (!LogCudaError(cuInit(0), "cuInit failed")) {
    return false;
  }

  int device = 0;
  if (!LogCudaRuntimeError(cudaGetDevice(&device), "cudaGetDevice failed")) {
    return false;
  }

  CUcontext context = nullptr;
  if (!LogCudaError(cuCtxGetCurrent(&context), "cuCtxGetCurrent failed")) {
    return false;
  }
  if (context == nullptr) {
    CUdevice cuDevice = 0;
    if (!LogCudaError(cuDeviceGet(&cuDevice, device), "cuDeviceGet failed")) {
      return false;
    }
    if (!LogCudaError(cuDevicePrimaryCtxRetain(&context, cuDevice), "cuDevicePrimaryCtxRetain failed")) {
      return false;
    }
    if (!LogCudaError(cuCtxSetCurrent(context), "cuCtxSetCurrent failed")) {
      return false;
    }
  }

  if (!LogCudaError(cuModuleLoad(&loaded.module, spec.binaryPath.c_str()), "cuModuleLoad failed")) {
    return false;
  }
  if (!LogCudaError(
          cuModuleGetFunction(&loaded.function, loaded.module, spec.kernelName.c_str()),
          "cuModuleGetFunction failed")) {
    cuModuleUnload(loaded.module);
    loaded.module = nullptr;
    return false;
  }

  const auto threads = spec.block[0] * spec.block[1] * spec.block[2];
  loaded.numWarps = std::max<int64_t>(1, threads / 32);
  loaded.loaded = true;

  int sharedOptin = 0;
  if (!LogCudaError(
          cuDeviceGetAttribute(
              &sharedOptin,
              CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
              device),
          "cuDeviceGetAttribute failed")) {
    return false;
  }
  if (spec.shmBytes > 49152 && sharedOptin > 49152) {
    LogCudaError(cuFuncSetCacheConfig(loaded.function, CU_FUNC_CACHE_PREFER_SHARED), "cuFuncSetCacheConfig failed");
    int sharedStatic = 0;
    if (LogCudaError(
            cuFuncGetAttribute(&sharedStatic, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, loaded.function),
            "cuFuncGetAttribute failed")) {
      LogCudaError(
          cuFuncSetAttribute(
              loaded.function,
              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
              std::max(0, sharedOptin - sharedStatic)),
          "cuFuncSetAttribute failed");
    }
  }
  return true;
}

bool DeepGenAttentionCorePlugin::ensureKernelsLoaded(cudaStream_t stream) noexcept {
  (void) stream;
  std::lock_guard<std::mutex> lock(loadMutex_);
  if (kernelsLoaded_) {
    return true;
  }
  for (int i = 0; i < std::min<int64_t>(spec_.numKernels, kMaxKernels); ++i) {
    if (!loadKernel(spec_.kernels[i], loadedKernels_[i])) {
      releaseKernels();
      return false;
    }
  }
  kernelsLoaded_ = true;
  return true;
}

bool DeepGenAttentionCorePlugin::launchKernel(
    const LoadedKernel& kernel,
    const KernelLaunchSpec& spec,
    const std::vector<uint64_t>& devicePtrs,
    cudaStream_t stream) const noexcept {
  std::vector<CUdeviceptr> paramsStorage(devicePtrs.size());
  std::vector<void*> params(devicePtrs.size());
  for (size_t i = 0; i < devicePtrs.size(); ++i) {
    paramsStorage[i] = static_cast<CUdeviceptr>(devicePtrs[i]);
    params[i] = &paramsStorage[i];
  }
  return LogCudaError(
      cuLaunchKernel(
          kernel.function,
          static_cast<unsigned int>(spec.grid[0]),
          static_cast<unsigned int>(spec.grid[1]),
          static_cast<unsigned int>(spec.grid[2]),
          static_cast<unsigned int>(kernel.numWarps * 32),
          1,
          1,
          static_cast<unsigned int>(spec.shmBytes),
          reinterpret_cast<CUstream>(stream),
          params.data(),
          nullptr),
      "cuLaunchKernel failed");
}

void DeepGenAttentionCorePlugin::releaseKernels() noexcept {
  for (auto& kernel : loadedKernels_) {
    if (kernel.loaded && kernel.module != nullptr) {
      cuModuleUnload(kernel.module);
    }
    kernel = LoadedKernel{};
  }
  kernelsLoaded_ = false;
}

AttentionPluginSpec DeepGenAttentionCorePlugin::deserializeSpec(const void* data, size_t length) const {
  (void) length;
  const char* cursor = static_cast<const char*>(data);
  AttentionPluginSpec spec;
  spec.opName = ReadString(cursor);
  spec.batchSize = ReadValue<int64_t>(cursor);
  spec.numHeads = ReadValue<int64_t>(cursor);
  spec.seqLen = ReadValue<int64_t>(cursor);
  spec.headDim = ReadValue<int64_t>(cursor);
  spec.numKernels = ReadValue<int64_t>(cursor);
  for (int i = 0; i < kMaxKernels; ++i) {
    auto& kernel = spec.kernels[i];
    kernel.binaryPath = ReadString(cursor);
    kernel.kernelName = ReadString(cursor);
    for (auto& value : kernel.grid) {
      value = ReadValue<int64_t>(cursor);
    }
    for (auto& value : kernel.block) {
      value = ReadValue<int64_t>(cursor);
    }
    kernel.shmBytes = ReadValue<int64_t>(cursor);
  }
  return spec;
}

int DeepGenAttentionCorePlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDescs,
    const nvinfer1::PluginTensorDesc* outputDescs,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  (void) outputDescs;
  if (!ensureKernelsLoaded(stream)) {
    return 1;
  }
  const auto elemSize = ElementSize(inputDescs[0].type);
  if (elemSize == 0) {
    std::cerr << "[DeepGenTRTPlugin] Unsupported input dtype." << std::endl;
    return 1;
  }

  char* cursor = AlignPtr(static_cast<char*>(workspace));
  const auto rows = static_cast<size_t>(spec_.batchSize * spec_.numHeads * spec_.seqLen);
  auto em = reinterpret_cast<uint64_t>(cursor);
  cursor += AlignSize(rows * elemSize);
  cursor = AlignPtr(cursor);
  auto denom = reinterpret_cast<uint64_t>(cursor);
  cursor += AlignSize(rows * elemSize);
  cursor = AlignPtr(cursor);
  auto rowSum = reinterpret_cast<uint64_t>(cursor);

  const auto query = reinterpret_cast<uint64_t>(inputs[0]);
  const auto key = reinterpret_cast<uint64_t>(inputs[1]);
  const auto value = reinterpret_cast<uint64_t>(inputs[2]);
  const auto output = reinterpret_cast<uint64_t>(outputs[0]);

  if (spec_.opName == "h2o") {
    if (!launchKernel(loadedKernels_[0], spec_.kernels[0], {query, key, em, denom}, stream)) {
      return 1;
    }
    if (!launchKernel(loadedKernels_[1], spec_.kernels[1], {key, query, em, denom, rowSum}, stream)) {
      return 1;
    }
    if (!launchKernel(
            loadedKernels_[2], spec_.kernels[2], {query, key, value, em, denom, output}, stream)) {
      return 1;
    }
    return 0;
  }

  if (!launchKernel(loadedKernels_[0], spec_.kernels[0], {query, key, em, denom}, stream)) {
    return 1;
  }
  if (!launchKernel(
          loadedKernels_[1], spec_.kernels[1], {query, key, value, em, denom, output}, stream)) {
    return 1;
  }
  return 0;
}

DeepGenAttentionCorePluginCreator::DeepGenAttentionCorePluginCreator() : nameSpace_(kDefaultNamespace) {
  if (fields_.empty()) {
    fields_ = BuildPluginFields();
    collection_.nbFields = static_cast<int>(fields_.size());
    collection_.fields = fields_.data();
  }
}

const char* DeepGenAttentionCorePluginCreator::getPluginName() const noexcept {
  return kPluginType;
}

const char* DeepGenAttentionCorePluginCreator::getPluginVersion() const noexcept {
  return kPluginVersion;
}

const char* DeepGenAttentionCorePluginCreator::getPluginNamespace() const noexcept {
  return nameSpace_.c_str();
}

const nvinfer1::PluginFieldCollection* DeepGenAttentionCorePluginCreator::getFieldNames() noexcept {
  return &collection_;
}

void DeepGenAttentionCorePluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
  nameSpace_ = pluginNamespace == nullptr ? "" : pluginNamespace;
}

nvinfer1::IPluginV2DynamicExt* DeepGenAttentionCorePluginCreator::createPlugin(
    const char* name,
    const nvinfer1::PluginFieldCollection* fieldCollection) noexcept {
  auto spec = SpecFromFields(fieldCollection);
  auto* plugin = new (std::nothrow) DeepGenAttentionCorePlugin(name == nullptr ? "" : name, spec);
  if (plugin != nullptr) {
    plugin->setPluginNamespace(nameSpace_.c_str());
  }
  return plugin;
}

nvinfer1::IPluginV2DynamicExt* DeepGenAttentionCorePluginCreator::deserializePlugin(
    const char* name,
    const void* serialData,
    size_t serialLength) noexcept {
  auto* plugin =
      new (std::nothrow) DeepGenAttentionCorePlugin(name == nullptr ? "" : name, serialData, serialLength);
  if (plugin != nullptr) {
    plugin->setPluginNamespace(nameSpace_.c_str());
  }
  return plugin;
}

}  // namespace tensorrt_plugin
}  // namespace deepgen
