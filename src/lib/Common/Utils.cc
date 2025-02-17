#include "Common/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "Common/cJSON.h"
#include <fstream>

namespace KernelCodeGen {


namespace tools {

char* _cjson_getString(cJSON* json, const std::string& key){
  auto value = cJSON_GetObjectItem(json, key.data());
  if(cJSON_IsString(value) && value->valuestring != nullptr){
    return value->valuestring;
  }
  return nullptr;
}

bool _parseKeyInt(cJSON* json, Config& cfg, const std::string& key){
  auto value = cJSON_GetObjectItem(json, key.data());
  if(cJSON_IsNumber(value)){
    cfg.at(key) = value->valueint;
    return true;
  }
  return false;
}

Config _parseCfgItem(cJSON* item){
  Config ret = {
      {KEY_BLOCK_SIZE_M, 0}, {KEY_BLOCK_SIZE_N, 0}, {KEY_BLOCK_SIZE_K, 0}, 
      {KEY_THREAD_SIZE_M, 0}, {KEY_THREAD_SIZE_N, 0}, 
      {KEY_WARP_SIZE, 0}, 
      {KEY_BLOCK_LAYOUT_M, 0}, {KEY_BLOCK_LAYOUT_N, 0}, 
      {KEY_WARP_LAYOUT_M, 0}, {KEY_WARP_LAYOUT_N, 0},
      {KEY_DTYPE_A, (int)0},{KEY_DTYPE_B, (int)0},{KEY_DTYPE_C, (int)0},
      {KEY_M, (int)0},{KEY_N, (int)0},{KEY_K, (int)0},
      {KEY_IS_A_TRANSPOSE,(int)0},
      {KEY_GLOB_LOAD_WIDTH_A,(int)0},
      {KEY_GLOB_LOAD_WIDTH_B,(int)0},
      {KEY_WARP_SCATTER_WIDTH_A,(int)0},
      {KEY_WARP_SCATTER_WIDTH_B,(int)0},
      {KEY_THREAD_SCATTER_WIDTH_A,(int)0},
      {KEY_THREAD_SCATTER_WIDTH_B,(int)0},
      {KEY_LOCAL_SPLIT_U,(int)0},
      {KEY_BLOCK_MAPPING,(int)0},
      {KEY_GLOB_STORE_WIDTH,(int)0}
    };

  assert(_parseKeyInt(item,ret,KEY_BLOCK_SIZE_M)) ;
  assert(_parseKeyInt(item,ret,KEY_BLOCK_SIZE_N)) ;
  assert(_parseKeyInt(item,ret,KEY_BLOCK_SIZE_K)) ;
  assert(_parseKeyInt(item,ret,KEY_THREAD_SIZE_M)) ;
  assert(_parseKeyInt(item,ret,KEY_THREAD_SIZE_N)) ;
  assert(_parseKeyInt(item,ret,KEY_WARP_SIZE)) ;
  assert(_parseKeyInt(item,ret,KEY_BLOCK_LAYOUT_M)) ;
  assert(_parseKeyInt(item,ret,KEY_BLOCK_LAYOUT_N)) ;
  assert(_parseKeyInt(item,ret,KEY_WARP_LAYOUT_M)) ;
  assert(_parseKeyInt(item,ret,KEY_WARP_LAYOUT_N)) ;
  assert(_parseKeyInt(item,ret,KEY_DTYPE_A)) ;
  assert(_parseKeyInt(item,ret,KEY_DTYPE_B)) ;
  assert(_parseKeyInt(item,ret,KEY_DTYPE_C)) ;
  assert(_parseKeyInt(item,ret,KEY_M)) ;
  assert(_parseKeyInt(item,ret,KEY_N)) ;
  assert(_parseKeyInt(item,ret,KEY_K)) ;
  assert(_parseKeyInt(item,ret,KEY_IS_A_TRANSPOSE)) ;
  assert(_parseKeyInt(item,ret,KEY_GLOB_LOAD_WIDTH_A)) ;
  assert(_parseKeyInt(item,ret,KEY_GLOB_LOAD_WIDTH_B)) ;
  assert(_parseKeyInt(item,ret,KEY_WARP_SCATTER_WIDTH_A)) ;
  assert(_parseKeyInt(item,ret,KEY_WARP_SCATTER_WIDTH_B)) ;
  assert(_parseKeyInt(item,ret,KEY_THREAD_SCATTER_WIDTH_A)) ;
  assert(_parseKeyInt(item,ret,KEY_THREAD_SCATTER_WIDTH_B)) ;
  assert(_parseKeyInt(item,ret,KEY_LOCAL_SPLIT_U)) ;
  assert(_parseKeyInt(item,ret,KEY_BLOCK_MAPPING)) ;
  assert(_parseKeyInt(item,ret,KEY_GLOB_STORE_WIDTH)) ;
  return ret;
}

bool parseJsonToConfigs(std::string filename, std::vector<Config>& res){
    std::ifstream file(filename); // 打开 JSON 文件
    if (!file.is_open()) {
        std::cout << "open jsonfile ["<<filename <<"] error" << std::endl;
        return false;
    }

    // 获取文件内容的长度
    file.seekg(0, std::ios::end);
    size_t length = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取文件内容到字符串中
    std::string content(length,' ');
    file.read(&content[0], length);
    // 关闭文件
    file.close();
    std::cout << content << std::endl;

    
    // 解析 JSON 文件内容
    cJSON *json = cJSON_Parse(content.data());
    if (!json) {
        fprintf(stderr, "Error parsing JSON.\n");
        return 1;
    }

    // 读取 JSON 中的字段
    cJSON *cfgs = cJSON_GetObjectItem(json, "cfgs");
    if(cJSON_IsArray(cfgs)){
      auto len = cJSON_GetArraySize(cfgs);
      for(int i=0;i<len;++i){
        auto config = cJSON_GetArrayItem(cfgs,i);
        if(config != nullptr){
          auto cfgmap = _parseCfgItem(config);
          res.push_back(cfgmap);
        }
      }
    }

    // 释放 cJSON 对象
    cJSON_Delete(json);
    return true;
}


mlir::Type getDType(mlir::OpBuilder& builder, const std::string& dtype) {
  if(dtype == "float32") return builder.getF32Type();
  if(dtype == "float64") return builder.getF64Type();
  if(dtype == "float16") return builder.getF16Type();
  if(dtype == "int64") return builder.getIntegerType(64);
  if(dtype == "int32") return builder.getIntegerType(32);
  if(dtype == "int16") return builder.getIntegerType(16);
  if(dtype == "index") return builder.getIndexType();
  if(dtype == "bool") return builder.getIntegerType(1);
  assert(false && "getDType:: Unsupported Type!");
  return nullptr;
}

std::string getenv(const char *name) {
    const char *cstr = std::getenv(name);
    if (!cstr)
        return "";
    std::string result(cstr);
    return result;
}


std::string KcgDtypeToStr(KcgDtype type){
  switch (type){
    case KcgDtype::float8   : return "";break;
    case KcgDtype::float16  : return "float16";break;
    case KcgDtype::float32  : return "float32";break;
    case KcgDtype::float64  : return "float64";break;
    case KcgDtype::float128 : return "";break;
    case KcgDtype::int8     : return "";break;
    case KcgDtype::int16    : return "int16";break;
    case KcgDtype::int32    : return "int32";break;
    case KcgDtype::int64    : return "int64";break;
  default:
    assert(false && "KcgDtypeToStr::invalid type!");
    break;
  }
  return "";
}

std::string typeToStr(mlir::Type type) {
  if(type.isa<mlir::Float16Type>()) return {"float16"};
  if(type.isa<mlir::Float32Type>()) return {"float32"};
  if(type.isa<mlir::Float64Type>()) return {"float64"};
  if(auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    if (int_type.getWidth() == 1) return {"bool"};
    else if (int_type.getWidth() == 16) return {"int16"};
    else if (int_type.getWidth() == 32) return {"int32"};
    else if (int_type.getWidth() == 64) return {"int64"};
  }
  if(type.isa<mlir::IndexType>()) return {"index"};
  assert(false && "not supported type!");
  return "";
}

void opSetAttr(mlir::Operation* op, const std::string& name, const std::string& val){
  assert(op != nullptr && "opSetAttr::nullptr error!");
  mlir::OpBuilder b(op->getContext());
  op->setAttr(name,b.getStringAttr(val)); 
}

void opSetAttr(mlir::Operation* op, const std::string& name, int val){
  assert(op != nullptr && "opSetAttr::nullptr error!");
  mlir::OpBuilder b(op->getContext());
  op->setAttr(name,b.getIntegerAttr(b.getI32Type(),val)); 
}

uint64_t getIntAttr(mlir::Operation* op, const std::string& name){
  assert(op != nullptr && "getAttr:nullptr error!");
  uint64_t ret = -1;
  if(op->hasAttr(name)){
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(op->getAttr(name));
    ret = attr.getValue().getLimitedValue();
  }
  return ret;
}

std::vector<int> getIntArrayAttr(mlir::Operation* op, const std::string& name){
  assert(op != nullptr && "getAttr:nullptr error!");
  std::vector<int> ret ;
  if(op->hasAttr(name)){
    auto attr = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(op->getAttr(name));
    for(const auto& e : attr.asArrayRef()) {
      ret.push_back(e);
    }
  }
  return ret;
}

bool isOpAttrEqualToString(mlir::Operation* op, const std::string& name, const std::string& expectedvalue){
  assert(op != nullptr && "isOpAttrEqualToString::nullptr error!");
  if(op->hasAttr(name)){
    auto attr = mlir::dyn_cast<mlir::StringAttr>(op->getAttr(name));
    if(attr.getValue() == expectedvalue){
      return true;
    }
  }
  return false;
}

std::string KcgKernelTypeToString(KcgKernelType type){
  switch (type)
  {
  case KcgKernelType::matmul:
    return "matmul"; break;
  case KcgKernelType::conv2d:
    return "conv2d"; break;
  case KcgKernelType::poolmax:
    return "poolmax"; break;
  default:
    assert(false && "KcgKernelTypeToString::invalid type!");break;
  }
  return "Unknown";
}

void _opSetDescription(mlir::Operation* op, const std::string& attrValue){
  mlir::OpBuilder b(op->getContext());
  op->setAttr(AttrDescription, b.getStringAttr(attrValue));
}

namespace mapUtils {
  
mlir::AffineExpr waprId(mlir::AffineExpr tid, const std::map<std::string, int>& config){
  return tid.floorDiv(config.at(KEY_WARP_SIZE));
}

mlir::AffineExpr wapr_x(mlir::AffineExpr tid, const std::map<std::string, int>& config){
  return waprId(tid,config) % config.at(KEY_BLOCK_LAYOUT_N);
}

mlir::AffineExpr wapr_y(mlir::AffineExpr tid, const std::map<std::string, int>& config){
  return waprId(tid,config).floorDiv(config.at(KEY_BLOCK_LAYOUT_N));
}

mlir::AffineExpr laneId(mlir::AffineExpr tid, const std::map<std::string, int>& config){
  return tid % (config.at(KEY_WARP_SIZE));
}

mlir::AffineExpr lane_x(mlir::AffineExpr tid, const std::map<std::string, int>& config){
  return laneId(tid,config) % config.at(KEY_WARP_LAYOUT_N);
}

mlir::AffineExpr lane_y(mlir::AffineExpr tid, const std::map<std::string, int>& config){
  return laneId(tid,config).floorDiv(config.at(KEY_WARP_LAYOUT_N));
}

mlir::AffineExpr bid_y(mlir::AffineExpr bid, const std::map<std::string, int>& config){
  return bid.floorDiv(config.at(KEY_N) / config.at(KEY_BLOCK_SIZE_N));
}

mlir::AffineExpr bid_x(mlir::AffineExpr bid, const std::map<std::string, int>& config){
  return bid % (config.at(KEY_N) / config.at(KEY_BLOCK_SIZE_N));
}

mlir::AffineExpr bid(mlir::AffineExpr bx,mlir::AffineExpr by, const std::map<std::string, int>& config){
  return bx + by * (config.at(KEY_N) / config.at(KEY_BLOCK_SIZE_N));
}

mlir::AffineExpr tid(mlir::AffineExpr tx,mlir::AffineExpr ty, const std::map<std::string, int>& config){
  return tx + ty * (config.at(KEY_BLOCK_SIZE_N) / config.at(KEY_THREAD_SIZE_N));
}

llvm::SmallVector<mlir::AffineExpr> reshapeBlock(mlir::AffineExpr tid, const std::vector<int> shape) {
  llvm::SmallVector<mlir::AffineExpr> exprs;
  for (int i=0; i<shape.size(); i++) {
    int64_t stride = 1;
    for (int j=i+1; j<shape.size(); j++ ) { stride *= shape[j]; }
    exprs.push_back(tid.floorDiv(stride));
    tid = tid % stride;
  }
  return exprs;
}

}  // mapUtils


}  // namespace tools
}  // namespace KernelCodeGen