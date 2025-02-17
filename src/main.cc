#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <string>
#include <vector>
#include "KernelCodeGen.h"
#include "Target/HSACOTranslation.h"
#include <stdio.h>
#include <stdlib.h>
#include "Common/Utils.h"
#include "Common/ThreadPool.h"
#include "Operators/Matmul.h"

#ifdef COMPILE_AS_PYMODULE
#include "Python.h"
#endif

#define DBG_USE_EXTERN_MLIR 0

using namespace KernelCodeGen;

std::string global_json_path{};


enum class MdlOperatorType : int{
  Matmul = 1,
  Convolution = 2,
  Pool = 3
};


class MatmulParams {
public:
  KcgDtype m_dtypeA, m_dtypeB, m_dtypeC; // 3
  int m_size,n_size,k_size;  // 3+3=6
  int m_isATranspose = 0;  
  int m_BLOCK_SIZE_M;
  int m_BLOCK_SIZE_N;
  int m_BLOCK_SIZE_K;
  int m_THREAD_SIZE_M;
  int m_THREAD_SIZE_N;
  int m_WARP_SIZE;
  int m_BLOCK_LAYOUT_M;
  int m_BLOCK_LAYOUT_N;
  int m_WARP_LAYOUT_M;
  int m_WARP_LAYOUT_N;  // 11+6=17
  // recently-added-params
  int m_GLOB_LOAD_WIDTH_A;
  int m_GLOB_LOAD_WIDTH_B;
  int m_WARP_SCATTER_WIDTH_A;
  int m_WARP_SCATTER_WIDTH_B;
  int m_THREAD_SCATTER_WIDTH_A;
  int m_THREAD_SCATTER_WIDTH_B;
  int m_LOCAL_SPLIT_U;
  int m_BLOCK_MAPPING;
  int m_GLOB_STORE_WIDTH;  // 17+9=26
  int m_UNROLL_NUM;
  int m_REG_PREFETCH;
  int m_SHARED_PREFETCH;
  int m_LOAD_CONTINUOUS;
  int m_REDUCE_C_CONTINUOUS;  // 31

#ifdef COMPILE_AS_PYMODULE
  // 此处应保证python传参的顺序和parse顺序相同
  bool parse(PyObject* args){
    if(PyArg_ParseTuple(args, std::string(31,'i').c_str(),
      &m_BLOCK_SIZE_M,
      &m_BLOCK_SIZE_N,
      &m_BLOCK_SIZE_K,
      &m_THREAD_SIZE_M,
      &m_THREAD_SIZE_N,
      &m_WARP_SIZE,
      &m_BLOCK_LAYOUT_M,
      &m_BLOCK_LAYOUT_N,
      &m_WARP_LAYOUT_M,
      &m_WARP_LAYOUT_N,
    // recent-added
      &m_GLOB_LOAD_WIDTH_A,
      &m_GLOB_LOAD_WIDTH_B,
      &m_WARP_SCATTER_WIDTH_A,
      &m_WARP_SCATTER_WIDTH_B,
      &m_THREAD_SCATTER_WIDTH_A,
      &m_THREAD_SCATTER_WIDTH_B,
      &m_LOCAL_SPLIT_U,
      &m_BLOCK_MAPPING,
      &m_GLOB_STORE_WIDTH,

      &m_UNROLL_NUM,
      &m_REG_PREFETCH,
      &m_SHARED_PREFETCH,
      &m_LOAD_CONTINUOUS,
      &m_REDUCE_C_CONTINUOUS, 

      &m_dtypeA, &m_dtypeB, &m_dtypeC,
      &m_size,&n_size,&k_size,
      &m_isATranspose

    )){
      return true;
    }
    assert(false && "PyArg_ParseTuple Error");
    return false;
  }
#endif

  template<typename T>
  void paramCombine(std::stringstream& ss, const char* title, T p1, T p2, T p3)  {
    ss << title << p1 << "x" << p2 << "x"<< p3 << "_";
  }

  template<typename T>
  void paramCombine(std::stringstream& ss, const char* title, T p1, T p2)  {
    ss << title << p1 << "x" << p2 << "_";
  }
  template<typename T>
  void paramCombine(std::stringstream& ss, const char* title, T p1)  {
    ss << title << p1 << "_";
  }

  std::string getKernelName()  {
    std::stringstream ss;
    ss << "GEMM_";
    paramCombine(ss, "MNK", m_size, n_size, k_size);
    paramCombine(ss, "DTabc", tools::KcgDtypeToStr(m_dtypeA),
            tools::KcgDtypeToStr(m_dtypeB),
            tools::KcgDtypeToStr(m_dtypeC));
    paramCombine(ss, "AT", m_isATranspose);
    paramCombine(ss, "TTmn", m_THREAD_SIZE_M, m_THREAD_SIZE_N);
    paramCombine(ss, "BTmnk", m_BLOCK_SIZE_M, m_BLOCK_SIZE_N, m_BLOCK_SIZE_K);
    paramCombine(ss, "BLmn", m_BLOCK_LAYOUT_M, m_BLOCK_LAYOUT_N);
    paramCombine(ss, "WLmn", m_WARP_LAYOUT_M,  m_WARP_LAYOUT_N);
    paramCombine(ss, "GLWab",m_GLOB_LOAD_WIDTH_A, m_GLOB_LOAD_WIDTH_B);
    paramCombine(ss, "GSW", m_GLOB_STORE_WIDTH);
    paramCombine(ss, "WSWab", m_WARP_SCATTER_WIDTH_A, m_WARP_SCATTER_WIDTH_B);
    paramCombine(ss, "TSWab", m_THREAD_SCATTER_WIDTH_A, m_THREAD_SCATTER_WIDTH_B);
    paramCombine(ss, "LSU", m_LOCAL_SPLIT_U);
    paramCombine(ss, "BM", m_BLOCK_MAPPING);

    paramCombine(ss,"UNROLL",m_UNROLL_NUM) ;
    paramCombine(ss,"REGP",m_REG_PREFETCH) ;
    paramCombine(ss,"SHMP",m_SHARED_PREFETCH) ;
    paramCombine(ss,"LC",m_LOAD_CONTINUOUS) ;
    paramCombine(ss,"RC",m_REDUCE_C_CONTINUOUS) ; 

    return ss.str();
  }

  Config asConfigMap(){
    Config ret = 
    {
      {KEY_BLOCK_SIZE_M , m_BLOCK_SIZE_M},
      {KEY_BLOCK_SIZE_N , m_BLOCK_SIZE_N},
      {KEY_BLOCK_SIZE_K , m_BLOCK_SIZE_K},
      {KEY_THREAD_SIZE_M , m_THREAD_SIZE_M},
      {KEY_THREAD_SIZE_N , m_THREAD_SIZE_N},
      {KEY_WARP_SIZE , m_WARP_SIZE},
      {KEY_BLOCK_LAYOUT_M , m_BLOCK_LAYOUT_M},
      {KEY_BLOCK_LAYOUT_N , m_BLOCK_LAYOUT_N},
      {KEY_WARP_LAYOUT_M , m_WARP_LAYOUT_M},
      {KEY_WARP_LAYOUT_N , m_WARP_LAYOUT_N},
      {KEY_DTYPE_A , int(m_dtypeA)},
      {KEY_DTYPE_B , int(m_dtypeB)},
      {KEY_DTYPE_C , int(m_dtypeC)},
      {KEY_M , m_size},
      {KEY_N , n_size},
      {KEY_K , k_size},
      {KEY_IS_A_TRANSPOSE , m_isATranspose},
      {KEY_GLOB_LOAD_WIDTH_A , m_GLOB_LOAD_WIDTH_A},
      {KEY_GLOB_LOAD_WIDTH_B , m_GLOB_LOAD_WIDTH_B},
      {KEY_WARP_SCATTER_WIDTH_A , m_WARP_SCATTER_WIDTH_A},
      {KEY_WARP_SCATTER_WIDTH_B , m_WARP_SCATTER_WIDTH_B},
      {KEY_THREAD_SCATTER_WIDTH_A , m_THREAD_SCATTER_WIDTH_A},
      {KEY_THREAD_SCATTER_WIDTH_B , m_THREAD_SCATTER_WIDTH_B},
      {KEY_LOCAL_SPLIT_U , m_LOCAL_SPLIT_U},
      {KEY_BLOCK_MAPPING , m_BLOCK_MAPPING},
      {KEY_GLOB_STORE_WIDTH , m_GLOB_STORE_WIDTH},

      {KEY_UNROLL_NUM, m_UNROLL_NUM},
      {KEY_REG_PREFETCH, m_REG_PREFETCH},
      {KEY_SHARED_PREFETCH, m_SHARED_PREFETCH},
      {KEY_LOAD_CONTINUOUS, m_LOAD_CONTINUOUS},
      {KEY_REDUCE_C_CONTINUOUS,  m_REDUCE_C_CONTINUOUS}, 
    };
    return ret;
  }

};

std::ostream& operator<<(std::ostream& os, MatmulParams arg){
  os << "== UserKernelCfg :\n";
  os << "- M : " << arg.m_size << std::endl;
  os << "- N : " << arg.n_size << std::endl;
  os << "- K : " << arg.k_size << std::endl;
  os << "- m_dtypeA : " << tools::KcgDtypeToStr(arg.m_dtypeA) << std::endl;
  os << "- m_dtypeB : " << tools::KcgDtypeToStr(arg.m_dtypeB) << std::endl;
  os << "- m_dtypeC : " << tools::KcgDtypeToStr(arg.m_dtypeC) << std::endl;
  os << "- m_isATranspose : " <<arg.m_isATranspose << std::endl;
  os << "- m_BLOCK_SIZE_M : " <<arg.m_BLOCK_SIZE_M << std::endl;
  os << "- m_BLOCK_SIZE_N : " <<arg.m_BLOCK_SIZE_N << std::endl;
  os << "- m_BLOCK_SIZE_K : " <<arg.m_BLOCK_SIZE_K << std::endl;
  os << "- m_THREAD_SIZE_M : " <<arg.m_THREAD_SIZE_M << std::endl;
  os << "- m_THREAD_SIZE_N : " <<arg.m_THREAD_SIZE_N << std::endl;
  os << "- m_WARP_SIZE : " <<arg.m_WARP_SIZE << std::endl;
  os << "- m_BLOCK_LAYOUT_M : " <<arg.m_BLOCK_LAYOUT_M << std::endl;
  os << "- m_BLOCK_LAYOUT_N : " <<arg.m_BLOCK_LAYOUT_N << std::endl;
  os << "- m_WARP_LAYOUT_M : " <<arg.m_WARP_LAYOUT_M << std::endl;
  os << "- m_WARP_LAYOUT_N : " <<arg.m_WARP_LAYOUT_N << std::endl;
  os << "- m_GLOB_LOAD_WIDTH_A : " << arg.m_GLOB_LOAD_WIDTH_A << std::endl;
  os << "- m_GLOB_LOAD_WIDTH_B : " << arg.m_GLOB_LOAD_WIDTH_B << std::endl;
  os << "- m_WARP_SCATTER_WIDTH_A : " << arg.m_WARP_SCATTER_WIDTH_A << std::endl;
  os << "- m_WARP_SCATTER_WIDTH_B : " << arg.m_WARP_SCATTER_WIDTH_B << std::endl;
  os << "- m_THREAD_SCATTER_WIDTH_A : " << arg.m_THREAD_SCATTER_WIDTH_A << std::endl;
  os << "- m_THREAD_SCATTER_WIDTH_B : " << arg.m_THREAD_SCATTER_WIDTH_B << std::endl;
  os << "- m_LOCAL_SPLIT_U : " << arg.m_LOCAL_SPLIT_U << std::endl;
  os << "- m_BLOCK_MAPPING : " << arg.m_BLOCK_MAPPING << std::endl;
  os << "- m_GLOB_STORE_WIDTH : " << arg.m_GLOB_STORE_WIDTH << std::endl;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Config& cfg){
  os << "C++ Config :" << std::endl;
  os << "- " << KEY_M << ":" << cfg.at(KEY_M) << std::endl;
  os << "- " << KEY_N << ":" << cfg.at(KEY_N) << std::endl;
  os << "- " << KEY_K << ":" << cfg.at(KEY_K) << std::endl;
  os << "- " << KEY_DTYPE_A << ":"<< cfg.at(KEY_DTYPE_A) << std::endl;
  os << "- " << KEY_DTYPE_B << ":"<< cfg.at(KEY_DTYPE_B) << std::endl;
  os << "- " << KEY_DTYPE_C << ":"<< cfg.at(KEY_DTYPE_C) << std::endl;
  os << "- " << KEY_IS_A_TRANSPOSE << ":"<< cfg.at(KEY_IS_A_TRANSPOSE) << std::endl;
  os << "- " << KEY_BLOCK_SIZE_M << ":"<< cfg.at(KEY_BLOCK_SIZE_M) << std::endl;
  os << "- " << KEY_BLOCK_SIZE_N << ":"<< cfg.at(KEY_BLOCK_SIZE_N) << std::endl;
  os << "- " << KEY_BLOCK_SIZE_K << ":"<< cfg.at(KEY_BLOCK_SIZE_K) << std::endl;
  os << "- " << KEY_THREAD_SIZE_M << ":"<< cfg.at(KEY_THREAD_SIZE_M) << std::endl;
  os << "- " << KEY_THREAD_SIZE_N << ":"<< cfg.at(KEY_THREAD_SIZE_N) << std::endl;
  os << "- " << KEY_WARP_SIZE << ":"<< cfg.at(KEY_WARP_SIZE) << std::endl;
  os << "- " << KEY_BLOCK_LAYOUT_M << ":"<< cfg.at(KEY_BLOCK_LAYOUT_M) << std::endl;
  os << "- " << KEY_BLOCK_LAYOUT_N << ":"<< cfg.at(KEY_BLOCK_LAYOUT_N) << std::endl;
  os << "- " << KEY_WARP_LAYOUT_M << ":"<< cfg.at(KEY_WARP_LAYOUT_M) << std::endl;
  os << "- " << KEY_WARP_LAYOUT_N << ":"<< cfg.at(KEY_WARP_LAYOUT_N) << std::endl;
  os << "- " << KEY_GLOB_LOAD_WIDTH_A << ":"<< cfg.at(KEY_GLOB_LOAD_WIDTH_A) << std::endl;
  os << "- " << KEY_GLOB_LOAD_WIDTH_B << ":"<< cfg.at(KEY_GLOB_LOAD_WIDTH_B) << std::endl;
  os << "- " << KEY_WARP_SCATTER_WIDTH_A << ":"<< cfg.at(KEY_WARP_SCATTER_WIDTH_A) << std::endl;
  os << "- " << KEY_WARP_SCATTER_WIDTH_B << ":"<< cfg.at(KEY_WARP_SCATTER_WIDTH_B) << std::endl;
  os << "- " << KEY_THREAD_SCATTER_WIDTH_A << ":"<< cfg.at(KEY_THREAD_SCATTER_WIDTH_A) << std::endl;
  os << "- " << KEY_THREAD_SCATTER_WIDTH_B << ":"<< cfg.at(KEY_THREAD_SCATTER_WIDTH_B) << std::endl;
  os << "- " << KEY_LOCAL_SPLIT_U << ":"<< cfg.at(KEY_LOCAL_SPLIT_U) << std::endl;
  os << "- " << KEY_BLOCK_MAPPING << ":"<< cfg.at(KEY_BLOCK_MAPPING) << std::endl;
  os << "- " << KEY_GLOB_STORE_WIDTH << ":"<< cfg.at(KEY_GLOB_STORE_WIDTH) << std::endl;
  return os;
}

std::vector<KernelInfo> generateKernels(
  std::vector<Config>& configs,
  const std::vector<std::string>& kernelNames) 
{
  assert(configs.size() == kernelNames.size() && "configs & kernelNames size match");
  std::vector<KernelInfo> result;
  // std::map<Config, KernelInfo> result;
  // ThreadPool pool(1);
  // pool.init();
  const int cfgszie = configs.size();
  mlir::registerAllPasses();
  for (int i=0;i< cfgszie;++i) {
    std::function<KernelInfo(Config&)> task = [i,&kernelNames](Config& cfg)->KernelInfo
    {
      // std::cout << cfg << std::endl;
      KernelCodeGenerator generator(Target::ROCm, "906");
      const auto config = cfg;
      const auto name = kernelNames[i];
      KernelInfo info;
      auto dtypeA = tools::KcgDtypeToStr((KcgDtype)config.at(KEY_DTYPE_A));
      auto dtypeB = tools::KcgDtypeToStr((KcgDtype)config.at(KEY_DTYPE_B));
      auto dtypeC = tools::KcgDtypeToStr((KcgDtype)config.at(KEY_DTYPE_C));
      auto M = config.at(KEY_M);
      auto N = config.at(KEY_N);
      auto K = config.at(KEY_K);
      bool isATranspose = config.at(KEY_IS_A_TRANSPOSE)> 0;
      auto kernel = generator.create<Operators::Matmul>(
        std::vector<int64_t>{M, N, K},
        std::vector<std::string>{dtypeA,dtypeB,dtypeC},
        name,isATranspose
      );

      auto res1 = generator.optimize(kernel, config);
      // std::cout << "==== optimize status: " << (res1?"SUCCESS":"FAILED") << "\n";
      auto res2 = generator.lowering(kernel);
      // std::cout << "==== lowering status: " << (res2?"SUCCESS":"FAILED") << "\n";
      std::string hsacoPath = generator.translate(kernel);
      // std::cout << "==== translate res :" << "\n";
      std::cout << hsacoPath << "\n";
      info.m_hsacoPath = hsacoPath;
      info.m_kernelName = generator.kernelFuncName<Operators::Matmul>();
      auto gridDim = tools::getIntArrayAttr(kernel,AttrGridDim);
      auto blockDim = tools::getIntArrayAttr(kernel,AttrBlockDim);
      info.m_gridDims = gridDim;
      info.m_blockDims = blockDim;
      // result[config] = info;
      std::cout << "==== kernel name : " << info.m_kernelName << "\n";
      return info;
    };  // end std::function<>
    result.push_back(task(configs[i]));
    // pool.push_task(std::move(task),configs[i]);
  }
  // pool.wait_finish(cfgszie);
  // return pool.get_result();
  return result;
}

std::vector< KernelInfo> _compile(MatmulParams& config) {
  std::vector<std::string> names = {config.getKernelName()};
  std::vector<Config> cfgs = {config.asConfigMap()};
  return generateKernels(cfgs,names);
}


#ifdef COMPILE_AS_PYMODULE
static PyObject* compile_kernel_matmul(PyObject* self, PyObject* args) {
  MatmulParams config;
  assert(config.parse(args));
  // std::cout << config << std::endl;
  std::vector<KernelInfo> kernels;
  std::string hsacoPath;
  Py_BEGIN_ALLOW_THREADS;
  // std::cout << "[pymod] start _compile" << std::endl;
  kernels = _compile(config);
  // std::cout << "[pymod] _compile success" << std::endl;
  Py_END_ALLOW_THREADS;
  Py_INCREF(Py_None);
  // return Py_None;
  PyObject *retArr;
  retArr = PyTuple_New(kernels.size());
    // 填充元组
  for (int i=0;i<kernels.size();++i) {
    // 假设元素数组元素是以字符串和整数对的方式存在
    // 这里你需要将每一对 (ss, i...) 插入
    const auto& kernel = kernels[i];
    std::vector<int> gridDims = {1,1,1};
    std::vector<int> blockDims = {1,1,1};
    for(int i=0;i<kernel.m_gridDims.size();++i){
      gridDims[i] = kernel.m_gridDims[i];
    }
    for(int i=0;i<kernel.m_blockDims.size();++i){
      blockDims[i] = kernel.m_blockDims[i];
    }
    PyObject* item = Py_BuildValue("(ssiiiiii)",
      kernel.m_hsacoPath.c_str(),
      kernel.m_kernelName.c_str(),
      gridDims[0],gridDims[1],gridDims[2],
      blockDims[0],blockDims[1],blockDims[2]
    );
    if (item == NULL) {
      Py_DECREF(retArr);
      return NULL;  // 如果构建某个元素失败，释放资源并返回NULL
    }
    PyTuple_SetItem(retArr, i, item);  // 将每个元素插入元组
  }
  // std::cout << "[pymod] ======== compile_kernel_matmul return " << std::endl;
  return retArr;
}



static PyMethodDef ModuleMethods[] = {
    {"compile_kernel_matmul", compile_kernel_matmul, METH_VARARGS,"compile kernel according to user input tiling and type"},
    {NULL, NULL, 0, NULL} // sentinel
};

static struct PyModuleDef ModuleDef = {PyModuleDef_HEAD_INIT, "KCGCompiler",
                                       NULL, // documentation
                                       -1,   // size
                                       ModuleMethods};

PyMODINIT_FUNC PyInit_KCGCompiler(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}

#else

int main(){

  // using namespace KernelCodeGen;
#if 0
  std::vector<Config> cfgs;
  global_json_path = "/home/xushilong/CodeGenDemo/cfg_cominations.json";
  MatmulParams config;
  config.m_BLOCK_SIZE_M = 32;
  config.m_BLOCK_SIZE_N = 32;
  config.m_BLOCK_SIZE_K = 16;
  config.m_THREAD_SIZE_M = 4;
  config.m_THREAD_SIZE_N = 4;
  config.m_GLOB_LOAD_WIDTH_A = 2;
  config.m_GLOB_LOAD_WIDTH_B = 2;
  config.m_BLOCK_LAYOUT_M = 2;
  config.m_BLOCK_LAYOUT_N = 2;
  config.m_WARP_LAYOUT_M = 8;
  config.m_WARP_LAYOUT_N = 8;
  config.m_WARP_SCATTER_WIDTH_A = 2;
  config.m_WARP_SCATTER_WIDTH_B = 2;
  config.m_THREAD_SCATTER_WIDTH_A = 2;
  config.m_THREAD_SCATTER_WIDTH_B = 2;
  config.m_LOCAL_SPLIT_U = 1;
  config.m_BLOCK_MAPPING = 8;
  config.m_WARP_SIZE = 64;
  config.m_GLOB_STORE_WIDTH = 2;
  config.m_dtypeA = KcgDtype::float32;
  config.m_dtypeB = KcgDtype::float32;
  config.m_size = 1024;
  config.n_size = 1024;
  config.k_size = 1024;
  config.m_isATranspose = true;
  config.m_dtypeC = KcgDtype::float32;

  auto ret = _compile(config);
#endif


  std::vector<Config> configs = {
    {
      {KEY_BLOCK_SIZE_M, 64}, {KEY_BLOCK_SIZE_N, 48}, {KEY_BLOCK_SIZE_K, 32}, {KEY_THREAD_SIZE_M, 4}, {KEY_THREAD_SIZE_N, 6}, 
      {KEY_GLOB_LOAD_WIDTH_A, 4}, {KEY_GLOB_LOAD_WIDTH_B, 2}, 
      {KEY_BLOCK_LAYOUT_M, 2}, {KEY_BLOCK_LAYOUT_N, 1}, {KEY_WARP_LAYOUT_M, 8}, {KEY_WARP_LAYOUT_N, 8},
      {KEY_WARP_SCATTER_WIDTH_A, 2}, {KEY_WARP_SCATTER_WIDTH_B, 2}, {KEY_THREAD_SCATTER_WIDTH_A, 2}, {KEY_THREAD_SCATTER_WIDTH_B, 2}, 
      {KEY_LOCAL_SPLIT_U, 2}, {KEY_BLOCK_MAPPING, 8}, {KEY_WARP_SIZE, 64}, {KEY_GLOB_STORE_WIDTH, 2}, 
      {KEY_UNROLL_NUM, 16}, {KEY_REG_PREFETCH, 1}, {KEY_SHARED_PREFETCH, 1}, {KEY_LOAD_CONTINUOUS, 1}, {KEY_REDUCE_C_CONTINUOUS, 1}, 
      {KEY_DTYPE_A, (int)KcgDtype::float32},
      {KEY_DTYPE_B, (int)KcgDtype::float32},
      {KEY_DTYPE_C, (int)KcgDtype::float32},
      {KEY_M, 1024},{KEY_N, 1056},{KEY_K, 1024}, 
      {KEY_IS_A_TRANSPOSE, 1}
    },
  };
  std::vector<std::string> names = {"GEMM_testKernel"};
  auto result = generateKernels(configs, names);

  return 0;
}


#endif

