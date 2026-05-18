
include_directories("${Python_INCLUDE_DIRS}")
include_directories(${pybind11_INCLUDE_DIRS})
if(COMPILE_AS_PYMODULE)
    add_definitions(-DCOMPILE_AS_PYMODULE)
    message("==== compile kcg_compiler as Lib ====")
    # link_directories("${PYTHON_CONDA_ENV_DIR}/lib")
    # add_library(${outName} SHARED main_.cc)
    add_library(${LIB_DEEPGEN} SHARED main.cc)
    target_link_libraries(${LIB_DEEPGEN} ${LIB_KCG_RUNTIME} ${Python_LIBRARIES} pthread)
    # pybind11_add_module(${outName} deepgen.cc)
    # target_link_libraries(${outName} PRIVATE kcg_runtime ${Python_LIBRARIES} pybind11::module pthread)
else()
    message(WARNING "==== [DebugOnly] compile kcg_compiler as Exe ====")
    # INVALID NOW
    add_executable(${LIB_DEEPGEN} test.cc)
    target_link_libraries(${LIB_DEEPGEN} ${LIB_KCG_RUNTIME} pthread)
endif()

add_executable(friskLib friskTest.cc)
target_link_libraries(friskLib PUBLIC 
    FriskIR
    ${LIB_KCG_RUNTIME}
)