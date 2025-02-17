#! /bin/bash
is_as_pymodule='ON'
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule
make -j8
