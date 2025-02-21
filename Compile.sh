#! /bin/bash
is_as_pymodule='OFF'
cd build  
cmake .. -DCOMPILE_AS_PYMODULE=$is_as_pymodule
make -j8
