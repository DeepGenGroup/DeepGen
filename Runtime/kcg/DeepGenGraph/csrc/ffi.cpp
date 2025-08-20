#include <pybind11/pybind11.h>
#include "op.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm", &rms_norm, "root mean square norm");
  m.def("silu_and_mul", &silu_and_mul, "");
  m.def("rotary_embedding_online", &rotary_embedding_online, "");
}
