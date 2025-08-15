from deepgengraph_exp.cases.kernels import KERNEL_ZOO
from deepgengraph_exp.utils import torch_module_to_onnx

import onnx
import onnxsim

for name, cls in KERNEL_ZOO.items():
  fn = f"{name}.onnx"
  print(f"{name=} {fn=}")
  model = cls()

  specs = model.prepare()
  input_names = list(specs['input'].keys())
  inputs = list(specs['input'].values())
  output_names = list(specs['output'])

  print(f"{input_names=}")
  print(f"{output_names=}")

  onnx_model = torch_module_to_onnx(
    module=model,
    input_names=input_names,
    inputs=inputs,
    output_names=output_names,
  )

  onnx_model, check = onnxsim.simplify(onnx_model)
  assert check
  onnx.save(onnx_model, fn)
