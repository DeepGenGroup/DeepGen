import onnx
from onnx import helper, shape_inference
import numpy as np
from typing import List,Tuple
#   struct KernelData {
#     /* kernel的数据结构 */
#     std::string name;
#     std::string type;
#     std::vector<std::string> argNames;
#     std::vector<std::vector<int64_t>> shapes;
#     std::vector<std::string> dtypes;
#     std::vector<bool> isTrans;
#     int outputArgNum;
#   };

class KernelData :
    def __init__(self):
        self.op_name = None
        self.op_type = None
        self.input_argNames = []
        self.input_shapes = []
        self.input_dtypes = []
        self.isTrans = []
        self.output_argNames = []
        self.output_shapes = []
        self.output_dtypes = []
        self.op_attrs = []
        
    def __str__(self):
        return f"{self.op_type} : {self.op_name}"
    
def get_tensor_info(model):
    """构建包含所有张量信息的字典"""
    tensor_info = {}
    
    # 处理输入
    for input in model.graph.input:
        tensor_type = input.type.tensor_type
        dtype = tensor_type.elem_type
        shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                for dim in tensor_type.shape.dim]
        tensor_info[input.name] = {
            'dtype': dtype,
            'shape': shape,
            'source': 'input'
        }
    
    # 处理输出
    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        dtype = tensor_type.elem_type
        shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                for dim in tensor_type.shape.dim]
        tensor_info[output.name] = {
            'dtype': dtype,
            'shape': shape,
            'source': 'output'
        }
    
    # 处理中间张量
    for value_info in model.graph.value_info:
        tensor_type = value_info.type.tensor_type
        dtype = tensor_type.elem_type
        shape = [dim.dim_param if dim.dim_param else dim.dim_value 
                for dim in tensor_type.shape.dim]
        tensor_info[value_info.name] = {
            'dtype': dtype,
            'shape': shape,
            'source': 'value_info'
        }
    
    # 处理初始权重
    for init in model.graph.initializer:
        if init.name not in tensor_info:
            tensor_info[init.name] = {
                'dtype': init.data_type,
                'shape': list(init.dims),
                'source': 'initializer'
            }
    
    return tensor_info

def dtype_to_name(dtype):
    """将数据类型编号转换为可读名称"""
    return helper.tensor_dtype_to_np_dtype(dtype).name

def parse_onnx_model(model_path) -> List[KernelData]:
    # 加载并推断形状
    model = onnx.load(model_path)
    model = shape_inference.infer_shapes(model)
    
    tensor_info = get_tensor_info(model)
    
    # 打印节点信息
    print("="*50)
    print("Operator Details:")
    kdList = []
    for node_idx, node in enumerate(model.graph.node):
        kd = KernelData()
        # print(f"\nOperator {node_idx}: {node.op_type}")
        # print(f"  Name: {node.name}" if node.name else "  [Unnamed Operator]")
        kd.op_name = node.name
        kd.op_type = node.op_type

        # 输入信息
        # print("  Inputs:")
        for input_name in node.input:
            info = tensor_info.get(input_name, {})
            kd.input_argNames.append(input_name)
            dtype = dtype_to_name(info.get('dtype', 0)) if info else "Unknown"
            kd.input_dtypes.append(dtype)
            shape = info.get('shape', [])
            kd.input_shapes.append(shape)
            source = info.get('source', 'unknown')
            
            shape_str = [f"'{s}'" if isinstance(s, str) else str(s) 
                       for s in shape]
            # print(f"    {input_name}: {dtype}{shape_str} ({source})")
        
        # 输出信息
        # print("  Outputs:")
        for output_name in node.output:
            info = tensor_info.get(output_name, {})
            dtype = dtype_to_name(info.get('dtype', 0)) if info else "Unknown"
            shape = info.get('shape', [])
            source = info.get('source', 'unknown')
            
            kd.output_argNames.append(output_name)
            kd.output_dtypes.append(shape)
            kd.output_shapes.append(shape)
            
            shape_str = [f"'{s}'" if isinstance(s, str) else str(s) 
                       for s in shape]
            # print(f"    {output_name}: {dtype},{shape_str}, ({source})")
        kdList.append(kd)
    return kdList

if __name__ == "__main__":
    model_path = "/home/xushilong/DeepGen/_TempCodes/attention_model.onnx"  # 修改为你的模型路径
    kdlist = parse_onnx_model(model_path)
    for kd in kdlist:
        if kd.op_type == 'MatMul' :
            print(kd)
            print(kd.input_shapes)
            print(kd.output_shapes) 
            