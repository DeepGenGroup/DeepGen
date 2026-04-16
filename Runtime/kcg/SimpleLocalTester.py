import time
import glob
import traceback
import uuid
from typing import Generator, List, Optional, Tuple
import multiprocessing
from kcg.Kernel import *
import kcg.Operators.matmul as kcg_mm
import kcg.Operators.attention as kcg_att
import kcg.Operators.attention_origin as kcg_att_origin
import kcg.Operators.attention_v2 as kcg_att_v2
import kcg.Operators.attention_split as kcg_att_split
import kcg.Operators.attention_gemma2 as kcg_att_gemma2
import kcg.Operators.attention_gemma2_origin as kcg_att_gemma2_origin
import kcg.Operators.attention_h2o as kcg_att_h2o
import kcg.Operators.attention_h2o_origin as kcg_att_h2o_origin
import numpy as np

ctx = multiprocessing.get_context('spawn')
Process = ctx.Process
Value  = ctx.Value

class BenchmarkConfig :
    def __init__(self):
        self.benchmarkCount = 5
        self.warmupCount = 1
        self.keepTopNum = 10
        self.max_kernel_per_iter = 20
        self.result_json_path = ""
        self.maxCount = 0

def init_cuda(_devId : List) :
    DeviceInfo.get_current_device()  # DO NOT REMOVE! Otherwise cuda will report Invalid device id error
    print("init_cuda devid=",_devId)
    DeviceInfo.set_visible_devices(_devId)
    DeviceInfo.set_current_device(_devId[0])  # no comment! set_current_device() still essential for gpu device initialilze. otherwise error occurs
    if not torch_ns.is_available() :
        torch_ns.init()
        torch_ns.empty_cache()

def __compile_task_func(OpTy : Type[OpInterface], info : CompileNeededInfo , deviceId:int, backendtype : EnumBackendType, arch : str , index : int, opt : CompileOption, task_id : str = "") :
    try:
        op = OpTy()
        result = op.Compile(deviceId, backendtype, arch, info, opt)
        pklName = f"{PathManager.pikle_dir()}/{deviceId}/{task_id}/kfg_{index}.pkl"
        if len(result) == 5:
            ba, kernlCfg, compiledKernel, k1Cfg, k2Cfg = result
            serialize_to_file(pklName, (ba, kernlCfg, k1Cfg, k2Cfg))
        elif len(result) == 4:
            ba, kernlCfg, compiledKernel, k1Cfg = result
            serialize_to_file(pklName, (ba, kernlCfg, k1Cfg))
        else:
            ba, kernlCfg, compiledKernel = result
            serialize_to_file(pklName, (ba, kernlCfg))
    except BaseException as e:
        kernel_name = getattr(info, "kernelName", f"cfg_{index}")
        print(f"[compile-skip] {kernel_name} compile failed: {e}", flush=True)
        traceback.print_exc()


g_index : int = 0
def compile_kernel(OpTy, tsGenerator : TsGeneratorType, deviceId:int, backendtype : EnumBackendType, arch : str, kernelLimit = 10, globalLimit = 0, compileOpt : CompileOption = None, task_id : str = "") -> bool:
    # shape, dtypeInt = [[1, 32, 2048, 128], 4]
    global g_index
    localIndex = 0
    maxProcsLimit = 100
    procs : List[Process] = []
    iterationEnds = False
    print('========= compiling ============')
    while True:
        try:
            if globalLimit > 0 and g_index >= globalLimit:
                iterationEnds = True
                break
            needInfo = next(tsGenerator)
        # for needInfo in tsGenerator :
            print(f"needInfo.tsArgs = {needInfo.tsArgs}") 
            print(f"shmBytes = ",needInfo.shmBytes) 
            print(f"blockDims = {needInfo.blockDims}") 
            print(f"gridDims = {needInfo.gridDims}") 
            # create compile process
            p = Process(target=__compile_task_func,args=(OpTy,needInfo,deviceId,backendtype,arch, g_index, compileOpt, task_id))
            procs.append(p)
            p.start()
            localIndex += 1
            g_index += 1
            if localIndex >= kernelLimit :
                break
            if len(procs) >= maxProcsLimit :
                for pp in procs :
                    pp.join()
                procs.clear()
        except StopIteration as e :
            iterationEnds = True
            break
        except BaseException as e :
            import traceback
            print('[Deepgen Exception during compiling] ', e)
            traceback.print_exc()
            continue
    # wait all procs end
    for pp in procs :
        pp.join()
        if pp.exitcode != 0:
            ...
            # raise RuntimeError(f"compile subprocess failed, exitcode={pp.exitcode}")
    procs.clear()
    return iterationEnds

def is_tflops_ok(b,m,n,k,t) :
    TargetTFLOPS = 12.2 * 0.6
    return (2*b*m*n*k / t / 1e-3 / 1e12) >= TargetTFLOPS
    

def _benchProcess( OpTy : Type[OpInterface] , benchConfig : BenchmarkConfig, findGoodCase, devId, time_0 ,pkls : List, checkTFLOPS = False, checkACC : float = 0) :
    init_cuda([devId])
    save_to = benchConfig.result_json_path
    maxSppedups = []
    if len(save_to) > 0 and os.path.exists(save_to) :
        with open(save_to,'r') as f:
            obj = json.load(f)
            maxSppedups = obj['testResult']
            f.close()
    for pkl in pkls :
        try:
            pklData = deserialize_from_file(pkl)
            os.remove(pkl)
            k1_config = None
            k2_config = None
            if len(pklData) == 4:
                ba, config, k1_config, k2_config = pklData
            elif len(pklData) == 3:
                ba, config, k1_config = pklData
            else:
                ba, config = pklData
            assert isinstance(ba, List)
            assert isinstance(config, KernelConfigs)
            print(f'[D] after desrialize : {ba}')
            if config.operatorKind is kcg_mm.MatmulOp :
                m,n,k = ba[1:4]
                b = 1
                if len(ba[0]) > 0:
                    b =  ba[0][0]
            elif config.operatorKind is kcg_att.AttentionOp :
                ...
            elif config.operatorKind is kcg_att_v2.AttentionV2Op :
                ...
            elif config.operatorKind is kcg_att_split.AttentionSplitOp :
                ...
            elif config.operatorKind is kcg_att_gemma2.Gemma2SplitOp :
                ...
            elif config.operatorKind is kcg_att_h2o.H2OSplitOp :
                ...
            # init tensors
            op = OpTy()
            op.InitBaseArgs(ba)
            op.GetBaselineInputTensor(devId)
            op.GetBenchmarkInputTensor(devId)
            print("[D] tensor shape verify ======",flush=True)
            
            # rebuild K1 compiled kernel if config was serialized
            if k1_config is not None:
                sig_k1 = op._get_k1_signature(op.BaseArgs.getTorchDType())
                op._kernel1 = CompiledKernel(
                    k1_config.backend, k1_config.binaryPath, k1_config.kernelFuncName,
                    k1_config.sharedMem(), sig_k1, k1_config.gridDims(), k1_config.blockDims(), devId)
            
            # rebuild K2 compiled kernel if config was serialized (H2O 3-kernel split)
            if k2_config is not None:
                sig_k2 = op._get_k2_signature(op.BaseArgs.getTorchDType())
                op._kernel2 = CompiledKernel(
                    k2_config.backend, k2_config.binaryPath, k2_config.kernelFuncName,
                    k2_config.sharedMem(), sig_k2, k2_config.gridDims(), k2_config.blockDims(), devId)
            
            kernel = op.GetCompiledKernel(config,devId)
            # warmup
            # op.Test_warmup(kernel,1,devId)
            time_base = []
            r0 = None
            t0 = 0
            for i in range(0,7):
                r0, t0 = op.Test_baseline(devId)
                time_base.append(t0)
            t0 = np.median(time_base)
            if time_0.value <= 0 :
                time_0.value = t0
            else:
                t0 = time_0.value
            # benchmark
            r,t = op.Test_benchmark(kernel, 5 , devId)
            # verify result
            acc = 0
            funName = config.kernelFuncName
            if torch.allclose(r,r0,rtol=1e-3,atol=1e-3) :
                acc = t0 / t
                print(f"Test Correct! {funName} , speedup = {acc}")
            else:
                abs_diff = torch.abs(r - r0)
                max_abs_err = abs_diff.max().item()
                rel_diff = abs_diff / (torch.abs(r0) + 1e-8)
                max_rel_err = rel_diff.max().item()
                total = r.numel()
                for thr in [1e-2, 1e-3, 1e-4]:
                    cnt = int((abs_diff > thr).sum().item())
                    print(f"[err-dist] abs_err > {thr:.0e}: {cnt}/{total} ({100.0*cnt/total:.2f}%)")
                pcts = [50, 90, 95, 99, 99.9, 100]
                q_vals = torch.quantile(abs_diff.float().flatten(), torch.tensor([p/100.0 for p in pcts], device=abs_diff.device))
                for p, qv in zip(pcts, q_vals):
                    print(f"[err-dist] P{p}={qv.item():.6e}")
                print(f"Test Error! {funName} max_abs_err={max_abs_err:.6e} max_rel_err={max_rel_err:.6e}")
                print(f"Test Error! {funName} r= {r} , r0 = {r0}")
            # save result

            # kernel.release()
            if acc > 0 :
                obj = {"name" : funName, "speedup" : acc ,"time" : t, "time_base" : t0}
                maxSppedups.append(obj)
                maxSppedups.sort(key= lambda x: x["speedup"],reverse=True)
                if len(maxSppedups) > benchConfig.keepTopNum :
                    maxSppedups = maxSppedups[0:benchConfig.keepTopNum]
                if checkTFLOPS and config.operatorKind is kcg_mm.MatmulOp and is_tflops_ok(b,m,n,k,t) :
                    findGoodCase.value = 1
                if checkACC > 0 and acc >= checkACC :
                    findGoodCase.value = 1 
        except BaseException as e: 
            print('[Deepgen Exception] ',e)
            traceback.print_exc()
        except IOError as e: 
            print('[Deepgen IOError] ',e)
            traceback.print_exc()
    if len(save_to) > 0 :
        with open(save_to,'w+') as f:
            result = {"testResult" : maxSppedups}
            json.dump(result,f,indent=2)
            f.flush()
        print(f"[result-json] wrote {save_to} with {len(maxSppedups)} entries", flush=True)
    return

g_time0 = Value('d',0.0)
tensor_input_baseline = None
tensor_input_benchmark = None
g_result = None
g_findAvaialbleCase = Value('d',0.0)
def do_benchmark(OpTy : Type[OpInterface], devId : int, benchConfig : BenchmarkConfig, maxSppedups : List[Dict], checkTflops : bool, checkAcc : float, task_id : str = ""):
    global g_time0 
    global tensor_input_baseline 
    global tensor_input_benchmark 
    global g_result 
    global g_findAvaialbleCase
    name_format = f"{PathManager.pikle_dir()}/{devId}/{task_id}/*.pkl"
    pkls = glob.glob(name_format)
    print(f"[benchmark] discovered {len(pkls)} serialized kernels under {name_format}", flush=True)
    p = Process(target = _benchProcess, args = (OpTy,benchConfig, g_findAvaialbleCase, devId, g_time0, pkls, checkTflops, checkAcc))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"benchmark subprocess failed, exitcode={p.exitcode}")
    # process terminated. clean undealed pkls
    # if p.is_alive():
    #     p.terminate()
    #     p.join(5)
    for pkl in pkls :
        if os.path.exists(pkl) :
            os.remove(pkl)   # delete crashed pkl
    
    
    
_dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _get_gemmstats_like_cfgs(cfg_dict, shape: List[int], type_width: int):
    """Build a K1-only tuning space without fake O-side constraints."""
    hd = shape[3]
    cfg = dict(cfg_dict)
    cfg["Hd"] = [hd]
    warp_size = cfg["WARP_SIZE"][0]
    block_layout_px = 1
    shared_softmax_rows = 1 if warp_size > 32 else 3
    results = []

    for ptr in cfg["PTr"]:
        for ptc in cfg["PTc"]:
            for br in cfg["Br"]:
                for bc in cfg["Bc"]:
                    if br > shape[2] or bc > shape[2]:
                        continue
                    if br % ptr != 0 or bc % ptc != 0:
                        continue
                    thread_num = (br // ptr) * (bc // ptc)
                    if thread_num <= 0 or thread_num > 256:
                        continue

                    by = br // ptr
                    bx = bc // ptc
                    p_layouts = []
                    for bly in cfg["BLOCK_LAYOUT_P_Y"]:
                        for wly in cfg["WARP_LAYOUT_P_Y"]:
                            for wlx in cfg["WARP_LAYOUT_P_X"]:
                                if wly * wlx != warp_size:
                                    continue
                                if wlx != bx:
                                    continue
                                if bly * wly != by:
                                    continue
                                p_layouts.append((bly, block_layout_px, wly, wlx))
                    if not p_layouts:
                        continue

                    # Canonical fake O-side params so kernel naming stays stable,
                    # but K1 space is still driven only by real K1 knobs.
                    otr = ptr
                    if (hd * ptc) % bc != 0:
                        continue
                    otc = (hd * ptc) // bc
                    if otc <= 0:
                        continue

                    for s1 in cfg["Slice1"]:
                        ldtw_q = br * s1 / thread_num
                        ldtw_k = bc * s1 / thread_num
                        for glwq in cfg["GLOB_LOAD_WIDTH_Q"]:
                            if ldtw_q < glwq:
                                continue
                            for glwk in cfg["GLOB_LOAD_WIDTH_K"]:
                                if ldtw_k < glwk:
                                    continue
                                for bly_p, blx_p, wly_p, wlx_p in p_layouts:
                                    for bswq in cfg["BLOCK_SCATTER_WIDTH_Q"]:
                                        if bswq > ptr:
                                            continue
                                        for bswk in cfg["BLOCK_SCATTER_WIDTH_K"]:
                                            if bswk > ptc:
                                                continue
                                            for wswq in cfg["WARP_SCATTER_WIDTH_Q"]:
                                                if wswq > bswq:
                                                    continue
                                                for wswk in cfg["WARP_SCATTER_WIDTH_K"]:
                                                    if wswk > bswk:
                                                        continue
                                                    fake_bswv = min(bswk, otc)
                                                    fake_wswv = min(wswk, fake_bswv)
                                                    for spp in cfg["SHARED_PREFETCH_P"]:
                                                        smem_elems = br * s1 + bc * s1 + shared_softmax_rows * br
                                                        if s1 != hd:
                                                            smem_elems += hd * br
                                                        if spp == 1:
                                                            smem_elems += bc * s1
                                                        smem_bytes = smem_elems * type_width
                                                        if smem_bytes > 65536:
                                                            continue
                                                        for rpp in cfg["REG_PREFETCH_P"]:
                                                            for unroll in cfg["UNROLL_NUM"]:
                                                                results.append((
                                                                    thread_num,
                                                                    (br, bc, hd, s1, 4),
                                                                    (ptr, ptc, otr, otc),
                                                                    (glwq, glwk, 4),
                                                                    (bly_p, blx_p, wly_p, wlx_p, bswq, bswk, wswq, wswk),
                                                                    (bly_p, blx_p, wly_p, wlx_p, bswq, fake_bswv, wswq, fake_wswv),
                                                                    (unroll, warp_size, cfg["LOAD_CONTINUOUS_P"][0], 1, spp, rpp, 0, 0, 0),
                                                                    smem_bytes,
                                                                ))
    return results


def _get_gemmnormcolsum_like_cfgs(cfg_dict, shape: List[int], type_width: int):
    """Build an H2O-K2 tune space using only the real P-side constraints."""
    hd = shape[3]
    cfg = dict(cfg_dict)
    cfg["Hd"] = [hd]
    warp_size = cfg["WARP_SIZE"][0]
    block_layout_px = 1
    wave64_fixcol_elems = hd if warp_size > 32 else 0
    results = []

    for ptr in cfg["PTr"]:
        for ptc in cfg["PTc"]:
            for br in cfg["Br"]:
                for bc in cfg["Bc"]:
                    if br > shape[2] or bc > shape[2]:
                        continue
                    if br % ptr != 0 or bc % ptc != 0:
                        continue
                    thread_num = (br // ptr) * (bc // ptc)
                    if thread_num <= 0 or thread_num > 256:
                        continue

                    by = br // ptr
                    bx = bc // ptc
                    p_layouts = []
                    for bly in cfg["BLOCK_LAYOUT_P_Y"]:
                        for wly in cfg["WARP_LAYOUT_P_Y"]:
                            for wlx in cfg["WARP_LAYOUT_P_X"]:
                                if wly * wlx != warp_size:
                                    continue
                                if wlx != bx:
                                    continue
                                if bly * wly != by:
                                    continue
                                p_layouts.append((bly, block_layout_px, wly, wlx))
                    if not p_layouts:
                        continue

                    # Canonical fake O-side params so kernel naming stays stable,
                    # while the tune space is driven only by GemmNormColSum knobs.
                    otr = ptr
                    if (hd * ptc) % bc != 0:
                        continue
                    otc = (hd * ptc) // bc
                    if otc <= 0:
                        continue

                    for s1 in cfg["Slice1"]:
                        ldtw_q = br * s1 / thread_num
                        ldtw_k = bc * s1 / thread_num
                        for glwq in cfg["GLOB_LOAD_WIDTH_Q"]:
                            if ldtw_q < glwq:
                                continue
                            for glwk in cfg["GLOB_LOAD_WIDTH_K"]:
                                if ldtw_k < glwk:
                                    continue
                                for bly_p, blx_p, wly_p, wlx_p in p_layouts:
                                    for bswq in cfg["BLOCK_SCATTER_WIDTH_Q"]:
                                        if bswq > ptr:
                                            continue
                                        for bswk in cfg["BLOCK_SCATTER_WIDTH_K"]:
                                            if bswk > ptc:
                                                continue
                                            for wswq in cfg["WARP_SCATTER_WIDTH_Q"]:
                                                if wswq > bswq:
                                                    continue
                                                for wswk in cfg["WARP_SCATTER_WIDTH_K"]:
                                                    if wswk > bswk:
                                                        continue
                                                    fake_bswv = min(bswk, otc)
                                                    fake_wswv = min(wswk, fake_bswv)
                                                    for spp in cfg["SHARED_PREFETCH_P"]:
                                                        # GemmNormColSum uses smK + smQ, optional smKFull,
                                                        # optional prefetch smQ, plus a wave64 fix-column buffer.
                                                        smem_elems = br * s1 + bc * s1 + wave64_fixcol_elems
                                                        if s1 != hd:
                                                            smem_elems += hd * br
                                                        if spp == 1:
                                                            smem_elems += bc * s1
                                                        smem_bytes = smem_elems * type_width
                                                        if smem_bytes > 65536:
                                                            continue
                                                        for rpp in cfg["REG_PREFETCH_P"]:
                                                            for unroll in cfg["UNROLL_NUM"]:
                                                                results.append((
                                                                    thread_num,
                                                                    (br, bc, hd, s1, 4),
                                                                    (ptr, ptc, otr, otc),
                                                                    (glwq, glwk, 4),
                                                                    (bly_p, blx_p, wly_p, wlx_p, bswq, bswk, wswq, wswk),
                                                                    (bly_p, blx_p, wly_p, wlx_p, bswq, fake_bswv, wswq, fake_wswv),
                                                                    (unroll, warp_size, cfg["LOAD_CONTINUOUS_P"][0], 1, spp, rpp, 0, 0, 0),
                                                                    smem_bytes,
                                                                ))
    return results

def get_tuning_space(OpTy : Type[OpInterface], cfgPath : str, torch_dtype : torch.dtype = torch.float32, seqlen: Optional[int] = None) -> TsGeneratorType :
    def _shape(default_seqlen: int) -> List[int]:
        shape = [1, 32, default_seqlen, 64]
        if seqlen is not None:
            shape[2] = seqlen
        return shape

    def _dump_subcfg(full_cfg, sub_key: Optional[str], fill_o_defaults: bool = False,
                     overrides: Optional[dict] = None) -> str:
        import json, tempfile

        if sub_key is not None and sub_key in full_cfg:
            sub_cfg = full_cfg[sub_key]
        else:
            sub_cfg = full_cfg
        sub_cfg = dict(sub_cfg)
        if fill_o_defaults:
            o_defaults = {
                "Slice2": [4], "OTr": [4], "OTc": [8], "GLOB_LOAD_WIDTH_V": [4],
                "BLOCK_LAYOUT_O_Y": [2], "BLOCK_LAYOUT_O_X": [1],
                "WARP_LAYOUT_O_Y": [4], "WARP_LAYOUT_O_X": [8],
                "BLOCK_SCATTER_WIDTH_P": [4], "BLOCK_SCATTER_WIDTH_V": [4],
                "WARP_SCATTER_WIDTH_P": [4], "WARP_SCATTER_WIDTH_V": [4],
                "LOAD_CONTINUOUS_O": [1], "REG_PREFETCH_O": [0],
                "SHUFFLE_P": [0], "SPLITK_PV": [0],
            }
            for k, v in o_defaults.items():
                sub_cfg.setdefault(k, v)
        if overrides:
            for k, v in overrides.items():
                sub_cfg[k] = v
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(sub_cfg, tmp)
        tmp.flush()
        tmp.close()
        return tmp.name

    origin_fixed_cfg = {
        "SHARED_PREFETCH_P": [0],
        "REG_PREFETCH_P": [0],
        "REG_PREFETCH_O": [0],
        "SHUFFLE_P": [0],
        "SPLITK_PV": [0],
    }

    if OpTy is kcg_mm.MatmulOp :
        import kcg.tuning.NewCfgTest as ns_mm
        return ns_mm.getTuneSpace(cfgPath)
    if OpTy is kcg_att.AttentionOp :
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        return ns_attentiopn.getTuneSpace(_shape(2048), cfgPath, [], torch_dtype)
    if OpTy is kcg_att_origin.AttentionOriginOp :
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, None, overrides=origin_fixed_cfg)
        return ns_attentiopn.getTuneSpace(_shape(2048), tmp_cfg, [], torch_dtype, kernel_prefix="AttentionOrigin")
    if OpTy is kcg_att_v2.AttentionV2Op :
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        return ns_attentiopn.getTuneSpace(_shape(2048), cfgPath, [], torch_dtype)
    if OpTy is kcg_att_split.AttentionSplitOp :
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, "k2")
        return ns_attentiopn.getTuneSpace(_shape(2048), tmp_cfg, [], torch_dtype)
    if OpTy is kcg_att_split.AttentionK1Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn

        shape = _shape(4096)
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        enum_dtype = ToEnumIntDType(torch_dtype)
        type_width = sizeof(enum_dtype)
        sub_cfg = full_cfg["k1"] if "k1" in full_cfg else full_cfg
        cfgs = _get_gemmstats_like_cfgs(sub_cfg, shape, type_width)
        if len(cfgs) == 0:
            print(f"[Warn] no valid AttentionK1 tuning cfg generated for shape={shape} from {cfgPath}", flush=True)
        return ns_attentiopn.getTuneSpace(shape, cfgPath, cfgs, torch_dtype, kernel_prefix="AttentionK1")
    if OpTy is kcg_att_split.AttentionK2Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn

        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, "k2")
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype, kernel_prefix="AttentionK2")
    if OpTy is kcg_att_gemma2.Gemma2SplitOp :
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, "k2")
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype, kernel_prefix="Gemma2")
    if OpTy is kcg_att_gemma2_origin.Gemma2OriginOp :
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, None, overrides=origin_fixed_cfg)
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype, kernel_prefix="Gemma2Origin")
    if OpTy is kcg_att_gemma2.Gemma2K1Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn

        shape = _shape(4096)
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        enum_dtype = ToEnumIntDType(torch_dtype)
        type_width = sizeof(enum_dtype)
        sub_cfg = full_cfg["k1"] if "k1" in full_cfg else full_cfg
        cfgs = _get_gemmstats_like_cfgs(sub_cfg, shape, type_width)
        if len(cfgs) == 0:
            print(f"[Warn] no valid Gemma2K1 tuning cfg generated for shape={shape} from {cfgPath}", flush=True)
        return ns_attentiopn.getTuneSpace(shape, cfgPath, cfgs, torch_dtype, kernel_prefix="Gemma2K1")
    if OpTy is kcg_att_gemma2.Gemma2K2Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn

        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, "k2")
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype, kernel_prefix="Gemma2K2")
    if OpTy is kcg_att_h2o.H2OSplitOp :
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, "k3")
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype)
    if OpTy is kcg_att_h2o_origin.H2OOriginOp :
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, None, overrides=origin_fixed_cfg)
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype, kernel_prefix="H2OOrigin")
    if OpTy is kcg_att_h2o.H2OK1Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn

        shape = _shape(4096)
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        enum_dtype = ToEnumIntDType(torch_dtype)
        type_width = sizeof(enum_dtype)
        sub_cfg = full_cfg["k1"] if "k1" in full_cfg else full_cfg
        cfgs = _get_gemmstats_like_cfgs(sub_cfg, shape, type_width)
        if len(cfgs) == 0:
            print(f"[Warn] no valid H2OK1 tuning cfg generated for shape={shape} from {cfgPath}", flush=True)
        return ns_attentiopn.getTuneSpace(shape, cfgPath, cfgs, torch_dtype, kernel_prefix="H2OK1")
    if OpTy is kcg_att_h2o.H2OK2Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn

        shape = _shape(4096)
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        enum_dtype = ToEnumIntDType(torch_dtype)
        type_width = sizeof(enum_dtype)
        sub_cfg = full_cfg["k2"] if "k2" in full_cfg else full_cfg
        cfgs = _get_gemmnormcolsum_like_cfgs(sub_cfg, shape, type_width)
        if len(cfgs) == 0:
            print(f"[Warn] no valid H2OK2 tuning cfg generated for shape={shape} from {cfgPath}", flush=True)
        return ns_attentiopn.getTuneSpace(shape, cfgPath, cfgs, torch_dtype, kernel_prefix="H2OK2")
    if OpTy is kcg_att_h2o.H2OK3Op:
        import json
        import kcg.tuning.attn_FP32_test as ns_attentiopn
        with open(cfgPath, 'r') as f:
            full_cfg = json.load(f)
        tmp_cfg = _dump_subcfg(full_cfg, "k3")
        return ns_attentiopn.getTuneSpace(_shape(4096), tmp_cfg, [], torch_dtype, kernel_prefix="H2OK3")
    assert False, f'[Error] getTuningSpace : Invalid OpTy:{OpTy.__name__}'
    

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
# def test_simple() :
#     hsacoPath = "/tmp/compile-ptx-src-ebe279.cubin"
#     kernelName = "attention1"
#     dt = torch.float32
#     dev = 7
#     op = attention.AttentionOp()
    
#     init_cuda(dev)
#     op.InitBaseArgs([[1, 32, 2048, 128],4])
#     info = KernelConfigs(hsacoPath,kernelName,[dt,dt,dt,dt], EnumBackendType.CUDA)
# # ==== shmBytes is 29440
# # attOp: 29440,[32, 32, 1],[256, 1, 1],
#     info.m_gridDims = [32, 32, 1]
#     info.m_blockDims = [256,1,1]
#     info.shmBytes = 29440
#     kernel = op.GetCompiledKernel(info,dev)
#     a,b,c =  op.GetBaselineInputTensor(dev)
#     aa,bb,cc,dd = op.GetBenchmarkInputTensor(dev)
#     kernel.run( aa,bb,cc,dd )
#     print(dd)
#     # op.Test_baseline(7)
#     # op.Test_benchmark(kernel,dev)


# 交替进行compile & benchmark，每次 {kernelLimit} 个 krnl
def do_compile_and_benchmark_alternatively(opty : Type[OpInterface], ts : TsGeneratorType , cc : BenchmarkConfig, compileOption : CompileOption, backend : EnumBackendType , arch : str ,devId : int, checktflops:bool, checkAcc:float, task_id : str = "", totalSize : int = 0) :
    maxSpeedups = []
    currIter = 0

    while not compile_kernel(opty,ts,devId,backend,arch, cc.max_kernel_per_iter, cc.maxCount, compileOption, task_id) :
        doneCount = min(g_index, totalSize) if totalSize > 0 else g_index
        pct = f" ({doneCount}/{totalSize}, {100.0*doneCount/totalSize:.1f}%)" if totalSize > 0 else ""
        print(f"=========== benchmark {currIter}, tuned {doneCount}{pct} ====== ", flush=True)
        currIter+=1
        do_benchmark(opty,devId,cc,maxSpeedups, checktflops, checkAcc, task_id)
        if g_findAvaialbleCase.value > 0 :
            print(f"=========== Find available Case ! Stopped ====== ")
            return
    do_benchmark(opty,devId,cc,maxSpeedups, checktflops, checkAcc, task_id)
    if g_findAvaialbleCase.value > 0 :
        print(f"=========== Find available Case ! Stopped ====== ")
        return


_opty_map = {
    "matmul":  kcg_mm.MatmulOp,
    "attn_v1": kcg_att.AttentionOp,
    "attention_origin": kcg_att_origin.AttentionOriginOp,
    "attn_v2": kcg_att_v2.AttentionV2Op,
    "attn_split": kcg_att_split.AttentionSplitOp,
    "attn_k1": kcg_att_split.AttentionK1Op,
    "attn_k2": kcg_att_split.AttentionK2Op,
    "gemma2_split": kcg_att_gemma2.Gemma2SplitOp,
    "gemma2_origin": kcg_att_gemma2_origin.Gemma2OriginOp,
    "gemma2_k1": kcg_att_gemma2.Gemma2K1Op,
    "gemma2_k2": kcg_att_gemma2.Gemma2K2Op,
    "h2o_split": kcg_att_h2o.H2OSplitOp,
    "h2o_origin": kcg_att_h2o_origin.H2OOriginOp,
    "h2o_k1": kcg_att_h2o.H2OK1Op,
    "h2o_k2": kcg_att_h2o.H2OK2Op,
    "h2o_k3": kcg_att_h2o.H2OK3Op,
}

def _extract_int_flag(argv: List[str], flag_name: str) -> Tuple[List[str], Optional[int]]:
    """Extract an int flag from argv (supports --name N and --name=N)."""
    out: List[str] = []
    value: Optional[int] = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == flag_name:
            if i + 1 >= len(argv):
                raise ValueError(f"missing value after {flag_name}")
            if value is not None:
                raise ValueError(f"duplicated flag {flag_name}")
            value = int(argv[i + 1])
            i += 2
            continue
        if tok.startswith(flag_name + "="):
            if value is not None:
                raise ValueError(f"duplicated flag {flag_name}")
            value = int(tok.split("=", 1)[1])
            i += 1
            continue
        out.append(tok)
        i += 1
    return out, value

def getInputs() :
    helpmsg = "Usage : cfgFile result_json_path start maxCount checktflops(1,0) checkAcc(float) [opty(matmul|attn_v1|attn_v2|attn_split|attn_k1|attn_k2|gemma2_split|gemma2_k1|gemma2_k2|h2o_split|h2o_k1|h2o_k2|h2o_k3)] [devId(int)] [dtype(float32|float16)] [--seqlen int]"
    argv = sys.argv[1:]
    try:
        argv, seqlen = _extract_int_flag(argv, "--seqlen")
    except ValueError as e:
        print(helpmsg)
        raise

    if len(argv) < 4 :
        print(helpmsg)
        assert False, f"invalid input args. {helpmsg}"
        
    cfgFile = argv[0]
    result_json_path = argv[1]
    start = int(argv[2])
    maxCount = int(argv[3])

    if len(argv) >= 6:
        checktflops, checkAcc = int(argv[4]) > 0 , float(argv[5])
    elif len(argv) >= 5:
        print(helpmsg)
        assert False, f"invalid input args (missing checkAcc). {helpmsg}"
    else:
        checktflops, checkAcc = True, 0

    opty_name = argv[6] if len(argv) >= 7 else "attn_v1"
    assert opty_name in _opty_map, f"invalid opty '{opty_name}', must be one of {list(_opty_map.keys())}"
    opty = _opty_map[opty_name]
    devId = int(argv[7]) if len(argv) >= 8 else 7
    dtype_name = argv[8] if len(argv) >= 9 else "float32"
    assert dtype_name in _dtype_map, f"invalid dtype '{dtype_name}', must be one of {list(_dtype_map.keys())}"
    torch_dtype = _dtype_map[dtype_name]
    if seqlen is not None and seqlen <= 0:
        raise ValueError("--seqlen must be > 0")
    return (cfgFile,result_json_path,start,maxCount,checktflops, checkAcc, opty, devId, torch_dtype, seqlen)

def main():
    cfgFile,result_json_path,start,maxCount,checktflops, checkAcc, opty, devId, torch_dtype, seqlen = getInputs()
    print(f"[Info] opty = {opty.__name__}, devId = {devId}, dtype = {torch_dtype}")

    if is_hip():
        backend = EnumBackendType.HIP
        arch = "906"
    else:
        backend = EnumBackendType.CUDA
        arch = "80"
    
    task_id = uuid.uuid4().hex[:8]
    print(f"[Info] task_id = {task_id}")
    PathManager.init(clearPkl=False, clearCache=True)
    os.makedirs(f"{PathManager().pikle_dir()}/{devId}/{task_id}", exist_ok=True)
    print("get_tune_space",flush=True)
    tssize = 0
    for c in  get_tuning_space(opty, cfgFile, torch_dtype, seqlen=seqlen):
        tssize += 1
    print(f"==== tune space size = {tssize}")
    
    # return
    print("=== checktflops, checkAcc",checktflops, checkAcc)
    ts = get_tuning_space(opty, cfgFile, torch_dtype, seqlen=seqlen)
    bc = BenchmarkConfig()
    bc.keepTopNum = 10
    bc.max_kernel_per_iter = 80
    bc.result_json_path = result_json_path
    bc.maxCount = maxCount
    st = time.time()
    print(f"=====  start at : {st}")
    
    
    if start > 0:
        i=0
        for _ in ts :
            i+=1
            if i >= start:
                break 
    co = CompileOption()
    co.fastCompile = False
    do_compile_and_benchmark_alternatively(opty,ts,bc,co,backend,arch,devId,checktflops, checkAcc, task_id, tssize)
    if result_json_path:
        if not os.path.exists(result_json_path):
            raise RuntimeError(f"result json not found: {result_json_path}")
        with open(result_json_path, 'r') as f:
            obj = json.load(f)
        result_count = len(obj.get("testResult", []))
        print(f"[result-json] final result count = {result_count}", flush=True)
        if result_count == 0:
            raise RuntimeError(f"no valid kernels were recorded in {result_json_path}")
    # compile_kernel(opty,ts,devId,backend,arch,kernelLimit=1)
    # do_benchmark(opty,devId,cc,[])
    et = time.time()
    print(f"===== Complete! Total spends {(et - st)/ 60} minutes")
    
    
if __name__ == '__main__' :
    print(time.strftime('------------- %Y-%m-%d %H:%M:%S ------------- ',time.localtime(time.time())))       # 打印按指定格式排版的时间
    main()
