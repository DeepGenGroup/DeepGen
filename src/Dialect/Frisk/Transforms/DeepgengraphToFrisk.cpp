#include "Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mlir::frisk {

#define GEN_PASS_DEF_KERNELOPTOFRISK
#define GEN_PASS_DEF_MEMOPTOFRISK
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

namespace dg = deepgengraph ;
namespace dgt = deepgengraph::triton;


static Type convertPointerType(deepgengraph::triton::PointerType ptrType) {
  auto tensorTy = ptrType.getPointeeType();
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{},  tensorTy.getEncoding());
}

static Type convertBlockPointerType(deepgengraph::triton::BlockPointerType blockPtrType) {
  auto tensorTy = blockPtrType.getPointeeType();
  SmallVector<int64_t> dynStrides(tensorTy.getRank(), ShapedType::kDynamic);
  // auto layout = StridedLayoutAttr::get(blockPtrType.getContext(), ShapedType::kDynamic, dynStrides);
  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{}, tensorTy.getEncoding());
}

static void addMaterializations(TypeConverter &tc) {
  tc.addTargetMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
  tc.addSourceMaterialization(
      [](OpBuilder &builder, Type resultType, ValueRange inputs, Location loc) -> Value {
        return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
      });
}

static Value getKernelArgById(Operation *op, int64_t argId) {
  auto kernelOp = op->getParentOfType<frisk::KernelOp>();
  if (!kernelOp)
    return {};
  if (argId < 0 || argId >= static_cast<int64_t>(kernelOp.getNumArguments()))
    return {};
  return kernelOp.getArgument(argId);
}

static bool isTritonPointerLike(Type type) {
  return isa<deepgengraph::triton::PointerType, deepgengraph::triton::BlockPointerType>(type);
}

static void AppendMemspaceToMemrefValue(Value& v, frisk::attr::MemorySpace ms){
  if(mlir::isa<MemRefType>(v.getType())){
    auto _ty = mlir::cast<MemRefType>(v.getType());
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    v.setType(tA);
  }
}

static Type ModifyMemrefType(Type t, frisk::attr::MemorySpace ms){
  if(mlir::isa<MemRefType>(t)){
    auto _ty = mlir::cast<MemRefType>(t);
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    return tA;
  }
  else{
    return t;
  }
}


// 从v开始，向上追溯其defOp，构建affine_expr表达式
static AffineExpr GetExprOfValue(
  mlir::Value v,  // 待分析的value
  std::map<std::string, AffineExpr>& dims,   // dims 容器
  std::map<int,Value>& arglist)  // 记录affinemap的参数的id与Value
{
  auto defOp = v.getDefiningOp();
  if(defOp == nullptr){
    if(auto blockarg = mlir::dyn_cast<BlockArgument>(v)){
      auto argId = blockarg.getArgNumber();
      auto parentOp = blockarg.getParentRegion()->getParentOp();
      if(auto concreteOp = mlir::dyn_cast<frisk::ParallelOp>(parentOp)){
        if(argId > 2){
          assert(false);
        }
        const char* labels[] = {"bz", "by", "bx"};
        if(dims.find(labels[argId]) == dims.end()){
          auto id = dims.size();
          arglist.insert(std::make_pair(id, blockarg));
          dims[labels[argId]] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims[labels[argId]];
      }
      else{
        assert(false);
      }
    }
    else{
      assert(false);
    }
  }
  if(mlir::isa<arith::AddIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) + GetExprOfValue(rhs, dims, arglist);
  }
  else if(mlir::isa<arith::SubIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) - GetExprOfValue(rhs, dims, arglist);
  }
  if(mlir::isa<arith::MulIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) * GetExprOfValue(rhs, dims, arglist);
  }
  else if(mlir::isa<arith::DivUIOp, arith::DivSIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist).floorDiv(GetExprOfValue(rhs, dims, arglist)) ;
  }
  else if(mlir::isa<arith::RemUIOp, arith::RemSIOp>(defOp)){
    auto lhs = defOp->getOperand(0);
    auto rhs = defOp->getOperand(1);
    return GetExprOfValue(lhs, dims, arglist) % GetExprOfValue(rhs, dims, arglist);
  }
  else if(mlir::isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp>(defOp)){
    int val = -999 ;
    auto constOp = mlir::dyn_cast<arith::ConstantOp>(defOp);
    if(constOp){
      val = mlir::cast<IntegerAttr>(constOp.getValue()).getInt();
    }
    return getAffineConstantExpr(val, v.getContext());
  }
  else if(mlir::isa<gpu::BlockIdOp>(defOp)){
    auto op = mlir::dyn_cast<gpu::BlockIdOp>(defOp);
    auto d = op.getDimension();
    const char* label[] = {"bx","by","bz"};
    size_t labelId = -1;
    switch (d) {
      case gpu::Dimension::x:
        labelId = 0; break;
      case gpu::Dimension::y:
        labelId = 1; break;
      case gpu::Dimension::z:
        labelId = 2; break;
      default:
        assert(false);
    }
    
    if(dims.find(label[labelId]) == dims.end()){
      auto id = dims.size();
      arglist[id] = op;
      dims[label[labelId]] = mlir::getAffineDimExpr(id, v.getContext());
    }
    return dims[label[labelId]] ;
  }
  else if(mlir::isa<gpu::ThreadIdOp>(defOp)){
    auto op = mlir::dyn_cast<gpu::BlockIdOp>(defOp);
    auto d = op.getDimension();
    AffineExpr ret;
    switch (d) {
      case gpu::Dimension::x:
        if(dims.find("tx") == dims.end()){
          auto id = dims.size();
          arglist[id] = op;
          dims["tx"] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims["tx"];
      case gpu::Dimension::y:
        if(dims.find("ty") == dims.end()){
          auto id = dims.size();
          arglist[id] = op;
          dims["ty"] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims["ty"];
      case gpu::Dimension::z:
        if(dims.find("tz") == dims.end()){
          auto id = dims.size();
          arglist[id] = op;
          dims["tz"] = mlir::getAffineDimExpr(id, v.getContext());
        }
        return dims["tz"];
      default:
        assert(false);
    }
  }
  // not supported op

  assert(false);
}

// 
struct ArgIdViewBuffer {
  frisk::BufferViewOp view = nullptr;
  frisk::AllocBufferOp buffer = nullptr;
};

// 存放 argId : { arg对应的initView ， arg开辟view时建立的shm buffer }
static std::map<int, ArgIdViewBuffer* > s_map_argId_initBufferView;

// ----------------- Patterns ----------

struct KernelOpConversionPattern : public OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto gridAttr = op->getAttr("grid");
    auto loc = op->getLoc();
    auto oldFuncType = op.getFunctionType();
    auto converter = getTypeConverter();

    llvm::SmallVector<Type> newInputs;
    llvm::SmallVector<Type> newOutputs;
    for (auto ty : oldFuncType.getInputs()) {
      auto newArgTy = converter->convertType(ty);
      if(mlir::isa<MemRefType>(newArgTy)){
        auto mem = mlir::cast<MemRefType>(newArgTy);
        auto newMem = MemRefType::get(mem.getShape(), mem.getElementType(), AffineMap{}, int(frisk::attr::MemorySpace::Global));
        newInputs.push_back(newMem);
      }
      else{
        newInputs.push_back(newArgTy);
      }
    }
    // 1. build new function type
    auto newFuncType = rewriter.getFunctionType(newInputs, newOutputs);
    // 2. convert old region signature, inline it after new frisk.kernel
    TypeConverter::SignatureConversion sc{oldFuncType.getNumInputs()};
    for (int i = 0; i < oldFuncType.getNumInputs(); ++i) {
      sc.addInputs(i, newInputs[i]);
      // sc.addInputs(i, converter->convertType(oldFuncType.getInput(i)));
    }

    rewriter.convertRegionTypes(&op->getRegion(0), *converter, &sc);
    rewriter.applySignatureConversion(&op.getFunctionBody().front(), sc);

    auto newKernelOp = rewriter.create<frisk::KernelOp>(loc, op.getName(), newFuncType);
    newKernelOp->setAttr("grid", gridAttr);
    rewriter.inlineRegionBefore(op->getRegion(0), newKernelOp.getRegion(), newKernelOp.getRegion().end());
    // 3. replace deepgengraph.return with frisk.end
    auto oldReturn = newKernelOp->getRegion(0).front().getOps<deepgengraph::ReturnOp>().begin();
    rewriter.setInsertionPoint(*oldReturn);
    auto newReturn = rewriter.create<frisk::EndOp>(op->getLoc());
    rewriter.replaceOp(*oldReturn, newReturn);
    
    // 4. insert frisk.parallel
    rewriter.setInsertionPointToStart(&newKernelOp->getRegion(0).front());
    auto ranges = cast<DenseI64ArrayAttr>(gridAttr).asArrayRef();
    auto parallelOp = rewriter.create<frisk::ParallelOp>(loc, ranges, 128);
    auto parallelEntry = parallelOp.addEntryBlock();
    // move all ops expect frisk.end into frisk.parallel
    auto nextOp = parallelOp->getNextNode();
    while (nextOp != nullptr && !isa<frisk::EndOp>(nextOp)) {
      auto *next = nextOp->getNextNode();
      rewriter.moveOpBefore(nextOp, parallelEntry, parallelEntry->end());
      nextOp = next;
    }
    // find frisk.end for frisk.parallel, move it to the block end
    auto innerEndOp = parallelEntry->getOps<frisk::EndOp>().begin();
    rewriter.moveOpBefore(*innerEndOp, parallelEntry, parallelEntry->end());
    // replace gpu.bid with parallel block args
    llvm::SmallVector<gpu::BlockIdOp> bidOps;
    parallelEntry->walk([&](gpu::BlockIdOp bid) { bidOps.push_back(bid); });

    for (auto bidOp : bidOps) {
      int argId = -1;
      switch (bidOp.getDimension()) {
      case gpu::Dimension::x:
        argId = 2;
        break;
      case gpu::Dimension::y:
        argId = 1;
        break;
      case gpu::Dimension::z:
        argId = 0;
        break;
      default:
        assert(false && "unexpected block_id dim");
      }
      rewriter.replaceOp(bidOp, ValueRange{parallelEntry->getArgument(argId)});
    }

    rewriter.replaceOp(op, newKernelOp);
    return success();
  }
};

struct PointerOfConversionPattern : public OpConversionPattern<deepgengraph::triton::PointerOfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::triton::PointerOfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 删除
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto blockArg = getKernelArgById(op, argId);
    rewriter.replaceAllUsesWith(op, blockArg);
    rewriter.eraseOp(op);
    return success();
  }
};

struct BlockPointerOfConversionPattern
    : public OpConversionPattern<deepgengraph::triton::BlockPointerOfOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_ptr_of base=%ptr. 将ptr绕过，直接绑定到 kernelOp的 mem 参数上
  // 新建 frisk.bufferview 建立 入参mem的 view， 替换 result
  // 删除对应的 dg.pointerOfOp

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockPointerOfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto convertedType = dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    if (!convertedType){
      return failure();
    }
    // 
    Value source = adaptor.getBasePointer();
    int argId = -1;
    if (auto argIdAttr = op->getAttrOfType<IntegerAttr>("argId")) {
      argId = argIdAttr.getInt();
      if (Value kernelArg = getKernelArgById(op, argId)){
        source = kernelArg;
      }
    }
    if (!source || !isa<MemRefType>(source.getType())){
      return failure();
    }

    Value baseoffset = adaptor.getBaseOffset();
    std::map<std::string, AffineExpr> dims {}; 
    std::map<int, Value> arglist {};
    auto exprBase = GetExprOfValue(baseoffset, dims, arglist );

    auto memref =  mlir::cast<MemRefType>(adaptor.getBasePointer().getType());
    auto rank = memref.getRank();
    auto offset = op.getOffset();
    auto stride = op.getStride();
    auto order = op.getOrder();
    // auto s0 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), stride[order[0]]);
    // auto s1 = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), stride[order[1]]);
    // 根据 baseOffset 和 stride，计算 base x,y 坐标偏移
    // auto base_x = rewriter.create<arith::DivUIOp>(op->getLoc(), baseoffset, s0);
    // auto base_y = rewriter.create<arith::DivUIOp>(op->getLoc(), baseoffset, s1);
    
    auto base_x = exprBase.floorDiv(stride[order[0]]); 
    auto base_y = exprBase.floorDiv(stride[order[1]]); 
    
    std::vector<AffineExpr> indices = {}; 
    // auto zero = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
    for(int i=0;i<rank-2;++i){
      indices.push_back(mlir::getAffineConstantExpr(0, op->getContext()));
    }
    indices.push_back(base_x);
    indices.push_back(base_y);

    // auto map2dim = AffineMap::getMultiDimIdentityMap(dims.size(), op->getContext());
    std::vector<Value> mapArgs{};
    for(int i = 0;i< arglist.size(); ++i ){
      mapArgs.push_back(arglist[i]);
    }

    auto map2dim = AffineMap::get(dims.size(), 0, indices, op->getContext());
    std::vector<int64_t> ranges = {1,1,1,1};
    ranges[2] = op.getBlockShape()[0];
    ranges[3] = op.getBlockShape()[1];
    auto view = rewriter.create<frisk::BufferViewOp>(op->getLoc(), source, mapArgs, map2dim, ranges);
    view->setAttr("argId", op->getAttr("argId"));
    auto dstMemType = mlir::dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getResult().getType()) ) ;
    auto dstBuffer = rewriter.create<frisk::AllocBufferOp>(op->getLoc(),  dstMemType.getShape(), dstMemType.getElementType(), 16, int(frisk::attr::MemorySpace::Shared));
    dstBuffer->setAttr("argId", op->getAttr("argId"));
    
    auto info = new ArgIdViewBuffer {view, dstBuffer};
    s_map_argId_initBufferView[argId] = info;
  
    rewriter.replaceOp(op, view);
    return success();
  }
};

struct BlockLoadConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_load ：先添加 %mem = frisk.alloc_buffer,
  // 之后 frisk.copy %memview, %mem
  // 使用 %mem 替换 op的结果 
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto _newRetType = mlir::dyn_cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
    auto newRetType = MemRefType::get(_newRetType.getShape(), _newRetType.getElementType(), AffineMap{}, int(frisk::attr::MemorySpace::Shared));
    auto shape = newRetType.getShape();
    auto eleTy = newRetType.getElementType();
    int id = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto buffer = s_map_argId_initBufferView[id]->buffer;
    auto copyOp = rewriter.create<frisk::CopyOp>(op->getLoc(), adaptor.getSrcPointer(), buffer);
    rewriter.replaceOp(op, buffer);
    return success();
  }
};

struct BlockStoreConversionPattern : public OpConversionPattern<deepgengraph::triton::BlockStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  // block_store %14, %24 :  直接替换
  LogicalResult matchAndRewrite(deepgengraph::triton::BlockStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto src = adaptor.getValue();
    auto dst = adaptor.getDstPointer();
    AppendMemspaceToMemrefValue(src, frisk::attr::MemorySpace::Shared);
    AppendMemspaceToMemrefValue(dst, frisk::attr::MemorySpace::Global);
    auto newOp = rewriter.create<frisk::CopyOp>(op->getLoc(), src, dst);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

/**
 * @file DeepgengraphSimplify.cpp
 * @author shilong.xu (shilong.xu@123.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-13
 * view 和 copy 连一起
    block_ptr_of -> alloc_buffer 
    block_load -> view + copy
    block_advance -> erase it. 用来计算 indice
 * @copyright Copyright (c) 2026
 * 
 */

struct BlockAdvanceConversionPattern
    : public OpConversionPattern<deepgengraph::triton::BlockAdvanceOp> {
  using OpConversionPattern::OpConversionPattern;
  // 本质上是将一个 buffer_view 圈出的窗口在baseMem上滑动。滑动距离= offsets. 累计滑动距离需要根据 offsets 和 上一次距离 计算得到
  // 替换为 %next = frisk.buffer_view, scf.yield %next.  将op结果替换为 %next

  LogicalResult matchAndRewrite(deepgengraph::triton::BlockAdvanceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    auto argId = op->getAttrOfType<IntegerAttr>("argId").getInt();
    auto baseMem = getKernelArgById(op, argId);
    
    auto info = s_map_argId_initBufferView[argId];
    auto initView = info->view;
    auto useBuffer = info->buffer;
    // 若其parentOp为forOp，表明窗口会滑动。则需要将 bufferview 的位置关联 forOp的 iv。重建 bufferVIew的位置，并重建其索引map
    // 如果其parent含有多层for，则每层的iv都需要考虑
    mlir::Operation* currOp = op;
    auto indexMap = initView.getIndexMap();
    std::vector<AffineExpr> indexExprs = indexMap.getResults();
    auto dimCount = indexMap.getNumDims();
    
    llvm::outs() << "indexMap = " << indexMap << "\n"; llvm::outs().flush(); 
    
    // 遍历所有parent for，拿到ivs 和 step ，(ivs / step) 为 当前循环次数
    /**
     for(int i=0;i<3;++i){
      for(int j=0;j< bx ; ++j){
        view = someView(arg0);
        blockAdvance(view, offset = (32,128));
        累计循环次数 = i * bx + j
        // offset 计算 ：
        如果 view 的 初始索引为 [0,0, bx*32, by * 512]
        那么 advance 后 view = [0,0, bx*32, by * 512]
      }
     }
     */
    std::vector<Value> ivs;
    std::vector<AffineExpr > loopCountExprs;
    std::vector<AffineExpr> ubs;
    std::vector<ValueRange> ubsOperands;
    
    while(currOp != nullptr){
      if(auto parentLoop = currOp->getParentOfType<affine::AffineForOp>()){
        ivs.push_back(parentLoop.getInductionVar());
        auto step = parentLoop.getStepAsInt();
        auto ubMapExpr = parentLoop.getUpperBoundMap().getResult(0);
        auto ubVals = parentLoop.getUpperBoundOperands();

        auto newDim = mlir::getAffineDimExpr(dimCount, op->getContext());
        newDim = newDim.floorDiv(step);
        loopCountExprs.push_back(newDim);
        ubs.push_back(ubMapExpr);
        ubsOperands.push_back(ubVals);
        
        dimCount++;
        currOp = parentLoop;
      }
      else{
        break;
      }
    }
    // 构建 iv 的expr
    AffineExpr loop_expr = mlir::getAffineConstantExpr(0, op->getContext());
    std::vector<Value> loop_expr_values;

    for(int i=0;i < loopCountExprs.size() ; ++i){
      auto temp = loopCountExprs[i];
      loop_expr_values.push_back(ivs[i]);
      if(i+1 < ubs.size()){
        temp = temp * ubs[i+1];
        for(auto v : ubsOperands[i]){
          loop_expr_values.push_back(v);
        }
      }
      loop_expr = loop_expr + temp;
    }

    auto loc = op->getLoc();
    auto indices = initView.getIndices();
    auto offset = op.getOffsets();
    // affineMap的操作数 value = 原有 + 新收集的ivs
    std::vector<Value> newIndices;
    for(auto v : indices){
      newIndices.push_back(v);
    }
    for(auto v : loop_expr_values){
      newIndices.push_back(v);
    }
    // 表达式重建 ：索引数目不变[x,y,z,w]。xy索引保持，zw 需要加上 (ivs/step * offset)
    std::vector<AffineExpr> newExprs;
    for(int i=0;i < indexExprs.size(); ++i){
      if(i < indexExprs.size() - 2){
        newExprs.push_back(indexExprs[i]);
      }
      else{
        auto id = i-(indexExprs.size() - 2);
        auto newexpr = indexExprs[i] + op.getOffsets()[id] * loop_expr; 
        newExprs.push_back(newexpr);
      }
    }
    // newMap dim增加，symbol不变，expr重建
    auto newMap = AffineMap::get(dimCount, indexMap.getNumSymbols(), newExprs, op->getContext());
    // newView的indices为newMap的操作数
    llvm::outs() << "newMap=" << newMap << "  newIndices.size=" << newIndices.size() << " indices.size() = " << indices.size() <<"\n";llvm::outs().flush();
    auto newView = rewriter.create<BufferViewOp>(loc, initView.getSource(), newIndices, newMap, initView.getRanges());
    rewriter.replaceOp(op, newView);
    return success();
  }
};

struct ZeroOpConversionPattern
    : public OpConversionPattern<dg::ZeroOp> {
  using OpConversionPattern::OpConversionPattern;


  LogicalResult matchAndRewrite(dg::ZeroOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    // %16 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
    auto loc = op->getLoc();
    auto buffer = rewriter.create<frisk::AllocBufferOp>(loc, op.getShape(), op.getElementType(), 16, int(frisk::attr::MemorySpace::Shared));
    mlir::Attribute valueAttr;
    auto eleTy = op.getElementType();
    if(eleTy.isFloat()){
      valueAttr = rewriter.getFloatAttr(eleTy, 0.0);
    }
    else if(eleTy.isInteger()){
      valueAttr = rewriter.getIntegerAttr(eleTy, 0);
    }
    else{
      assert(false);
    }
    rewriter.create<frisk::FillOp>(loc, buffer, valueAttr);
    rewriter.replaceOp(op, buffer);
    return success();
  }
};


struct ConvertOpConversionPattern
    : public OpConversionPattern<dg::ConvertOp> {
  using OpConversionPattern::OpConversionPattern;


  LogicalResult matchAndRewrite(dg::ConvertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override 
  {
    // %16 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
    auto loc = op->getLoc();
    auto operand = adaptor.getOperand();
    AppendMemspaceToMemrefValue( operand , frisk::attr::MemorySpace::Shared);
    auto dstType = adaptor.getDstType();
    ModifyMemrefType(dstType, frisk::attr::MemorySpace::Shared);
    auto newOp = rewriter.create<frisk::ConvertOp>(loc, operand, dstType);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};


struct ForTypeConversionPattern : public OpConversionPattern<affine::AffineForOp> {
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    
    // 1. 用 adaptor 拿到已经被 TypeConverter 转换好的 inits
    //    adaptor.getInits() 在 ConversionPatternRewriter 框架下
    //    已经是转换后的 Value (memref)
    // 需要把 iterarg 中的 bufferview 相关类型直接赋值 <1x1x128x128> 这种类型。通过 copy 明确拷贝 src dst 语法上更清楚
    for(auto e : op.getInits()){
      ;
    }
    SmallVector<Value> newIterArgs(adaptor.getInits());
    
    bool needConvert = false;
    for (auto [oldVal, newVal] : llvm::zip(op.getInits(), newIterArgs)) {
      if (oldVal.getType() != newVal.getType()) {
        needConvert = true;
        break;
      }
    }
    if (!needConvert) return failure();

    // 2. 用转换后的 newIterArgs 创建新 ForOp
    auto newForOp = rewriter.create<affine::AffineForOp>(
        op.getLoc(),
        adaptor.getLowerBoundOperands(), op.getLowerBoundMap(),
        adaptor.getUpperBoundOperands(), op.getUpperBoundMap(),
        op.getStepAsInt(),
        newIterArgs);  // ✅ 关键：传入转换后的 args

    // 3. 构建 SignatureConversion
    //    旧 Block 参数: [iv: index, arg1: BlockPtrType, arg2: ...]
    //    新 Block 参数: [iv: index, arg1: memref, arg2: ...]
    TypeConverter::SignatureConversion sigConv(op.getBody()->getNumArguments());
    
    // IV 不变
    sigConv.addInputs(0, rewriter.getIndexType());
    
    // iter_args: 用新类型替换
    for (unsigned i = 0; i < newIterArgs.size(); ++i) {
      sigConv.addInputs(i + 1, newIterArgs[i].getType()); // ✅ 用 newIterArgs 的类型
    }

    // 4. 移动旧 Region 到新 ForOp，并应用参数类型转换
    rewriter.eraseBlock(newForOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), 
                                 newForOp.getRegion(), 
                                 newForOp.getRegion().end());
    
    // applySignatureConversion 会在 block 入口插入 cast 处理类型不匹配
    if (failed(rewriter.convertRegionTypes(&newForOp.getRegion(), 
                                            *getTypeConverter(), &sigConv))) {
      return failure();
    }

    // 5. 替换旧 Op 的结果
    rewriter.replaceOp(op, newForOp.getResults());
    return success();
  }
};

struct YieldTypeConversionPattern : public OpConversionPattern<affine::AffineYieldOp> {
  using OpConversionPattern<affine::AffineYieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineYieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    bool needConvert = false;
    for(auto v : op->getOperands()){
      if(mlir::isa<dgt::PointerType, dgt::BlockPointerType>(v.getType())){
        needConvert = true;
        break;
      }
    }
    if(!needConvert){
      return failure();
    }
    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(op, adaptor.getOperands());
    return success();
  }
};


// %cst = arith.constant dense<0.127531052> : tensor<1xf32> loc(#loc)
struct ArithTensorConversionPattern : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto retType = op.getResult().getType();
    if(mlir::isa<TensorType>(retType)){
      auto tensorTy = mlir::dyn_cast<TensorType>(retType);
      MemRefType memrefTy = MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{}, int(frisk::attr::MemorySpace::Shared));

      if(!memrefTy){
        return failure();
      }
      auto allocOp = rewriter.create<memref::AllocOp>(op->getLoc(), memrefTy);
      auto val = mlir::cast<DenseFPElementsAttr>(op.getValue());
      float v = 0;
      if(!val){
        return failure();
      }
      auto vals = val.getValues<APFloat>();
      for(auto it : vals){
        v = it.convertToFloat();
      }
      auto constVal = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getF32FloatAttr(v));
      auto zero = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexAttr(0));
      std::vector<Value> indices;
      for(auto dim : memrefTy.getShape()){
        indices.push_back(zero);
      }
      auto newOp = rewriter.create<affine::AffineStoreOp>(op->getLoc(), constVal ,allocOp, indices);
      rewriter.replaceOp(op, allocOp);
      return success();
    }
    else{
      return failure();
    }
  }
};

} // namespace

class ConvertKernelOpToFrisk : public impl::KernelOpToFriskBase<ConvertKernelOpToFrisk> {
public:
  void runOnOperation() override {
    auto *ctx = getOperation()->getContext();
    Operation *op = getOperation();

    TypeConverter tc;
    tc.addConversion([](Type type) { return type; });
    tc.addConversion([](TensorType tensorTy) {
      return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
    });
    tc.addConversion([](deepgengraph::triton::PointerType ptrType) { return convertPointerType(ptrType); });
    tc.addConversion(
        [](deepgengraph::triton::BlockPointerType blockPtrType) { return convertBlockPointerType(blockPtrType); });
    addMaterializations(tc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<FriskDialect, memref::MemRefDialect, func::FuncDialect, deepgengraph::DeepgengraphDialect,
                           deepgengraph::triton::DeepgengraphTritonDialect, arith::ArithDialect, 
                           tensor::TensorDialect>();
    target.addIllegalOp<deepgengraph::KernelOp>();

    RewritePatternSet ps(ctx);
    ps.add<KernelOpConversionPattern>(tc, ctx);

    if (failed(applyPartialConversion(op, target, std::move(ps)))) {
      signalPassFailure();
    }
  }
};

class ConvertMemOpToFrisk : public impl::MemOpToFriskBase<ConvertMemOpToFrisk> {
public:
  void runOnOperation() override {
    auto *ctx = getOperation()->getContext();
    Operation *op = getOperation();

    TypeConverter tc;
    // typeconversion rules :
    // tensor -> memref ; dgt.ptr -> memref ; dgt.block_ptr -> memref
    tc.addConversion([](Type type) { return type; });
    tc.addConversion([](deepgengraph::triton::PointerType ptrType) { 
      auto tensorTy = ptrType.getPointeeType();
      return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{});
    });
    tc.addConversion(
    [](deepgengraph::triton::BlockPointerType blockPtrType) { 
      auto tensorTy = blockPtrType.getPointeeType();
      return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), AffineMap{});
    });
    tc.addConversion([](TensorType ty){
      return MemRefType::get(ty.getShape(), ty.getElementType());
    });
    addMaterializations(tc);

    ConversionTarget target(*ctx);
    target.addLegalDialect<FriskDialect, memref::MemRefDialect, func::FuncDialect, deepgengraph::DeepgengraphDialect,
                           deepgengraph::triton::DeepgengraphTritonDialect, arith::ArithDialect, scf::SCFDialect, affine::AffineDialect,
                           tensor::TensorDialect>();


    // stage 1 : 转化指针定义op -> memref buffer
    ConversionTarget t0 = target;
    t0.addIllegalOp<dgt::PointerOfOp, dgt::BlockPointerOfOp>();

    RewritePatternSet ps0(ctx);
    ps0.add<PointerOfConversionPattern, BlockPointerOfConversionPattern>(tc, ctx);
    applyPartialConversion(op, t0, std::move(ps0));
    
    // stage 2 : 指针读写op -> memref 读写
    RewritePatternSet ps1(ctx);
    ps1.add<BlockLoadConversionPattern,
      BlockStoreConversionPattern,ZeroOpConversionPattern ,ConvertOpConversionPattern,
      BlockAdvanceConversionPattern, ForTypeConversionPattern, YieldTypeConversionPattern
    >(tc, ctx);
    ConversionTarget t1 = target;
    t1.addIllegalOp<dgt::PointerOfOp, dgt::BlockPointerOfOp,
      dgt::BlockLoadOp, dgt::BlockStoreOp,
      dgt::TensorFromOp, dg::ZeroOp, dg::ConvertOp,
      dgt::BlockAdvanceOp >();
    t1.addDynamicallyLegalOp<affine::AffineForOp>([](affine::AffineForOp forOp) {
      for (Value initArg : forOp.getInits()) {
        if (isTritonPointerLike(initArg.getType())){
          return false;
        }
      }
      for (Type resultType : forOp.getResultTypes()) {
        if (isTritonPointerLike(resultType)){
          return false;
        }
      }
      return true;
    });

    t1.addDynamicallyLegalOp<affine::AffineYieldOp>([](affine::AffineYieldOp yieldOp) {
      for (Value operand : yieldOp.getOperands()) {
        if (isTritonPointerLike(operand.getType())){
          return false;
        }
      }
      return true;
    });

    applyPartialConversion(op, t1, std::move(ps1));

    
    // stage 3 ：constant 分配的tensor 改为 分配memref
    ConversionTarget t2(*ctx);
    t2.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op){
      return !mlir::isa<TensorType>(op.getResult().getType());
    });
    t2.markUnknownOpDynamicallyLegal([](mlir::Operation* op){return true;});
    RewritePatternSet p2(ctx);
    p2.add<ArithTensorConversionPattern>(tc,ctx);
    applyPartialConversion(op, t2, std::move(p2));
  }
};




struct SCFForToAffineFor : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {

    std::map<std::string, AffineExpr> dims_lb; 
    std::map<int, Value> arglist_lb;
    std::map<std::string, AffineExpr> dims_ub; 
    std::map<int, Value> arglist_ub;
    
    auto lbexpr = GetExprOfValue(op.getLowerBound(), dims_lb, arglist_lb);
    auto ubexpr = GetExprOfValue(op.getUpperBound(), dims_ub, arglist_ub);
    
    std::vector<Value> lbvr, ubvr;
    for(int i=0;i<arglist_lb.size();++i){
      lbvr.push_back(arglist_lb[i]);
    }
    for(int i=0;i<arglist_ub.size();++i){
      ubvr.push_back(arglist_ub[i]);
    }
    
    auto lbMap = AffineMap::get(dims_lb.size(), 0, lbexpr);
    auto ubMap = AffineMap::get(dims_ub.size(), 0, ubexpr);

    int stepNum;
    auto stepOp = op.getStep().getDefiningOp<arith::ConstantOp>();
    if(stepOp){
      stepNum = mlir::dyn_cast<IntegerAttr>(stepOp.getValue()).getInt();
    }
    
    auto affineFor = rewriter.create<affine::AffineForOp>(op->getLoc(), lbvr, lbMap, ubvr, ubMap, stepNum, op.getInitArgs());
        
    rewriter.inlineRegionBefore(op.getRegion(), affineFor.getRegion(), affineFor.getRegion().end());
    Block* contentBlock = &affineFor->getRegion(0).back();
    Block* entryBlock = &affineFor->getRegion(0).front();
    rewriter.mergeBlocks(contentBlock, entryBlock, entryBlock->getArguments());
    // 5. 对移入的 Block 执行“签名转换 (Signature Conversion)”
    // 这一步是让 MLIR 框架安全地将 Block 参数从 block_ptr 转换成 memref，
    // 并且会自动在内部插入 "unrealized_conversion_cast"，保证内部尚未被转换的 block_load 不会因为类型校验崩溃！
    TypeConverter::SignatureConversion sigConversion(affineFor.getBody()->getNumArguments());
    
    // 第 0 个参数是归纳变量 (Induction Variable)，保持为 index 类型
    sigConversion.addInputs(0, rewriter.getIndexType());
    
    // 剩下的参数是 iter_args，转换为 adaptor 中对应的已转换类型
    for (auto [idx, arg] : llvm::enumerate(op.getInitArgs())) {
      sigConversion.addInputs(idx + 1, arg.getType());
    }
    
    // 应用签名转换
    
    rewriter.applySignatureConversion(&affineFor.getRegion().front(), sigConversion, nullptr);

    // 6. 替换 Op (Yield 的替换可以交给独立的 SCFYieldTypeConversionPattern 处理)
    auto oldTerm = affineFor.getBody()->getTerminator();
    rewriter.setInsertionPoint(oldTerm);
    auto newTerm = rewriter.create<affine::AffineYieldOp>(op->getLoc(), oldTerm->getOperands());
    rewriter.replaceOp(oldTerm, newTerm);
    rewriter.replaceOp(op, affineFor);
    return success();
  }
};


// 定义 Pass，继承自 OperationPass 并且作用于 func::FuncOp
struct ConvertSCFForToAffineForPass 
    : public PassWrapper<ConvertSCFForToAffineForPass, OperationPass<deepgengraph::KernelOp>> {
    
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertSCFForToAffineForPass)

    StringRef getArgument() const final { return "add-tensor-memspace"; }
    StringRef getDescription() const final { return "Add memspace encoding to tensors based on their position."; }

    void runOnOperation() override {
      auto ctx = getOperation()->getContext();
      RewritePatternSet ps(ctx);

      ps.add<SCFForToAffineFor>(ctx);
      ConversionTarget tar(*ctx);
      tar.addIllegalOp<scf::ForOp>();
      tar.markUnknownOpDynamicallyLegal([](mlir::Operation* op){return true;});

      applyPartialConversion(getOperation(), tar, std::move(ps));

    }
};

std::unique_ptr<Pass> createConvertScfForOpPass() {
  return std::make_unique<ConvertSCFForToAffineForPass>();
}

std::unique_ptr<Pass> createConvertKernelOpToFriskPass() {
  return std::make_unique<ConvertKernelOpToFrisk>();
}

std::unique_ptr<Pass> createConvertMemOpPass() {
  return std::make_unique<ConvertMemOpToFrisk>();
}




} // namespace mlir::frisk
