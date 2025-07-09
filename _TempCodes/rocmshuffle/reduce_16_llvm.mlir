module {
  llvm.func @reduce(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes { rocdl.kernel, func.block.dim = array<i32: 64>, func.grid.dim = array<i32: 2> , rocdl.flat_work_group_size = "64,64", rocdl.reqd_work_group_size = array<i32: 64, 1, 1> } 
  {
    %cn1 = llvm.mlir.constant(-1 : i32) : i32
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c32 = llvm.mlir.constant(32 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %i4 = llvm.mlir.constant(4 : index) : i32
    %in1 = llvm.mlir.constant(-1 : index) : i32
    %i16 = llvm.mlir.constant(16 : index) : i32
    %i0 = llvm.mlir.constant(0 : index) : i32
    %bx = rocdl.workgroup.id.x : i32
    %tx = rocdl.workitem.id.x : i32
    %13 = llvm.mul %bx, %i4 overflow<nsw> : i32  // bx*4
    %14 = llvm.icmp "slt" %tx, %i0 : i32  // (tx<0)
    %15 = llvm.sub %in1, %tx : i32  // -1-tx
    %16 = llvm.select %14, %15, %tx : i1, i32  // tx
    %17 = llvm.sdiv %16, %i16 : i32  // tx/16
    %18 = llvm.sub %in1, %17 : i32  // -1-tx/16
    %19 = llvm.select %14, %18, %17 : i1, i32  // tx/16
    %20 = llvm.add %13, %19 : i32  // bx*4 + tx/16
    %21 = llvm.srem %tx, %i16 : i32  // tx % 16
    %22 = llvm.icmp "slt" %21, %i0 : i32  // tx % 16 < 0
    %23 = llvm.add %21, %i16 : i32  // tx % 16 + 16
    %24 = llvm.select %22, %23, %21 : i1, i32  // tx % 16
    %25 = llvm.mul %20, %i16 : i32  // (bx*4 + tx/16) * 16
    %correctLoc = llvm.add %25, %24 : i32  // (bx*4 + tx/16) * 16 +(tx % 16)
    %27 = llvm.getelementptr %arg0[%correctLoc] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %28 = llvm.load %27 : !llvm.ptr<1> -> f32
    %29 = rocdl.mbcnt.lo %cn1, %c0 : (i32, i32) -> i32
    %30 = rocdl.mbcnt.hi %cn1, %29 : (i32, i32) -> i32  // laneid
    %31 = llvm.sub %c0, %c32 : i32
    %32 = llvm.add %30, %c32 : i32
    %33 = llvm.and %32, %31 : i32
    %34 = llvm.add %30, %c1 : i32
    %35 = llvm.icmp "slt" %34, %33 : i32
    %36 = llvm.select %35, %34, %30 : i1, i32
    %37 = llvm.shl %36, %c2 : i32
    %38 = llvm.bitcast %28 : f32 to i32
    %39 = rocdl.ds_bpermute %37, %38 : (i32, i32) -> i32
    %40 = llvm.bitcast %39 : i32 to f32
    %41 = llvm.fadd %28, %40 : f32
    %42 = llvm.add %30, %c2 : i32
    %43 = llvm.icmp "slt" %42, %33 : i32
    %44 = llvm.select %43, %42, %30 : i1, i32
    %45 = llvm.shl %44, %c2 : i32
    %46 = llvm.bitcast %41 : f32 to i32
    %47 = rocdl.ds_bpermute %45, %46 : (i32, i32) -> i32
    %48 = llvm.bitcast %47 : i32 to f32
    %49 = llvm.fadd %41, %48 : f32
    %50 = llvm.add %30, %c4 : i32
    %51 = llvm.icmp "slt" %50, %33 : i32
    %52 = llvm.select %51, %50, %30 : i1, i32
    %53 = llvm.shl %52, %c2 : i32
    %54 = llvm.bitcast %49 : f32 to i32
    %55 = rocdl.ds_bpermute %53, %54 : (i32, i32) -> i32
    %56 = llvm.bitcast %55 : i32 to f32
    %57 = llvm.fadd %49, %56 : f32
    %58 = llvm.add %30, %c8 : i32
    %59 = llvm.icmp "slt" %58, %33 : i32
    %60 = llvm.select %59, %58, %30 : i1, i32
    %61 = llvm.shl %60, %c2 : i32
    %62 = llvm.bitcast %57 : f32 to i32
    %63 = rocdl.ds_bpermute %61, %62 : (i32, i32) -> i32
    %64 = llvm.bitcast %63 : i32 to f32
    %65 = llvm.fadd %57, %64 : f32
    %66 = llvm.getelementptr %arg1[%correctLoc] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    llvm.store %65, %66 : f32, !llvm.ptr<1>
    llvm.return
  }
}