
; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx906
; ModuleID = './reduce_16.cpp'
source_filename = "./reduce_16.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: argmemonly convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @_Z6reducePfS_(float addrspace(1)* nocapture readonly %0, float addrspace(1)* nocapture writeonly %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %5 = shl nsw i32 %3, 6
  %6 = add nsw i32 %5, %4
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds float, float addrspace(1)* %0, i64 %7
  %9 = load float, float addrspace(1)* %8, align 4, !tbaa !5, !amdgpu.noclobber !9
  %10 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %11 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %10)
  %12 = and i32 %11, 63
  %13 = bitcast float %9 to i32
  %14 = icmp ne i32 %12, 63
  %15 = zext i1 %14 to i32
  %16 = add i32 %11, %15
  %17 = shl i32 %16, 2
  %18 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %13)
  %19 = bitcast i32 %18 to float
  %20 = fadd contract float %9, %19
  %21 = bitcast float %20 to i32
  %22 = icmp ugt i32 %12, 61
  %23 = select i1 %22, i32 0, i32 2
  %24 = add i32 %23, %11
  %25 = shl i32 %24, 2
  %26 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %25, i32 %21)
  %27 = bitcast i32 %26 to float
  %28 = fadd contract float %20, %27
  %29 = bitcast float %28 to i32
  %30 = icmp ugt i32 %12, 59
  %31 = select i1 %30, i32 0, i32 4
  %32 = add i32 %31, %11
  %33 = shl i32 %32, 2
  %34 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %33, i32 %29)
  %35 = bitcast i32 %34 to float
  %36 = fadd contract float %28, %35
  %37 = bitcast float %36 to i32
  %38 = icmp ugt i32 %12, 55
  %39 = select i1 %38, i32 0, i32 8
  %40 = add i32 %39, %11
  %41 = shl i32 %40, 2
  %42 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %41, i32 %37)
  %43 = bitcast i32 %42 to float
  %44 = fadd contract float %36, %43
  %45 = getelementptr inbounds float, float addrspace(1)* %1, i64 %7
  store float %44, float addrspace(1)* %45, align 4, !tbaa !5
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #1

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #3

attributes #0 = { argmemonly convergent mustprogress nofree norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+sramecc" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn }
attributes #2 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #3 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 1}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0"}
!4 = !{i32 0, i32 1024}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{}

; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx906

; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx926
; ModuleID = './reduce_16.cpp'
source_filename = "./reduce_16.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: argmemonly convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @_Z6reducePfS_(float addrspace(1)* nocapture readonly %0, float addrspace(1)* nocapture writeonly %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %5 = shl nsw i32 %3, 6
  %6 = add nsw i32 %5, %4
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds float, float addrspace(1)* %0, i64 %7
  %9 = load float, float addrspace(1)* %8, align 4, !tbaa !5, !amdgpu.noclobber !9
  %10 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %11 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %10)
  %12 = and i32 %11, 63
  %13 = bitcast float %9 to i32
  %14 = icmp ne i32 %12, 63
  %15 = zext i1 %14 to i32
  %16 = add i32 %11, %15
  %17 = shl i32 %16, 2
  %18 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %13)
  %19 = bitcast i32 %18 to float
  %20 = fadd contract float %9, %19
  %21 = bitcast float %20 to i32
  %22 = icmp ugt i32 %12, 61
  %23 = select i1 %22, i32 0, i32 2
  %24 = add i32 %23, %11
  %25 = shl i32 %24, 2
  %26 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %25, i32 %21)
  %27 = bitcast i32 %26 to float
  %28 = fadd contract float %20, %27
  %29 = bitcast float %28 to i32
  %30 = icmp ugt i32 %12, 59
  %31 = select i1 %30, i32 0, i32 4
  %32 = add i32 %31, %11
  %33 = shl i32 %32, 2
  %34 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %33, i32 %29)
  %35 = bitcast i32 %34 to float
  %36 = fadd contract float %28, %35
  %37 = bitcast float %36 to i32
  %38 = icmp ugt i32 %12, 55
  %39 = select i1 %38, i32 0, i32 8
  %40 = add i32 %39, %11
  %41 = shl i32 %40, 2
  %42 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %41, i32 %37)
  %43 = bitcast i32 %42 to float
  %44 = fadd contract float %36, %43
  %45 = getelementptr inbounds float, float addrspace(1)* %1, i64 %7
  store float %44, float addrspace(1)* %45, align 4, !tbaa !5
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #1

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #3

attributes #0 = { argmemonly convergent mustprogress nofree norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx926" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+gfx926-insts,+mmop-insts,+mmop64-insts,+s-memrealtime,+s-memtime-inst,+sramecc" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn }
attributes #2 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #3 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 1}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0"}
!4 = !{i32 0, i32 1024}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{}

; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx926

; __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa-gfx928
; ModuleID = './reduce_16.cpp'
source_filename = "./reduce_16.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

@softfloat_countLeadingZeros8 = internal unnamed_addr addrspace(1) constant <{ [128 x i8], [128 x i8] }> <{ [128 x i8] c"\08\07\06\06\05\05\05\05\04\04\04\04\04\04\04\04\03\03\03\03\03\03\03\03\03\03\03\03\03\03\03\03\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\02\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01\01", [128 x i8] zeroinitializer }>, align 1
@llvm.compiler.used = appending addrspace(1) global [4 x i8*] [i8* bitcast (double (double, double)* @__softfloat_f64_add to i8*), i8* bitcast (double (double, double)* @__softfloat_f64_div to i8*), i8* bitcast (double (double, double)* @__softfloat_f64_mul to i8*), i8* bitcast (double (double, double)* @__softfloat_f64_sub to i8*)], section "llvm.metadata"

; Function Attrs: argmemonly convergent mustprogress nofree norecurse nounwind
define protected amdgpu_kernel void @_Z6reducePfS_(float addrspace(1)* nocapture readonly %0, float addrspace(1)* nocapture writeonly %1) local_unnamed_addr #0 {
  %3 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %4 = tail call i32 @llvm.amdgcn.workitem.id.x(), !range !4
  %5 = shl nsw i32 %3, 6
  %6 = add nsw i32 %5, %4
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds float, float addrspace(1)* %0, i64 %7
  %9 = load float, float addrspace(1)* %8, align 4, !tbaa !5, !amdgpu.noclobber !9
  %10 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %11 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %10)
  %12 = and i32 %11, 63
  %13 = bitcast float %9 to i32
  %14 = icmp ne i32 %12, 63
  %15 = zext i1 %14 to i32
  %16 = add i32 %11, %15
  %17 = shl i32 %16, 2
  %18 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %17, i32 %13)
  %19 = bitcast i32 %18 to float
  %20 = fadd contract float %9, %19
  %21 = bitcast float %20 to i32
  %22 = icmp ugt i32 %12, 61
  %23 = select i1 %22, i32 0, i32 2
  %24 = add i32 %23, %11
  %25 = shl i32 %24, 2
  %26 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %25, i32 %21)
  %27 = bitcast i32 %26 to float
  %28 = fadd contract float %20, %27
  %29 = bitcast float %28 to i32
  %30 = icmp ugt i32 %12, 59
  %31 = select i1 %30, i32 0, i32 4
  %32 = add i32 %31, %11
  %33 = shl i32 %32, 2
  %34 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %33, i32 %29)
  %35 = bitcast i32 %34 to float
  %36 = fadd contract float %28, %35
  %37 = bitcast float %36 to i32
  %38 = icmp ugt i32 %12, 55
  %39 = select i1 %38, i32 0, i32 8
  %40 = add i32 %39, %11
  %41 = shl i32 %40, 2
  %42 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %41, i32 %37)
  %43 = bitcast i32 %42 to float
  %44 = fadd contract float %36, %43
  %45 = getelementptr inbounds float, float addrspace(1)* %1, i64 %7
  store float %44, float addrspace(1)* %45, align 4, !tbaa !5
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #1

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind readnone willreturn
define internal double @__softfloat_f64_add(double noundef %0, double noundef %1) #3 {
  %3 = bitcast double %0 to i64
  %4 = bitcast double %1 to i64
  %5 = icmp slt i64 %3, 0
  %6 = lshr i64 %3, 63
  %7 = trunc i64 %6 to i32
  %8 = lshr i64 %4, 63
  %9 = trunc i64 %8 to i32
  %10 = icmp eq i32 %7, %9
  %11 = lshr i64 %3, 52
  %12 = and i64 %11, 2047
  %13 = and i64 %3, 4503599627370495
  %14 = lshr i64 %4, 52
  %15 = and i64 %14, 2047
  %16 = and i64 %4, 4503599627370495
  %17 = sub nsw i64 %12, %15
  %18 = icmp eq i64 %12, %15
  br i1 %10, label %19, label %129

19:                                               ; preds = %2
  br i1 %18, label %20, label %30

20:                                               ; preds = %19
  switch i64 %12, label %26 [
    i64 0, label %21
    i64 2047, label %23
  ]

21:                                               ; preds = %20
  %22 = add i64 %16, %3
  br label %326

23:                                               ; preds = %20
  %24 = or i64 %16, %13
  %25 = icmp eq i64 %24, 0
  br i1 %25, label %326, label %123

26:                                               ; preds = %20
  %27 = or i64 %13, 9007199254740992
  %28 = add nuw nsw i64 %27, %16
  %29 = shl nuw nsw i64 %28, 9
  br label %89

30:                                               ; preds = %19
  %31 = shl nuw nsw i64 %13, 9
  %32 = shl nuw nsw i64 %16, 9
  %33 = icmp slt i64 %17, 0
  br i1 %33, label %34, label %57

34:                                               ; preds = %30
  %35 = icmp eq i64 %15, 2047
  br i1 %35, label %36, label %40

36:                                               ; preds = %34
  %37 = icmp eq i64 %16, 0
  br i1 %37, label %38, label %123

38:                                               ; preds = %36
  %39 = select i1 %5, i64 -4503599627370496, i64 9218868437227405312
  br label %326

40:                                               ; preds = %34
  %41 = icmp eq i64 %12, 0
  %42 = add nuw nsw i64 %31, 2305843009213693952
  %43 = shl nuw nsw i64 %13, 10
  %44 = select i1 %41, i64 %43, i64 %42
  %45 = sub nsw i64 0, %17
  %46 = icmp ult i64 %45, 63
  br i1 %46, label %47, label %54

47:                                               ; preds = %40
  %48 = lshr i64 %44, %45
  %49 = and i64 %17, 63
  %50 = shl i64 %44, %49
  %51 = icmp ne i64 %50, 0
  %52 = zext i1 %51 to i64
  %53 = or i64 %48, %52
  br label %78

54:                                               ; preds = %40
  %55 = icmp ne i64 %44, 0
  %56 = zext i1 %55 to i64
  br label %78

57:                                               ; preds = %30
  %58 = icmp eq i64 %12, 2047
  br i1 %58, label %59, label %61

59:                                               ; preds = %57
  %60 = icmp eq i64 %13, 0
  br i1 %60, label %326, label %123

61:                                               ; preds = %57
  %62 = icmp eq i64 %15, 0
  %63 = add nuw nsw i64 %32, 2305843009213693952
  %64 = shl nuw nsw i64 %16, 10
  %65 = select i1 %62, i64 %64, i64 %63
  %66 = icmp ult i64 %17, 63
  br i1 %66, label %67, label %75

67:                                               ; preds = %61
  %68 = lshr i64 %65, %17
  %69 = sub nsw i64 0, %17
  %70 = and i64 %69, 63
  %71 = shl i64 %65, %70
  %72 = icmp ne i64 %71, 0
  %73 = zext i1 %72 to i64
  %74 = or i64 %68, %73
  br label %78

75:                                               ; preds = %61
  %76 = icmp ne i64 %65, 0
  %77 = zext i1 %76 to i64
  br label %78

78:                                               ; preds = %75, %67, %54, %47
  %79 = phi i64 [ %32, %47 ], [ %32, %54 ], [ %74, %67 ], [ %77, %75 ]
  %80 = phi i64 [ %53, %47 ], [ %56, %54 ], [ %31, %67 ], [ %31, %75 ]
  %81 = phi i64 [ %15, %47 ], [ %15, %54 ], [ %12, %67 ], [ %12, %75 ]
  %82 = add nuw nsw i64 %79, 2305843009213693952
  %83 = add nuw nsw i64 %82, %80
  %84 = icmp ult i64 %83, 4611686018427387904
  %85 = sext i1 %84 to i64
  %86 = add nsw i64 %81, %85
  %87 = zext i1 %84 to i64
  %88 = shl nuw i64 %83, %87
  br label %89

89:                                               ; preds = %78, %26
  %90 = phi i64 [ %12, %26 ], [ %86, %78 ]
  %91 = phi i64 [ %29, %26 ], [ %88, %78 ]
  %92 = and i64 %90, 65535
  %93 = icmp ugt i64 %92, 2044
  br i1 %93, label %94, label %107

94:                                               ; preds = %89
  %95 = icmp slt i64 %90, 0
  br i1 %95, label %96, label %100

96:                                               ; preds = %94
  %97 = lshr i64 %91, 1
  %98 = and i64 %91, 1
  %99 = or i64 %97, %98
  br label %107

100:                                              ; preds = %94
  %101 = icmp ugt i64 %90, 2045
  %102 = add i64 %91, 512
  %103 = icmp slt i64 %102, 0
  %104 = or i1 %101, %103
  br i1 %104, label %105, label %107

105:                                              ; preds = %100
  %106 = select i1 %5, i64 -4503599627370496, i64 9218868437227405312
  br label %326

107:                                              ; preds = %100, %96, %89
  %108 = phi i64 [ %91, %89 ], [ %91, %100 ], [ %99, %96 ]
  %109 = phi i64 [ %90, %89 ], [ 2045, %100 ], [ 0, %96 ]
  %110 = and i64 %108, 1023
  %111 = add i64 %108, 512
  %112 = lshr i64 %111, 10
  %113 = icmp eq i64 %110, 512
  %114 = zext i1 %113 to i64
  %115 = xor i64 %114, -1
  %116 = and i64 %112, %115
  %117 = icmp eq i64 %116, 0
  %118 = and i64 %3, -9223372036854775808
  %119 = shl nsw i64 %109, 52
  %120 = select i1 %117, i64 0, i64 %119
  %121 = or i64 %116, %118
  %122 = add i64 %121, %120
  br label %326

123:                                              ; preds = %59, %36, %23
  %124 = and i64 %3, 9218868437227405312
  %125 = icmp ne i64 %124, 9218868437227405312
  %126 = icmp eq i64 %13, 0
  %127 = or i1 %125, %126
  %128 = select i1 %127, i64 %4, i64 %3
  br label %326

129:                                              ; preds = %2
  br i1 %18, label %130, label %177

130:                                              ; preds = %129
  %131 = icmp eq i64 %12, 2047
  br i1 %131, label %132, label %135

132:                                              ; preds = %130
  %133 = or i64 %16, %13
  %134 = icmp eq i64 %133, 0
  br i1 %134, label %326, label %320

135:                                              ; preds = %130
  %136 = sub nsw i64 %13, %16
  %137 = icmp eq i64 %13, %16
  br i1 %137, label %326, label %138

138:                                              ; preds = %135
  %139 = tail call i64 @llvm.usub.sat.i64(i64 %12, i64 1)
  %140 = xor i64 %136, %3
  %141 = tail call i64 @llvm.abs.i64(i64 %136, i1 true)
  %142 = lshr i64 %141, 32
  %143 = trunc i64 %142 to i32
  %144 = icmp eq i32 %143, 0
  %145 = trunc i64 %141 to i32
  %146 = select i1 %144, i8 32, i8 0
  %147 = select i1 %144, i32 %145, i32 %143
  %148 = icmp ult i32 %147, 65536
  %149 = or i8 %146, 16
  %150 = shl i32 %147, 16
  %151 = select i1 %148, i8 %149, i8 %146
  %152 = select i1 %148, i32 %150, i32 %147
  %153 = icmp ult i32 %152, 16777216
  %154 = or i8 %151, 8
  %155 = shl i32 %152, 8
  %156 = select i1 %153, i8 %154, i8 %151
  %157 = select i1 %153, i32 %155, i32 %152
  %158 = lshr i32 %157, 24
  %159 = zext i32 %158 to i64
  %160 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %159
  %161 = load i8, i8 addrspace(1)* %160, align 1, !tbaa !10
  %162 = add nsw i8 %156, -11
  %163 = add i8 %162, %161
  %164 = sext i8 %163 to i64
  %165 = sub nsw i64 %139, %164
  %166 = icmp slt i64 %165, 0
  %167 = trunc i64 %139 to i8
  %168 = select i1 %166, i8 %167, i8 %163
  %169 = tail call i64 @llvm.smax.i64(i64 %165, i64 0)
  %170 = and i64 %140, -9223372036854775808
  %171 = shl nuw i64 %169, 52
  %172 = add i64 %171, %170
  %173 = and i8 %168, 63
  %174 = zext i8 %173 to i64
  %175 = shl i64 %141, %174
  %176 = add i64 %172, %175
  br label %326

177:                                              ; preds = %129
  %178 = shl nuw nsw i64 %13, 10
  %179 = shl nuw nsw i64 %16, 10
  %180 = icmp slt i64 %17, 0
  br i1 %180, label %181, label %207

181:                                              ; preds = %177
  %182 = icmp eq i64 %15, 2047
  br i1 %182, label %183, label %187

183:                                              ; preds = %181
  %184 = icmp eq i64 %16, 0
  br i1 %184, label %185, label %320

185:                                              ; preds = %183
  %186 = select i1 %5, i64 9218868437227405312, i64 -4503599627370496
  br label %326

187:                                              ; preds = %181
  %188 = xor i1 %5, true
  %189 = icmp eq i64 %12, 0
  %190 = select i1 %189, i64 %178, i64 4611686018427387904
  %191 = add nuw nsw i64 %190, %178
  %192 = sub nsw i64 0, %17
  %193 = icmp ult i64 %192, 63
  br i1 %193, label %194, label %201

194:                                              ; preds = %187
  %195 = lshr i64 %191, %192
  %196 = and i64 %17, 63
  %197 = shl i64 %191, %196
  %198 = icmp ne i64 %197, 0
  %199 = zext i1 %198 to i64
  %200 = or i64 %195, %199
  br label %204

201:                                              ; preds = %187
  %202 = icmp ne i64 %191, 0
  %203 = zext i1 %202 to i64
  br label %204

204:                                              ; preds = %201, %194
  %205 = phi i64 [ %200, %194 ], [ %203, %201 ]
  %206 = sub nsw i64 %179, %205
  br label %230

207:                                              ; preds = %177
  %208 = icmp eq i64 %12, 2047
  br i1 %208, label %209, label %211

209:                                              ; preds = %207
  %210 = icmp eq i64 %13, 0
  br i1 %210, label %326, label %320

211:                                              ; preds = %207
  %212 = icmp eq i64 %15, 0
  %213 = select i1 %212, i64 %179, i64 4611686018427387904
  %214 = add nuw nsw i64 %213, %179
  %215 = icmp ult i64 %17, 63
  br i1 %215, label %216, label %224

216:                                              ; preds = %211
  %217 = lshr i64 %214, %17
  %218 = sub nsw i64 0, %17
  %219 = and i64 %218, 63
  %220 = shl i64 %214, %219
  %221 = icmp ne i64 %220, 0
  %222 = zext i1 %221 to i64
  %223 = or i64 %217, %222
  br label %227

224:                                              ; preds = %211
  %225 = icmp ne i64 %214, 0
  %226 = zext i1 %225 to i64
  br label %227

227:                                              ; preds = %224, %216
  %228 = phi i64 [ %223, %216 ], [ %226, %224 ]
  %229 = sub nsw i64 %178, %228
  br label %230

230:                                              ; preds = %227, %204
  %231 = phi i1 [ %188, %204 ], [ %5, %227 ]
  %232 = phi i64 [ %15, %204 ], [ %12, %227 ]
  %233 = phi i64 [ %206, %204 ], [ %229, %227 ]
  %234 = add nsw i64 %233, 4611686018427387904
  %235 = lshr i64 %234, 32
  %236 = trunc i64 %235 to i32
  %237 = icmp eq i32 %236, 0
  %238 = trunc i64 %233 to i32
  %239 = select i1 %237, i8 32, i8 0
  %240 = select i1 %237, i32 %238, i32 %236
  %241 = icmp ult i32 %240, 65536
  %242 = or i8 %239, 16
  %243 = shl i32 %240, 16
  %244 = select i1 %241, i8 %242, i8 %239
  %245 = select i1 %241, i32 %243, i32 %240
  %246 = icmp ult i32 %245, 16777216
  %247 = or i8 %244, 8
  %248 = shl i32 %245, 8
  %249 = select i1 %246, i8 %247, i8 %244
  %250 = select i1 %246, i32 %248, i32 %245
  %251 = lshr i32 %250, 24
  %252 = zext i32 %251 to i64
  %253 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %252
  %254 = load i8, i8 addrspace(1)* %253, align 1, !tbaa !10
  %255 = add i8 %254, -1
  %256 = add i8 %255, %249
  %257 = xor i8 %256, -1
  %258 = sext i8 %257 to i64
  %259 = add nsw i64 %232, %258
  %260 = zext i8 %256 to i32
  %261 = icmp sgt i8 %256, 9
  %262 = trunc i64 %259 to i32
  %263 = icmp ult i32 %262, 2045
  %264 = select i1 %261, i1 %263, i1 false
  br i1 %264, label %265, label %276

265:                                              ; preds = %230
  %266 = select i1 %231, i64 -9223372036854775808, i64 0
  %267 = icmp eq i64 %234, 0
  %268 = shl i64 %259, 52
  %269 = select i1 %267, i64 0, i64 %268
  %270 = add i64 %269, %266
  %271 = add nuw nsw i32 %260, 54
  %272 = and i32 %271, 63
  %273 = zext i32 %272 to i64
  %274 = shl i64 %234, %273
  %275 = add i64 %270, %274
  br label %326

276:                                              ; preds = %230
  %277 = and i32 %260, 63
  %278 = zext i32 %277 to i64
  %279 = shl i64 %234, %278
  %280 = and i64 %259, 65535
  %281 = icmp ugt i64 %280, 2044
  br i1 %281, label %282, label %304

282:                                              ; preds = %276
  %283 = icmp slt i64 %259, 0
  br i1 %283, label %284, label %297

284:                                              ; preds = %282
  %285 = sub nsw i64 0, %259
  %286 = icmp ult i64 %285, 63
  br i1 %286, label %287, label %294

287:                                              ; preds = %284
  %288 = lshr i64 %279, %285
  %289 = and i64 %259, 63
  %290 = shl i64 %279, %289
  %291 = icmp ne i64 %290, 0
  %292 = zext i1 %291 to i64
  %293 = or i64 %288, %292
  br label %304

294:                                              ; preds = %284
  %295 = icmp ne i64 %279, 0
  %296 = zext i1 %295 to i64
  br label %304

297:                                              ; preds = %282
  %298 = icmp ugt i64 %259, 2045
  %299 = add i64 %279, 512
  %300 = icmp slt i64 %299, 0
  %301 = or i1 %298, %300
  br i1 %301, label %302, label %304

302:                                              ; preds = %297
  %303 = select i1 %231, i64 -4503599627370496, i64 9218868437227405312
  br label %326

304:                                              ; preds = %297, %294, %287, %276
  %305 = phi i64 [ %279, %276 ], [ %279, %297 ], [ %293, %287 ], [ %296, %294 ]
  %306 = phi i64 [ %259, %276 ], [ 2045, %297 ], [ 0, %287 ], [ 0, %294 ]
  %307 = and i64 %305, 1023
  %308 = add i64 %305, 512
  %309 = lshr i64 %308, 10
  %310 = icmp eq i64 %307, 512
  %311 = zext i1 %310 to i64
  %312 = xor i64 %311, -1
  %313 = and i64 %309, %312
  %314 = icmp eq i64 %313, 0
  %315 = select i1 %231, i64 -9223372036854775808, i64 0
  %316 = shl i64 %306, 52
  %317 = select i1 %314, i64 0, i64 %316
  %318 = or i64 %313, %315
  %319 = add i64 %318, %317
  br label %326

320:                                              ; preds = %209, %183, %132
  %321 = and i64 %3, 9218868437227405312
  %322 = icmp ne i64 %321, 9218868437227405312
  %323 = icmp eq i64 %13, 0
  %324 = or i1 %322, %323
  %325 = select i1 %324, i64 %4, i64 %3
  br label %326

326:                                              ; preds = %320, %304, %302, %265, %209, %185, %138, %135, %132, %123, %107, %105, %59, %38, %23, %21
  %327 = phi i64 [ %128, %123 ], [ %39, %38 ], [ %22, %21 ], [ %3, %23 ], [ %3, %59 ], [ %122, %107 ], [ %106, %105 ], [ %325, %320 ], [ %186, %185 ], [ %176, %138 ], [ -2251799813685248, %132 ], [ 0, %135 ], [ %3, %209 ], [ %275, %265 ], [ %319, %304 ], [ %303, %302 ]
  %328 = bitcast i64 %327 to double
  ret double %328
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind readnone willreturn
define internal double @__softfloat_f64_div(double noundef %0, double noundef %1) #3 {
  %3 = bitcast double %0 to i64
  %4 = bitcast double %1 to i64
  %5 = lshr i64 %3, 52
  %6 = and i64 %5, 2047
  %7 = and i64 %3, 4503599627370495
  %8 = lshr i64 %4, 52
  %9 = and i64 %8, 2047
  %10 = and i64 %4, 4503599627370495
  %11 = xor i64 %4, %3
  %12 = icmp eq i64 %6, 2047
  br i1 %12, label %13, label %19

13:                                               ; preds = %2
  %14 = icmp eq i64 %7, 0
  br i1 %14, label %15, label %189

15:                                               ; preds = %13
  %16 = icmp eq i64 %9, 2047
  br i1 %16, label %17, label %195

17:                                               ; preds = %15
  %18 = icmp eq i64 %10, 0
  br i1 %18, label %200, label %189

19:                                               ; preds = %2
  switch i64 %9, label %56 [
    i64 2047, label %20
    i64 0, label %22
  ]

20:                                               ; preds = %19
  %21 = icmp eq i64 %10, 0
  br i1 %21, label %198, label %189

22:                                               ; preds = %19
  %23 = icmp eq i64 %10, 0
  br i1 %23, label %24, label %27

24:                                               ; preds = %22
  %25 = or i64 %6, %7
  %26 = icmp eq i64 %25, 0
  br i1 %26, label %200, label %195

27:                                               ; preds = %22
  %28 = lshr i64 %10, 32
  %29 = trunc i64 %28 to i32
  %30 = icmp eq i32 %29, 0
  %31 = trunc i64 %4 to i32
  %32 = select i1 %30, i8 32, i8 0
  %33 = select i1 %30, i32 %31, i32 %29
  %34 = icmp ult i32 %33, 65536
  %35 = or i8 %32, 16
  %36 = shl i32 %33, 16
  %37 = select i1 %34, i8 %35, i8 %32
  %38 = select i1 %34, i32 %36, i32 %33
  %39 = icmp ult i32 %38, 16777216
  %40 = or i8 %37, 8
  %41 = shl i32 %38, 8
  %42 = select i1 %39, i8 %40, i8 %37
  %43 = select i1 %39, i32 %41, i32 %38
  %44 = lshr i32 %43, 24
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %45
  %47 = load i8, i8 addrspace(1)* %46, align 1, !tbaa !10
  %48 = add nsw i8 %42, -11
  %49 = add i8 %48, %47
  %50 = sext i8 %49 to i32
  %51 = sub nsw i32 1, %50
  %52 = sext i32 %51 to i64
  %53 = and i32 %50, 63
  %54 = zext i32 %53 to i64
  %55 = shl i64 %10, %54
  br label %56

56:                                               ; preds = %27, %19
  %57 = phi i64 [ %55, %27 ], [ %10, %19 ]
  %58 = phi i64 [ %52, %27 ], [ %9, %19 ]
  %59 = icmp eq i64 %6, 0
  br i1 %59, label %60, label %91

60:                                               ; preds = %56
  %61 = icmp eq i64 %7, 0
  br i1 %61, label %198, label %62

62:                                               ; preds = %60
  %63 = lshr i64 %7, 32
  %64 = trunc i64 %63 to i32
  %65 = icmp eq i32 %64, 0
  %66 = trunc i64 %3 to i32
  %67 = select i1 %65, i8 32, i8 0
  %68 = select i1 %65, i32 %66, i32 %64
  %69 = icmp ult i32 %68, 65536
  %70 = or i8 %67, 16
  %71 = shl i32 %68, 16
  %72 = select i1 %69, i8 %70, i8 %67
  %73 = select i1 %69, i32 %71, i32 %68
  %74 = icmp ult i32 %73, 16777216
  %75 = or i8 %72, 8
  %76 = shl i32 %73, 8
  %77 = select i1 %74, i8 %75, i8 %72
  %78 = select i1 %74, i32 %76, i32 %73
  %79 = lshr i32 %78, 24
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %80
  %82 = load i8, i8 addrspace(1)* %81, align 1, !tbaa !10
  %83 = add nsw i8 %77, -11
  %84 = add i8 %83, %82
  %85 = sext i8 %84 to i32
  %86 = sub nsw i32 1, %85
  %87 = sext i32 %86 to i64
  %88 = and i32 %85, 63
  %89 = zext i32 %88 to i64
  %90 = shl i64 %7, %89
  br label %91

91:                                               ; preds = %62, %56
  %92 = phi i64 [ %7, %56 ], [ %90, %62 ]
  %93 = phi i64 [ %6, %56 ], [ %87, %62 ]
  %94 = sub nsw i64 %93, %58
  %95 = or i64 %92, 4503599627370496
  %96 = or i64 %57, 4503599627370496
  %97 = icmp ult i64 %95, %96
  %98 = select i1 %97, i64 1021, i64 1022
  %99 = select i1 %97, i64 11, i64 10
  %100 = add nsw i64 %98, %94
  %101 = shl i64 %95, %99
  %102 = shl i64 %96, 11
  %103 = lshr i64 %102, 32
  %104 = udiv i64 9223372036854775807, %103
  %105 = add nuw nsw i64 %104, 4294967294
  %106 = lshr i64 %101, 32
  %107 = and i64 %105, 4294967295
  %108 = mul nuw i64 %106, %107
  %109 = lshr i64 %108, 31
  %110 = and i64 %109, 4294967294
  %111 = mul nuw i64 %110, %103
  %112 = sub i64 %101, %111
  %113 = shl i64 %112, 28
  %114 = lshr exact i64 %102, 4
  %115 = and i64 %114, 268435328
  %116 = mul nuw nsw i64 %110, %115
  %117 = sub i64 %113, %116
  %118 = lshr i64 %117, 32
  %119 = mul nuw i64 %118, %107
  %120 = lshr i64 %119, 32
  %121 = trunc i64 %120 to i32
  %122 = add i32 %121, 4
  %123 = and i64 %108, -4294967296
  %124 = zext i32 %122 to i64
  %125 = shl nuw nsw i64 %124, 4
  %126 = add i64 %125, %123
  %127 = and i64 %124, 28
  %128 = icmp eq i64 %127, 0
  br i1 %128, label %129, label %146

129:                                              ; preds = %91
  %130 = and i64 %126, -128
  %131 = shl i32 %122, 1
  %132 = and i32 %131, -16
  %133 = zext i32 %132 to i64
  %134 = mul nuw i64 %103, %133
  %135 = sub i64 %117, %134
  %136 = shl i64 %135, 28
  %137 = mul nuw nsw i64 %115, %133
  %138 = sub i64 %136, %137
  %139 = icmp sgt i64 %138, -1
  br i1 %139, label %142, label %140

140:                                              ; preds = %129
  %141 = add i64 %130, -128
  br label %146

142:                                              ; preds = %129
  %143 = icmp ne i64 %136, %137
  %144 = zext i1 %143 to i64
  %145 = or i64 %130, %144
  br label %146

146:                                              ; preds = %142, %140, %91
  %147 = phi i64 [ %141, %140 ], [ %126, %91 ], [ %145, %142 ]
  %148 = and i64 %100, 65535
  %149 = icmp ugt i64 %148, 2044
  br i1 %149, label %150, label %173

150:                                              ; preds = %146
  %151 = icmp slt i64 %100, 0
  br i1 %151, label %152, label %165

152:                                              ; preds = %150
  %153 = sub nsw i64 0, %100
  %154 = icmp ult i64 %153, 63
  br i1 %154, label %155, label %162

155:                                              ; preds = %152
  %156 = lshr i64 %147, %153
  %157 = and i64 %100, 63
  %158 = shl i64 %147, %157
  %159 = icmp ne i64 %158, 0
  %160 = zext i1 %159 to i64
  %161 = or i64 %156, %160
  br label %173

162:                                              ; preds = %152
  %163 = icmp ne i64 %147, 0
  %164 = zext i1 %163 to i64
  br label %173

165:                                              ; preds = %150
  %166 = icmp ugt i64 %100, 2045
  %167 = add i64 %147, 512
  %168 = icmp slt i64 %167, 0
  %169 = or i1 %166, %168
  br i1 %169, label %170, label %173

170:                                              ; preds = %165
  %171 = and i64 %11, -9223372036854775808
  %172 = or i64 %171, 9218868437227405312
  br label %200

173:                                              ; preds = %165, %162, %155, %146
  %174 = phi i64 [ %147, %146 ], [ %147, %165 ], [ %161, %155 ], [ %164, %162 ]
  %175 = phi i64 [ %100, %146 ], [ 2045, %165 ], [ 0, %155 ], [ 0, %162 ]
  %176 = and i64 %174, 1023
  %177 = add i64 %174, 512
  %178 = lshr i64 %177, 10
  %179 = icmp eq i64 %176, 512
  %180 = zext i1 %179 to i64
  %181 = xor i64 %180, -1
  %182 = and i64 %178, %181
  %183 = icmp eq i64 %182, 0
  %184 = and i64 %11, -9223372036854775808
  %185 = shl i64 %175, 52
  %186 = select i1 %183, i64 0, i64 %185
  %187 = or i64 %182, %184
  %188 = add i64 %187, %186
  br label %200

189:                                              ; preds = %20, %17, %13
  %190 = and i64 %3, 9218868437227405312
  %191 = icmp ne i64 %190, 9218868437227405312
  %192 = icmp eq i64 %7, 0
  %193 = or i1 %191, %192
  %194 = select i1 %193, i64 %4, i64 %3
  br label %200

195:                                              ; preds = %24, %15
  %196 = and i64 %11, -9223372036854775808
  %197 = or i64 %196, 9218868437227405312
  br label %200

198:                                              ; preds = %60, %20
  %199 = and i64 %11, -9223372036854775808
  br label %200

200:                                              ; preds = %198, %195, %189, %173, %170, %24, %17
  %201 = phi i64 [ %194, %189 ], [ %197, %195 ], [ %199, %198 ], [ -2251799813685248, %24 ], [ -2251799813685248, %17 ], [ %188, %173 ], [ %172, %170 ]
  %202 = bitcast i64 %201 to double
  ret double %202
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind readnone willreturn
define internal double @__softfloat_f64_mul(double noundef %0, double noundef %1) #3 {
  %3 = bitcast double %0 to i64
  %4 = bitcast double %1 to i64
  %5 = lshr i64 %3, 52
  %6 = and i64 %5, 2047
  %7 = and i64 %3, 4503599627370495
  %8 = lshr i64 %4, 52
  %9 = and i64 %8, 2047
  %10 = and i64 %4, 4503599627370495
  %11 = xor i64 %4, %3
  %12 = icmp eq i64 %6, 2047
  br i1 %12, label %13, label %21

13:                                               ; preds = %2
  %14 = icmp eq i64 %7, 0
  br i1 %14, label %15, label %158

15:                                               ; preds = %13
  %16 = icmp eq i64 %9, 2047
  %17 = icmp ne i64 %10, 0
  %18 = and i1 %17, %16
  br i1 %18, label %158, label %19

19:                                               ; preds = %15
  %20 = or i64 %9, %10
  br label %164

21:                                               ; preds = %2
  %22 = icmp eq i64 %9, 2047
  br i1 %22, label %23, label %27

23:                                               ; preds = %21
  %24 = icmp eq i64 %10, 0
  br i1 %24, label %25, label %158

25:                                               ; preds = %23
  %26 = or i64 %6, %7
  br label %164

27:                                               ; preds = %21
  %28 = icmp eq i64 %6, 0
  br i1 %28, label %29, label %60

29:                                               ; preds = %27
  %30 = icmp eq i64 %7, 0
  br i1 %30, label %170, label %31

31:                                               ; preds = %29
  %32 = lshr i64 %7, 32
  %33 = trunc i64 %32 to i32
  %34 = icmp eq i32 %33, 0
  %35 = trunc i64 %3 to i32
  %36 = select i1 %34, i8 32, i8 0
  %37 = select i1 %34, i32 %35, i32 %33
  %38 = icmp ult i32 %37, 65536
  %39 = or i8 %36, 16
  %40 = shl i32 %37, 16
  %41 = select i1 %38, i8 %39, i8 %36
  %42 = select i1 %38, i32 %40, i32 %37
  %43 = icmp ult i32 %42, 16777216
  %44 = or i8 %41, 8
  %45 = shl i32 %42, 8
  %46 = select i1 %43, i8 %44, i8 %41
  %47 = select i1 %43, i32 %45, i32 %42
  %48 = lshr i32 %47, 24
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %49
  %51 = load i8, i8 addrspace(1)* %50, align 1, !tbaa !10
  %52 = add nsw i8 %46, -11
  %53 = add i8 %52, %51
  %54 = sext i8 %53 to i32
  %55 = sub nsw i32 1, %54
  %56 = sext i32 %55 to i64
  %57 = and i32 %54, 63
  %58 = zext i32 %57 to i64
  %59 = shl i64 %7, %58
  br label %60

60:                                               ; preds = %31, %27
  %61 = phi i64 [ %7, %27 ], [ %59, %31 ]
  %62 = phi i64 [ %6, %27 ], [ %56, %31 ]
  %63 = icmp eq i64 %9, 0
  br i1 %63, label %64, label %95

64:                                               ; preds = %60
  %65 = icmp eq i64 %10, 0
  br i1 %65, label %170, label %66

66:                                               ; preds = %64
  %67 = lshr i64 %10, 32
  %68 = trunc i64 %67 to i32
  %69 = icmp eq i32 %68, 0
  %70 = trunc i64 %4 to i32
  %71 = select i1 %69, i8 32, i8 0
  %72 = select i1 %69, i32 %70, i32 %68
  %73 = icmp ult i32 %72, 65536
  %74 = or i8 %71, 16
  %75 = shl i32 %72, 16
  %76 = select i1 %73, i8 %74, i8 %71
  %77 = select i1 %73, i32 %75, i32 %72
  %78 = icmp ult i32 %77, 16777216
  %79 = or i8 %76, 8
  %80 = shl i32 %77, 8
  %81 = select i1 %78, i8 %79, i8 %76
  %82 = select i1 %78, i32 %80, i32 %77
  %83 = lshr i32 %82, 24
  %84 = zext i32 %83 to i64
  %85 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %84
  %86 = load i8, i8 addrspace(1)* %85, align 1, !tbaa !10
  %87 = add nsw i8 %81, -11
  %88 = add i8 %87, %86
  %89 = sext i8 %88 to i32
  %90 = sub nsw i32 1, %89
  %91 = sext i32 %90 to i64
  %92 = and i32 %89, 63
  %93 = zext i32 %92 to i64
  %94 = shl i64 %10, %93
  br label %95

95:                                               ; preds = %66, %60
  %96 = phi i64 [ %10, %60 ], [ %94, %66 ]
  %97 = phi i64 [ %9, %60 ], [ %91, %66 ]
  %98 = add nsw i64 %97, %62
  %99 = shl i64 %61, 10
  %100 = or i64 %99, 4611686018427387904
  %101 = shl i64 %96, 11
  %102 = or i64 %101, -9223372036854775808
  %103 = zext i64 %100 to i128
  %104 = zext i64 %102 to i128
  %105 = mul nuw i128 %104, %103
  %106 = trunc i128 %105 to i64
  %107 = lshr i128 %105, 64
  %108 = trunc i128 %107 to i64
  %109 = icmp ne i64 %106, 0
  %110 = zext i1 %109 to i64
  %111 = or i64 %110, %108
  %112 = icmp ult i64 %111, 4611686018427387904
  %113 = select i1 %112, i64 -1024, i64 -1023
  %114 = add nsw i64 %98, %113
  %115 = zext i1 %112 to i64
  %116 = shl i64 %111, %115
  %117 = and i64 %114, 65535
  %118 = icmp ugt i64 %117, 2044
  br i1 %118, label %119, label %142

119:                                              ; preds = %95
  %120 = icmp slt i64 %114, 0
  br i1 %120, label %121, label %134

121:                                              ; preds = %119
  %122 = sub nsw i64 0, %114
  %123 = icmp ult i64 %122, 63
  br i1 %123, label %124, label %131

124:                                              ; preds = %121
  %125 = lshr i64 %116, %122
  %126 = and i64 %114, 63
  %127 = shl i64 %116, %126
  %128 = icmp ne i64 %127, 0
  %129 = zext i1 %128 to i64
  %130 = or i64 %125, %129
  br label %142

131:                                              ; preds = %121
  %132 = icmp ne i64 %116, 0
  %133 = zext i1 %132 to i64
  br label %142

134:                                              ; preds = %119
  %135 = icmp ugt i64 %114, 2045
  %136 = add i64 %116, 512
  %137 = icmp slt i64 %136, 0
  %138 = or i1 %135, %137
  br i1 %138, label %139, label %142

139:                                              ; preds = %134
  %140 = and i64 %11, -9223372036854775808
  %141 = or i64 %140, 9218868437227405312
  br label %172

142:                                              ; preds = %134, %131, %124, %95
  %143 = phi i64 [ %116, %95 ], [ %116, %134 ], [ %130, %124 ], [ %133, %131 ]
  %144 = phi i64 [ %114, %95 ], [ 2045, %134 ], [ 0, %124 ], [ 0, %131 ]
  %145 = and i64 %143, 1023
  %146 = add i64 %143, 512
  %147 = lshr i64 %146, 10
  %148 = icmp eq i64 %145, 512
  %149 = zext i1 %148 to i64
  %150 = xor i64 %149, -1
  %151 = and i64 %147, %150
  %152 = icmp eq i64 %151, 0
  %153 = and i64 %11, -9223372036854775808
  %154 = shl i64 %144, 52
  %155 = select i1 %152, i64 0, i64 %154
  %156 = or i64 %151, %153
  %157 = add i64 %156, %155
  br label %172

158:                                              ; preds = %23, %15, %13
  %159 = and i64 %3, 9218868437227405312
  %160 = icmp ne i64 %159, 9218868437227405312
  %161 = icmp eq i64 %7, 0
  %162 = or i1 %160, %161
  %163 = select i1 %162, i64 %4, i64 %3
  br label %172

164:                                              ; preds = %25, %19
  %165 = phi i64 [ %20, %19 ], [ %26, %25 ]
  %166 = icmp eq i64 %165, 0
  %167 = and i64 %11, -9223372036854775808
  %168 = or i64 %167, 9218868437227405312
  %169 = select i1 %166, i64 -2251799813685248, i64 %168
  br label %172

170:                                              ; preds = %64, %29
  %171 = and i64 %11, -9223372036854775808
  br label %172

172:                                              ; preds = %170, %164, %158, %142, %139
  %173 = phi i64 [ %163, %158 ], [ %171, %170 ], [ %169, %164 ], [ %157, %142 ], [ %141, %139 ]
  %174 = bitcast i64 %173 to double
  ret double %174
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind readnone willreturn
define internal double @__softfloat_f64_sub(double noundef %0, double noundef %1) #3 {
  %3 = bitcast double %0 to i64
  %4 = bitcast double %1 to i64
  %5 = icmp slt i64 %3, 0
  %6 = lshr i64 %3, 63
  %7 = trunc i64 %6 to i32
  %8 = lshr i64 %4, 63
  %9 = trunc i64 %8 to i32
  %10 = icmp eq i32 %7, %9
  %11 = lshr i64 %3, 52
  %12 = and i64 %11, 2047
  %13 = and i64 %3, 4503599627370495
  %14 = lshr i64 %4, 52
  %15 = and i64 %14, 2047
  %16 = and i64 %4, 4503599627370495
  %17 = sub nsw i64 %12, %15
  %18 = icmp eq i64 %12, %15
  br i1 %10, label %19, label %216

19:                                               ; preds = %2
  br i1 %18, label %20, label %67

20:                                               ; preds = %19
  %21 = icmp eq i64 %12, 2047
  br i1 %21, label %22, label %25

22:                                               ; preds = %20
  %23 = or i64 %16, %13
  %24 = icmp eq i64 %23, 0
  br i1 %24, label %326, label %210

25:                                               ; preds = %20
  %26 = sub nsw i64 %13, %16
  %27 = icmp eq i64 %13, %16
  br i1 %27, label %326, label %28

28:                                               ; preds = %25
  %29 = tail call i64 @llvm.usub.sat.i64(i64 %12, i64 1)
  %30 = xor i64 %26, %3
  %31 = tail call i64 @llvm.abs.i64(i64 %26, i1 true)
  %32 = lshr i64 %31, 32
  %33 = trunc i64 %32 to i32
  %34 = icmp eq i32 %33, 0
  %35 = trunc i64 %31 to i32
  %36 = select i1 %34, i8 32, i8 0
  %37 = select i1 %34, i32 %35, i32 %33
  %38 = icmp ult i32 %37, 65536
  %39 = or i8 %36, 16
  %40 = shl i32 %37, 16
  %41 = select i1 %38, i8 %39, i8 %36
  %42 = select i1 %38, i32 %40, i32 %37
  %43 = icmp ult i32 %42, 16777216
  %44 = or i8 %41, 8
  %45 = shl i32 %42, 8
  %46 = select i1 %43, i8 %44, i8 %41
  %47 = select i1 %43, i32 %45, i32 %42
  %48 = lshr i32 %47, 24
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %49
  %51 = load i8, i8 addrspace(1)* %50, align 1, !tbaa !10
  %52 = add nsw i8 %46, -11
  %53 = add i8 %52, %51
  %54 = sext i8 %53 to i64
  %55 = sub nsw i64 %29, %54
  %56 = icmp slt i64 %55, 0
  %57 = trunc i64 %29 to i8
  %58 = select i1 %56, i8 %57, i8 %53
  %59 = tail call i64 @llvm.smax.i64(i64 %55, i64 0)
  %60 = and i64 %30, -9223372036854775808
  %61 = shl nuw i64 %59, 52
  %62 = add i64 %61, %60
  %63 = and i8 %58, 63
  %64 = zext i8 %63 to i64
  %65 = shl i64 %31, %64
  %66 = add i64 %62, %65
  br label %326

67:                                               ; preds = %19
  %68 = shl nuw nsw i64 %13, 10
  %69 = shl nuw nsw i64 %16, 10
  %70 = icmp slt i64 %17, 0
  br i1 %70, label %71, label %97

71:                                               ; preds = %67
  %72 = icmp eq i64 %15, 2047
  br i1 %72, label %73, label %77

73:                                               ; preds = %71
  %74 = icmp eq i64 %16, 0
  br i1 %74, label %75, label %210

75:                                               ; preds = %73
  %76 = select i1 %5, i64 9218868437227405312, i64 -4503599627370496
  br label %326

77:                                               ; preds = %71
  %78 = xor i1 %5, true
  %79 = icmp eq i64 %12, 0
  %80 = select i1 %79, i64 %68, i64 4611686018427387904
  %81 = add nuw nsw i64 %80, %68
  %82 = sub nsw i64 0, %17
  %83 = icmp ult i64 %82, 63
  br i1 %83, label %84, label %91

84:                                               ; preds = %77
  %85 = lshr i64 %81, %82
  %86 = and i64 %17, 63
  %87 = shl i64 %81, %86
  %88 = icmp ne i64 %87, 0
  %89 = zext i1 %88 to i64
  %90 = or i64 %85, %89
  br label %94

91:                                               ; preds = %77
  %92 = icmp ne i64 %81, 0
  %93 = zext i1 %92 to i64
  br label %94

94:                                               ; preds = %91, %84
  %95 = phi i64 [ %90, %84 ], [ %93, %91 ]
  %96 = sub nsw i64 %69, %95
  br label %120

97:                                               ; preds = %67
  %98 = icmp eq i64 %12, 2047
  br i1 %98, label %99, label %101

99:                                               ; preds = %97
  %100 = icmp eq i64 %13, 0
  br i1 %100, label %326, label %210

101:                                              ; preds = %97
  %102 = icmp eq i64 %15, 0
  %103 = select i1 %102, i64 %69, i64 4611686018427387904
  %104 = add nuw nsw i64 %103, %69
  %105 = icmp ult i64 %17, 63
  br i1 %105, label %106, label %114

106:                                              ; preds = %101
  %107 = lshr i64 %104, %17
  %108 = sub nsw i64 0, %17
  %109 = and i64 %108, 63
  %110 = shl i64 %104, %109
  %111 = icmp ne i64 %110, 0
  %112 = zext i1 %111 to i64
  %113 = or i64 %107, %112
  br label %117

114:                                              ; preds = %101
  %115 = icmp ne i64 %104, 0
  %116 = zext i1 %115 to i64
  br label %117

117:                                              ; preds = %114, %106
  %118 = phi i64 [ %113, %106 ], [ %116, %114 ]
  %119 = sub nsw i64 %68, %118
  br label %120

120:                                              ; preds = %117, %94
  %121 = phi i1 [ %78, %94 ], [ %5, %117 ]
  %122 = phi i64 [ %15, %94 ], [ %12, %117 ]
  %123 = phi i64 [ %96, %94 ], [ %119, %117 ]
  %124 = add nsw i64 %123, 4611686018427387904
  %125 = lshr i64 %124, 32
  %126 = trunc i64 %125 to i32
  %127 = icmp eq i32 %126, 0
  %128 = trunc i64 %123 to i32
  %129 = select i1 %127, i8 32, i8 0
  %130 = select i1 %127, i32 %128, i32 %126
  %131 = icmp ult i32 %130, 65536
  %132 = or i8 %129, 16
  %133 = shl i32 %130, 16
  %134 = select i1 %131, i8 %132, i8 %129
  %135 = select i1 %131, i32 %133, i32 %130
  %136 = icmp ult i32 %135, 16777216
  %137 = or i8 %134, 8
  %138 = shl i32 %135, 8
  %139 = select i1 %136, i8 %137, i8 %134
  %140 = select i1 %136, i32 %138, i32 %135
  %141 = lshr i32 %140, 24
  %142 = zext i32 %141 to i64
  %143 = getelementptr inbounds [256 x i8], [256 x i8] addrspace(1)* bitcast (<{ [128 x i8], [128 x i8] }> addrspace(1)* @softfloat_countLeadingZeros8 to [256 x i8] addrspace(1)*), i64 0, i64 %142
  %144 = load i8, i8 addrspace(1)* %143, align 1, !tbaa !10
  %145 = add i8 %144, -1
  %146 = add i8 %145, %139
  %147 = xor i8 %146, -1
  %148 = sext i8 %147 to i64
  %149 = add nsw i64 %122, %148
  %150 = zext i8 %146 to i32
  %151 = icmp sgt i8 %146, 9
  %152 = trunc i64 %149 to i32
  %153 = icmp ult i32 %152, 2045
  %154 = select i1 %151, i1 %153, i1 false
  br i1 %154, label %155, label %166

155:                                              ; preds = %120
  %156 = select i1 %121, i64 -9223372036854775808, i64 0
  %157 = icmp eq i64 %124, 0
  %158 = shl i64 %149, 52
  %159 = select i1 %157, i64 0, i64 %158
  %160 = add i64 %159, %156
  %161 = add nuw nsw i32 %150, 54
  %162 = and i32 %161, 63
  %163 = zext i32 %162 to i64
  %164 = shl i64 %124, %163
  %165 = add i64 %160, %164
  br label %326

166:                                              ; preds = %120
  %167 = and i32 %150, 63
  %168 = zext i32 %167 to i64
  %169 = shl i64 %124, %168
  %170 = and i64 %149, 65535
  %171 = icmp ugt i64 %170, 2044
  br i1 %171, label %172, label %194

172:                                              ; preds = %166
  %173 = icmp slt i64 %149, 0
  br i1 %173, label %174, label %187

174:                                              ; preds = %172
  %175 = sub nsw i64 0, %149
  %176 = icmp ult i64 %175, 63
  br i1 %176, label %177, label %184

177:                                              ; preds = %174
  %178 = lshr i64 %169, %175
  %179 = and i64 %149, 63
  %180 = shl i64 %169, %179
  %181 = icmp ne i64 %180, 0
  %182 = zext i1 %181 to i64
  %183 = or i64 %178, %182
  br label %194

184:                                              ; preds = %174
  %185 = icmp ne i64 %169, 0
  %186 = zext i1 %185 to i64
  br label %194

187:                                              ; preds = %172
  %188 = icmp ugt i64 %149, 2045
  %189 = add i64 %169, 512
  %190 = icmp slt i64 %189, 0
  %191 = or i1 %188, %190
  br i1 %191, label %192, label %194

192:                                              ; preds = %187
  %193 = select i1 %121, i64 -4503599627370496, i64 9218868437227405312
  br label %326

194:                                              ; preds = %187, %184, %177, %166
  %195 = phi i64 [ %169, %166 ], [ %169, %187 ], [ %183, %177 ], [ %186, %184 ]
  %196 = phi i64 [ %149, %166 ], [ 2045, %187 ], [ 0, %177 ], [ 0, %184 ]
  %197 = and i64 %195, 1023
  %198 = add i64 %195, 512
  %199 = lshr i64 %198, 10
  %200 = icmp eq i64 %197, 512
  %201 = zext i1 %200 to i64
  %202 = xor i64 %201, -1
  %203 = and i64 %199, %202
  %204 = icmp eq i64 %203, 0
  %205 = select i1 %121, i64 -9223372036854775808, i64 0
  %206 = shl i64 %196, 52
  %207 = select i1 %204, i64 0, i64 %206
  %208 = or i64 %203, %205
  %209 = add i64 %208, %207
  br label %326

210:                                              ; preds = %99, %73, %22
  %211 = and i64 %3, 9218868437227405312
  %212 = icmp ne i64 %211, 9218868437227405312
  %213 = icmp eq i64 %13, 0
  %214 = or i1 %212, %213
  %215 = select i1 %214, i64 %4, i64 %3
  br label %326

216:                                              ; preds = %2
  br i1 %18, label %217, label %227

217:                                              ; preds = %216
  switch i64 %12, label %223 [
    i64 0, label %218
    i64 2047, label %220
  ]

218:                                              ; preds = %217
  %219 = add i64 %16, %3
  br label %326

220:                                              ; preds = %217
  %221 = or i64 %16, %13
  %222 = icmp eq i64 %221, 0
  br i1 %222, label %326, label %320

223:                                              ; preds = %217
  %224 = or i64 %13, 9007199254740992
  %225 = add nuw nsw i64 %224, %16
  %226 = shl nuw nsw i64 %225, 9
  br label %286

227:                                              ; preds = %216
  %228 = shl nuw nsw i64 %13, 9
  %229 = shl nuw nsw i64 %16, 9
  %230 = icmp slt i64 %17, 0
  br i1 %230, label %231, label %254

231:                                              ; preds = %227
  %232 = icmp eq i64 %15, 2047
  br i1 %232, label %233, label %237

233:                                              ; preds = %231
  %234 = icmp eq i64 %16, 0
  br i1 %234, label %235, label %320

235:                                              ; preds = %233
  %236 = select i1 %5, i64 -4503599627370496, i64 9218868437227405312
  br label %326

237:                                              ; preds = %231
  %238 = icmp eq i64 %12, 0
  %239 = add nuw nsw i64 %228, 2305843009213693952
  %240 = shl nuw nsw i64 %13, 10
  %241 = select i1 %238, i64 %240, i64 %239
  %242 = sub nsw i64 0, %17
  %243 = icmp ult i64 %242, 63
  br i1 %243, label %244, label %251

244:                                              ; preds = %237
  %245 = lshr i64 %241, %242
  %246 = and i64 %17, 63
  %247 = shl i64 %241, %246
  %248 = icmp ne i64 %247, 0
  %249 = zext i1 %248 to i64
  %250 = or i64 %245, %249
  br label %275

251:                                              ; preds = %237
  %252 = icmp ne i64 %241, 0
  %253 = zext i1 %252 to i64
  br label %275

254:                                              ; preds = %227
  %255 = icmp eq i64 %12, 2047
  br i1 %255, label %256, label %258

256:                                              ; preds = %254
  %257 = icmp eq i64 %13, 0
  br i1 %257, label %326, label %320

258:                                              ; preds = %254
  %259 = icmp eq i64 %15, 0
  %260 = add nuw nsw i64 %229, 2305843009213693952
  %261 = shl nuw nsw i64 %16, 10
  %262 = select i1 %259, i64 %261, i64 %260
  %263 = icmp ult i64 %17, 63
  br i1 %263, label %264, label %272

264:                                              ; preds = %258
  %265 = lshr i64 %262, %17
  %266 = sub nsw i64 0, %17
  %267 = and i64 %266, 63
  %268 = shl i64 %262, %267
  %269 = icmp ne i64 %268, 0
  %270 = zext i1 %269 to i64
  %271 = or i64 %265, %270
  br label %275

272:                                              ; preds = %258
  %273 = icmp ne i64 %262, 0
  %274 = zext i1 %273 to i64
  br label %275

275:                                              ; preds = %272, %264, %251, %244
  %276 = phi i64 [ %229, %244 ], [ %229, %251 ], [ %271, %264 ], [ %274, %272 ]
  %277 = phi i64 [ %250, %244 ], [ %253, %251 ], [ %228, %264 ], [ %228, %272 ]
  %278 = phi i64 [ %15, %244 ], [ %15, %251 ], [ %12, %264 ], [ %12, %272 ]
  %279 = add nuw nsw i64 %276, 2305843009213693952
  %280 = add nuw nsw i64 %279, %277
  %281 = icmp ult i64 %280, 4611686018427387904
  %282 = sext i1 %281 to i64
  %283 = add nsw i64 %278, %282
  %284 = zext i1 %281 to i64
  %285 = shl nuw i64 %280, %284
  br label %286

286:                                              ; preds = %275, %223
  %287 = phi i64 [ %12, %223 ], [ %283, %275 ]
  %288 = phi i64 [ %226, %223 ], [ %285, %275 ]
  %289 = and i64 %287, 65535
  %290 = icmp ugt i64 %289, 2044
  br i1 %290, label %291, label %304

291:                                              ; preds = %286
  %292 = icmp slt i64 %287, 0
  br i1 %292, label %293, label %297

293:                                              ; preds = %291
  %294 = lshr i64 %288, 1
  %295 = and i64 %288, 1
  %296 = or i64 %294, %295
  br label %304

297:                                              ; preds = %291
  %298 = icmp ugt i64 %287, 2045
  %299 = add i64 %288, 512
  %300 = icmp slt i64 %299, 0
  %301 = or i1 %298, %300
  br i1 %301, label %302, label %304

302:                                              ; preds = %297
  %303 = select i1 %5, i64 -4503599627370496, i64 9218868437227405312
  br label %326

304:                                              ; preds = %297, %293, %286
  %305 = phi i64 [ %288, %286 ], [ %288, %297 ], [ %296, %293 ]
  %306 = phi i64 [ %287, %286 ], [ 2045, %297 ], [ 0, %293 ]
  %307 = and i64 %305, 1023
  %308 = add i64 %305, 512
  %309 = lshr i64 %308, 10
  %310 = icmp eq i64 %307, 512
  %311 = zext i1 %310 to i64
  %312 = xor i64 %311, -1
  %313 = and i64 %309, %312
  %314 = icmp eq i64 %313, 0
  %315 = and i64 %3, -9223372036854775808
  %316 = shl nsw i64 %306, 52
  %317 = select i1 %314, i64 0, i64 %316
  %318 = or i64 %313, %315
  %319 = add i64 %318, %317
  br label %326

320:                                              ; preds = %256, %233, %220
  %321 = and i64 %3, 9218868437227405312
  %322 = icmp ne i64 %321, 9218868437227405312
  %323 = icmp eq i64 %13, 0
  %324 = or i1 %322, %323
  %325 = select i1 %324, i64 %4, i64 %3
  br label %326

326:                                              ; preds = %320, %304, %302, %256, %235, %220, %218, %210, %194, %192, %155, %99, %75, %28, %25, %22
  %327 = phi i64 [ %215, %210 ], [ %76, %75 ], [ %66, %28 ], [ -2251799813685248, %22 ], [ 0, %25 ], [ %3, %99 ], [ %165, %155 ], [ %209, %194 ], [ %193, %192 ], [ %325, %320 ], [ %236, %235 ], [ %219, %218 ], [ %3, %220 ], [ %3, %256 ], [ %319, %304 ], [ %303, %302 ]
  %328 = bitcast i64 %327 to double
  ret double %328
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.usub.sat.i64(i64, i64) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.abs.i64(i64, i1 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn
declare i64 @llvm.smax.i64(i64, i64) #4

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #5

; Function Attrs: mustprogress nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #5

attributes #0 = { argmemonly convergent mustprogress nofree norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx928" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+gfx928-insts,+mmop-insts,+mmop2-insts,+s-memrealtime,+s-memtime-inst,+sramecc" "uniform-work-group-size"="true" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn }
attributes #2 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #3 = { mustprogress nofree noinline norecurse nosync nounwind readnone willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #5 = { mustprogress nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 1}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 15.0.0"}
!4 = !{i32 0, i32 1024}
!5 = !{!6, !6, i64 0}
!6 = !{!"float", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{}
!10 = !{!11, !11, i64 0}
!11 = !{!"omnipotent char", !12, i64 0}
!12 = !{!"Simple C/C++ TBAA"}

; __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa-gfx928

; __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu
; ModuleID = './reduce_16.cpp'
source_filename = "./reduce_16.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }
%struct.dim3 = type { i32, i32, i32 }
%struct.ihipStream_t = type opaque

$__hip_gpubin_handle = comdat any

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external hidden global i8
@_Z6reducePfS_ = dso_local constant void (float*, float*)* @_Z21__device_stub__reducePfS_, align 8
@_ZSt4cerr = external dso_local global %"class.std::basic_ostream", align 8
@.str = private unnamed_addr constant [31 x i8] c"\E6\B2\A1\E6\9C\89\E6\89\BE\E5\88\B0\E5\8F\AF\E7\94\A8\E7\9A\84HIP\E8\AE\BE\E5\A4\87\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"%.1f \00", align 1
@0 = private unnamed_addr constant [14 x i8] c"_Z6reducePfS_\00", align 1
@__hip_fatbin = external constant i8, section ".hip_fatbin"
@__hip_fatbin_wrapper = internal constant { i32, i32, i8*, i8* } { i32 1212764230, i32 1, i8* @__hip_fatbin, i8* null }, section ".hipFatBinSegment", align 8
@__hip_gpubin_handle = linkonce hidden local_unnamed_addr global i8** null, comdat, align 8
@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_reduce_16.cpp, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @__hip_module_ctor, i8* null }]
@str.4 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

declare dso_local void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #0

; Function Attrs: nounwind
declare dso_local void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1)) unnamed_addr #1

; Function Attrs: nofree nounwind
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr #2

; Function Attrs: norecurse uwtable
define dso_local void @_Z21__device_stub__reducePfS_(float* noundef %0, float* noundef %1) #3 {
  %3 = alloca float*, align 8
  %4 = alloca float*, align 8
  %5 = alloca %struct.dim3, align 8
  %6 = alloca %struct.dim3, align 8
  %7 = alloca i64, align 8
  %8 = alloca i8*, align 8
  store float* %0, float** %3, align 8, !tbaa !3
  store float* %1, float** %4, align 8, !tbaa !3
  %9 = alloca [2 x i8*], align 16
  %10 = getelementptr inbounds [2 x i8*], [2 x i8*]* %9, i64 0, i64 0
  %11 = bitcast [2 x i8*]* %9 to float***
  store float** %3, float*** %11, align 16
  %12 = getelementptr inbounds [2 x i8*], [2 x i8*]* %9, i64 0, i64 1
  %13 = bitcast i8** %12 to float***
  store float** %4, float*** %13, align 8
  %14 = call i32 @__hipPopCallConfiguration(%struct.dim3* nonnull %5, %struct.dim3* nonnull %6, i64* nonnull %7, i8** nonnull %8)
  %15 = load i64, i64* %7, align 8
  %16 = bitcast i8** %8 to %struct.ihipStream_t**
  %17 = load %struct.ihipStream_t*, %struct.ihipStream_t** %16, align 8
  %18 = bitcast %struct.dim3* %5 to i64*
  %19 = load i64, i64* %18, align 8
  %20 = getelementptr inbounds %struct.dim3, %struct.dim3* %5, i64 0, i32 2
  %21 = load i32, i32* %20, align 8
  %22 = bitcast %struct.dim3* %6 to i64*
  %23 = load i64, i64* %22, align 8
  %24 = getelementptr inbounds %struct.dim3, %struct.dim3* %6, i64 0, i32 2
  %25 = load i32, i32* %24, align 8
  %26 = call noundef i32 @hipLaunchKernel(i8* noundef bitcast (void (float*, float*)** @_Z6reducePfS_ to i8*), i64 %19, i32 %21, i64 %23, i32 %25, i8** noundef nonnull %10, i64 noundef %15, %struct.ihipStream_t* noundef %17)
  ret void
}

declare dso_local i32 @__hipPopCallConfiguration(%struct.dim3*, %struct.dim3*, i64*, i8**) local_unnamed_addr

declare dso_local i32 @hipLaunchKernel(i8*, i64, i32, i64, i32, i8**, i64, %struct.ihipStream_t*) local_unnamed_addr

; Function Attrs: norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca float*, align 8
  %2 = alloca float*, align 8
  %3 = alloca %struct.dim3, align 8
  %4 = alloca %struct.dim3, align 8
  %5 = alloca i64, align 8
  %6 = alloca i8*, align 8
  %7 = alloca [2 x i8*], align 16
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca float*, align 8
  %11 = alloca float*, align 8
  %12 = bitcast i32* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %12) #10
  %13 = call i32 @hipGetDeviceCount(i32* noundef nonnull %8)
  %14 = load i32, i32* %8, align 4, !tbaa !7
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %45

16:                                               ; preds = %0
  %17 = call noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8* noundef nonnull getelementptr inbounds ([31 x i8], [31 x i8]* @.str, i64 0, i64 0), i64 noundef 30)
  %18 = load i8*, i8** bitcast (%"class.std::basic_ostream"* @_ZSt4cerr to i8**), align 8, !tbaa !9
  %19 = getelementptr i8, i8* %18, i64 -24
  %20 = bitcast i8* %19 to i64*
  %21 = load i64, i64* %20, align 8
  %22 = getelementptr inbounds i8, i8* bitcast (%"class.std::basic_ostream"* @_ZSt4cerr to i8*), i64 %21
  %23 = getelementptr inbounds i8, i8* %22, i64 240
  %24 = bitcast i8* %23 to %"class.std::ctype"**
  %25 = load %"class.std::ctype"*, %"class.std::ctype"** %24, align 8, !tbaa !11
  %26 = icmp eq %"class.std::ctype"* %25, null
  br i1 %26, label %27, label %28

27:                                               ; preds = %16
  call void @_ZSt16__throw_bad_castv() #11
  unreachable

28:                                               ; preds = %16
  %29 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %25, i64 0, i32 8
  %30 = load i8, i8* %29, align 8, !tbaa !20
  %31 = icmp eq i8 %30, 0
  br i1 %31, label %35, label %32

32:                                               ; preds = %28
  %33 = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %25, i64 0, i32 9, i64 10
  %34 = load i8, i8* %33, align 1, !tbaa !23
  br label %41

35:                                               ; preds = %28
  call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* noundef nonnull align 8 dereferenceable(570) %25)
  %36 = bitcast %"class.std::ctype"* %25 to i8 (%"class.std::ctype"*, i8)***
  %37 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %36, align 8, !tbaa !9
  %38 = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %37, i64 6
  %39 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %38, align 8
  %40 = call noundef signext i8 %39(%"class.std::ctype"* noundef nonnull align 8 dereferenceable(570) %25, i8 noundef signext 10)
  br label %41

41:                                               ; preds = %32, %35
  %42 = phi i8 [ %34, %32 ], [ %40, %35 ]
  %43 = call noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) @_ZSt4cerr, i8 noundef signext %42)
  %44 = call noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8) %43)
  br label %287

45:                                               ; preds = %0
  %46 = bitcast i32* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %46) #10
  %47 = call i32 @hipGetDevice(i32* noundef nonnull %9)
  %48 = load i32, i32* %9, align 4, !tbaa !7
  %49 = call i32 @hipSetDevice(i32 noundef %48)
  %50 = call noalias noundef nonnull dereferenceable(512) i8* @_Znam(i64 noundef 512) #12
  %51 = bitcast i8* %50 to float*
  %52 = call noalias noundef nonnull dereferenceable(512) i8* @_Znam(i64 noundef 512) #12
  call void @srand(i32 noundef 1) #10
  %53 = bitcast i8* %50 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %53, align 4, !tbaa !24
  %54 = getelementptr inbounds float, float* %51, i64 4
  %55 = bitcast float* %54 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %55, align 4, !tbaa !24
  %56 = getelementptr inbounds float, float* %51, i64 8
  %57 = bitcast float* %56 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %57, align 4, !tbaa !24
  %58 = getelementptr inbounds float, float* %51, i64 12
  %59 = bitcast float* %58 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %59, align 4, !tbaa !24
  %60 = getelementptr inbounds float, float* %51, i64 16
  %61 = bitcast float* %60 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %61, align 4, !tbaa !24
  %62 = getelementptr inbounds float, float* %51, i64 20
  %63 = bitcast float* %62 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %63, align 4, !tbaa !24
  %64 = getelementptr inbounds float, float* %51, i64 24
  %65 = bitcast float* %64 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %65, align 4, !tbaa !24
  %66 = getelementptr inbounds float, float* %51, i64 28
  %67 = bitcast float* %66 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %67, align 4, !tbaa !24
  %68 = getelementptr inbounds float, float* %51, i64 32
  %69 = bitcast float* %68 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %69, align 4, !tbaa !24
  %70 = getelementptr inbounds float, float* %51, i64 36
  %71 = bitcast float* %70 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %71, align 4, !tbaa !24
  %72 = getelementptr inbounds float, float* %51, i64 40
  %73 = bitcast float* %72 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %73, align 4, !tbaa !24
  %74 = getelementptr inbounds float, float* %51, i64 44
  %75 = bitcast float* %74 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %75, align 4, !tbaa !24
  %76 = getelementptr inbounds float, float* %51, i64 48
  %77 = bitcast float* %76 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %77, align 4, !tbaa !24
  %78 = getelementptr inbounds float, float* %51, i64 52
  %79 = bitcast float* %78 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %79, align 4, !tbaa !24
  %80 = getelementptr inbounds float, float* %51, i64 56
  %81 = bitcast float* %80 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %81, align 4, !tbaa !24
  %82 = getelementptr inbounds float, float* %51, i64 60
  %83 = bitcast float* %82 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %83, align 4, !tbaa !24
  %84 = getelementptr inbounds float, float* %51, i64 64
  %85 = bitcast float* %84 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %85, align 4, !tbaa !24
  %86 = getelementptr inbounds float, float* %51, i64 68
  %87 = bitcast float* %86 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %87, align 4, !tbaa !24
  %88 = getelementptr inbounds float, float* %51, i64 72
  %89 = bitcast float* %88 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %89, align 4, !tbaa !24
  %90 = getelementptr inbounds float, float* %51, i64 76
  %91 = bitcast float* %90 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %91, align 4, !tbaa !24
  %92 = getelementptr inbounds float, float* %51, i64 80
  %93 = bitcast float* %92 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %93, align 4, !tbaa !24
  %94 = getelementptr inbounds float, float* %51, i64 84
  %95 = bitcast float* %94 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %95, align 4, !tbaa !24
  %96 = getelementptr inbounds float, float* %51, i64 88
  %97 = bitcast float* %96 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %97, align 4, !tbaa !24
  %98 = getelementptr inbounds float, float* %51, i64 92
  %99 = bitcast float* %98 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %99, align 4, !tbaa !24
  %100 = getelementptr inbounds float, float* %51, i64 96
  %101 = bitcast float* %100 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %101, align 4, !tbaa !24
  %102 = getelementptr inbounds float, float* %51, i64 100
  %103 = bitcast float* %102 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %103, align 4, !tbaa !24
  %104 = getelementptr inbounds float, float* %51, i64 104
  %105 = bitcast float* %104 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %105, align 4, !tbaa !24
  %106 = getelementptr inbounds float, float* %51, i64 108
  %107 = bitcast float* %106 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %107, align 4, !tbaa !24
  %108 = getelementptr inbounds float, float* %51, i64 112
  %109 = bitcast float* %108 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %109, align 4, !tbaa !24
  %110 = getelementptr inbounds float, float* %51, i64 116
  %111 = bitcast float* %110 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %111, align 4, !tbaa !24
  %112 = getelementptr inbounds float, float* %51, i64 120
  %113 = bitcast float* %112 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %113, align 4, !tbaa !24
  %114 = getelementptr inbounds float, float* %51, i64 124
  %115 = bitcast float* %114 to <4 x float>*
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, <4 x float>* %115, align 4, !tbaa !24
  br label %116

116:                                              ; preds = %45, %116
  %117 = phi i64 [ %199, %116 ], [ 0, %45 ]
  %118 = shl nsw i64 %117, 4
  %119 = getelementptr inbounds float, float* %51, i64 %118
  %120 = load float, float* %119, align 4, !tbaa !24
  %121 = fpext float %120 to double
  %122 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %121)
  %123 = add nuw nsw i64 %118, 1
  %124 = getelementptr inbounds float, float* %51, i64 %123
  %125 = load float, float* %124, align 4, !tbaa !24
  %126 = fpext float %125 to double
  %127 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %126)
  %128 = add nuw nsw i64 %118, 2
  %129 = getelementptr inbounds float, float* %51, i64 %128
  %130 = load float, float* %129, align 4, !tbaa !24
  %131 = fpext float %130 to double
  %132 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %131)
  %133 = add nuw nsw i64 %118, 3
  %134 = getelementptr inbounds float, float* %51, i64 %133
  %135 = load float, float* %134, align 4, !tbaa !24
  %136 = fpext float %135 to double
  %137 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %136)
  %138 = add nuw nsw i64 %118, 4
  %139 = getelementptr inbounds float, float* %51, i64 %138
  %140 = load float, float* %139, align 4, !tbaa !24
  %141 = fpext float %140 to double
  %142 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %141)
  %143 = add nuw nsw i64 %118, 5
  %144 = getelementptr inbounds float, float* %51, i64 %143
  %145 = load float, float* %144, align 4, !tbaa !24
  %146 = fpext float %145 to double
  %147 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %146)
  %148 = add nuw nsw i64 %118, 6
  %149 = getelementptr inbounds float, float* %51, i64 %148
  %150 = load float, float* %149, align 4, !tbaa !24
  %151 = fpext float %150 to double
  %152 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %151)
  %153 = add nuw nsw i64 %118, 7
  %154 = getelementptr inbounds float, float* %51, i64 %153
  %155 = load float, float* %154, align 4, !tbaa !24
  %156 = fpext float %155 to double
  %157 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %156)
  %158 = add nuw nsw i64 %118, 8
  %159 = getelementptr inbounds float, float* %51, i64 %158
  %160 = load float, float* %159, align 4, !tbaa !24
  %161 = fpext float %160 to double
  %162 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %161)
  %163 = add nuw nsw i64 %118, 9
  %164 = getelementptr inbounds float, float* %51, i64 %163
  %165 = load float, float* %164, align 4, !tbaa !24
  %166 = fpext float %165 to double
  %167 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %166)
  %168 = add nuw nsw i64 %118, 10
  %169 = getelementptr inbounds float, float* %51, i64 %168
  %170 = load float, float* %169, align 4, !tbaa !24
  %171 = fpext float %170 to double
  %172 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %171)
  %173 = add nuw nsw i64 %118, 11
  %174 = getelementptr inbounds float, float* %51, i64 %173
  %175 = load float, float* %174, align 4, !tbaa !24
  %176 = fpext float %175 to double
  %177 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %176)
  %178 = add nuw nsw i64 %118, 12
  %179 = getelementptr inbounds float, float* %51, i64 %178
  %180 = load float, float* %179, align 4, !tbaa !24
  %181 = fpext float %180 to double
  %182 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %181)
  %183 = add nuw nsw i64 %118, 13
  %184 = getelementptr inbounds float, float* %51, i64 %183
  %185 = load float, float* %184, align 4, !tbaa !24
  %186 = fpext float %185 to double
  %187 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %186)
  %188 = add nuw nsw i64 %118, 14
  %189 = getelementptr inbounds float, float* %51, i64 %188
  %190 = load float, float* %189, align 4, !tbaa !24
  %191 = fpext float %190 to double
  %192 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %191)
  %193 = add nuw nsw i64 %118, 15
  %194 = getelementptr inbounds float, float* %51, i64 %193
  %195 = load float, float* %194, align 4, !tbaa !24
  %196 = fpext float %195 to double
  %197 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %196)
  %198 = call i32 @putchar(i32 10)
  %199 = add nuw nsw i64 %117, 1
  %200 = icmp eq i64 %199, 8
  br i1 %200, label %201, label %116, !llvm.loop !26

201:                                              ; preds = %116
  %202 = call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([2 x i8], [2 x i8]* @str.4, i64 0, i64 0))
  %203 = bitcast float** %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %203) #10
  %204 = bitcast float** %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %204) #10
  %205 = bitcast float** %10 to i8**
  %206 = call i32 @hipMalloc(i8** noundef nonnull %205, i64 noundef 512)
  %207 = bitcast float** %11 to i8**
  %208 = call i32 @hipMalloc(i8** noundef nonnull %207, i64 noundef 512)
  %209 = load i8*, i8** %205, align 8, !tbaa !3
  %210 = call i32 @hipMemcpy(i8* noundef %209, i8* noundef nonnull %50, i64 noundef 512, i32 noundef 1)
  %211 = call i32 @__hipPushCallConfiguration(i64 4294967298, i32 1, i64 4294967360, i32 1, i64 noundef 0, %struct.ihipStream_t* noundef null)
  %212 = icmp eq i32 %211, 0
  br i1 %212, label %213, label %240

213:                                              ; preds = %201
  %214 = load float*, float** %10, align 8, !tbaa !3
  %215 = load float*, float** %11, align 8, !tbaa !3
  %216 = bitcast float** %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %216)
  %217 = bitcast float** %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %217)
  %218 = bitcast %struct.dim3* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %218)
  %219 = bitcast %struct.dim3* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %219)
  %220 = bitcast i64* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %220)
  %221 = bitcast i8** %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %221)
  %222 = bitcast [2 x i8*]* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %222)
  store float* %214, float** %1, align 8, !tbaa !3
  store float* %215, float** %2, align 8, !tbaa !3
  %223 = getelementptr inbounds [2 x i8*], [2 x i8*]* %7, i64 0, i64 0
  %224 = bitcast [2 x i8*]* %7 to float***
  store float** %1, float*** %224, align 16
  %225 = getelementptr inbounds [2 x i8*], [2 x i8*]* %7, i64 0, i64 1
  %226 = bitcast i8** %225 to float***
  store float** %2, float*** %226, align 8
  %227 = call i32 @__hipPopCallConfiguration(%struct.dim3* nonnull %3, %struct.dim3* nonnull %4, i64* nonnull %5, i8** nonnull %6)
  %228 = load i64, i64* %5, align 8
  %229 = bitcast i8** %6 to %struct.ihipStream_t**
  %230 = load %struct.ihipStream_t*, %struct.ihipStream_t** %229, align 8
  %231 = bitcast %struct.dim3* %3 to i64*
  %232 = load i64, i64* %231, align 8
  %233 = getelementptr inbounds %struct.dim3, %struct.dim3* %3, i64 0, i32 2
  %234 = load i32, i32* %233, align 8
  %235 = bitcast %struct.dim3* %4 to i64*
  %236 = load i64, i64* %235, align 8
  %237 = getelementptr inbounds %struct.dim3, %struct.dim3* %4, i64 0, i32 2
  %238 = load i32, i32* %237, align 8
  %239 = call noundef i32 @hipLaunchKernel(i8* noundef bitcast (void (float*, float*)** @_Z6reducePfS_ to i8*), i64 %232, i32 %234, i64 %236, i32 %238, i8** noundef nonnull %223, i64 noundef %228, %struct.ihipStream_t* noundef %230)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %216)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %217)
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %218)
  call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %219)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %220)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %221)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %222)
  br label %240

240:                                              ; preds = %213, %201
  %241 = load i8*, i8** %207, align 8, !tbaa !3
  %242 = call i32 @hipMemcpy(i8* noundef nonnull %52, i8* noundef %241, i64 noundef 512, i32 noundef 2)
  %243 = load float*, float** %11, align 8, !tbaa !3
  %244 = load float, float* %243, align 4, !tbaa !24
  %245 = fpext float %244 to double
  %246 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %245)
  %247 = load float*, float** %11, align 8, !tbaa !3
  %248 = getelementptr inbounds float, float* %247, i64 4
  %249 = load float, float* %248, align 4, !tbaa !24
  %250 = fpext float %249 to double
  %251 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %250)
  %252 = load float*, float** %11, align 8, !tbaa !3
  %253 = getelementptr inbounds float, float* %252, i64 8
  %254 = load float, float* %253, align 4, !tbaa !24
  %255 = fpext float %254 to double
  %256 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %255)
  %257 = load float*, float** %11, align 8, !tbaa !3
  %258 = getelementptr inbounds float, float* %257, i64 12
  %259 = load float, float* %258, align 4, !tbaa !24
  %260 = fpext float %259 to double
  %261 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %260)
  %262 = load float*, float** %11, align 8, !tbaa !3
  %263 = getelementptr inbounds float, float* %262, i64 16
  %264 = load float, float* %263, align 4, !tbaa !24
  %265 = fpext float %264 to double
  %266 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %265)
  %267 = load float*, float** %11, align 8, !tbaa !3
  %268 = getelementptr inbounds float, float* %267, i64 20
  %269 = load float, float* %268, align 4, !tbaa !24
  %270 = fpext float %269 to double
  %271 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %270)
  %272 = load float*, float** %11, align 8, !tbaa !3
  %273 = getelementptr inbounds float, float* %272, i64 24
  %274 = load float, float* %273, align 4, !tbaa !24
  %275 = fpext float %274 to double
  %276 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %275)
  %277 = load float*, float** %11, align 8, !tbaa !3
  %278 = getelementptr inbounds float, float* %277, i64 28
  %279 = load float, float* %278, align 4, !tbaa !24
  %280 = fpext float %279 to double
  %281 = call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([6 x i8], [6 x i8]* @.str.1, i64 0, i64 0), double noundef %280)
  %282 = call i32 @puts(i8* nonnull dereferenceable(1) getelementptr inbounds ([2 x i8], [2 x i8]* @str.4, i64 0, i64 0))
  call void @_ZdaPv(i8* noundef nonnull %50) #13
  %283 = load i8*, i8** %205, align 8, !tbaa !3
  %284 = call i32 @hipFree(i8* noundef %283)
  call void @_ZdaPv(i8* noundef nonnull %52) #13
  %285 = load i8*, i8** %207, align 8, !tbaa !3
  %286 = call i32 @hipFree(i8* noundef %285)
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %204) #10
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %203) #10
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %46) #10
  br label %287

287:                                              ; preds = %240, %41
  %288 = phi i32 [ 1, %41 ], [ 0, %240 ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %12) #10
  ret i32 %288
}

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

declare dso_local i32 @hipGetDeviceCount(i32* noundef) local_unnamed_addr #0

declare dso_local i32 @hipGetDevice(i32* noundef) local_unnamed_addr #0

declare dso_local i32 @hipSetDevice(i32 noundef) local_unnamed_addr #0

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull i8* @_Znam(i64 noundef) local_unnamed_addr #5

; Function Attrs: nounwind
declare dso_local void @srand(i32 noundef) local_unnamed_addr #1

; Function Attrs: argmemonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: nofree nounwind
declare dso_local noundef i32 @printf(i8* nocapture noundef readonly, ...) local_unnamed_addr #6

declare dso_local i32 @hipMemcpy(i8* noundef, i8* noundef, i64 noundef, i32 noundef) local_unnamed_addr #0

declare dso_local i32 @__hipPushCallConfiguration(i64, i32, i64, i32, i64 noundef, %struct.ihipStream_t* noundef) local_unnamed_addr #0

; Function Attrs: nobuiltin nounwind
declare dso_local void @_ZdaPv(i8* noundef) local_unnamed_addr #7

declare dso_local i32 @hipFree(i8* noundef) local_unnamed_addr #0

declare dso_local noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8), i8* noundef, i64 noundef) local_unnamed_addr #0

declare dso_local noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8), i8 noundef signext) local_unnamed_addr #0

declare dso_local noundef nonnull align 8 dereferenceable(8) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* noundef nonnull align 8 dereferenceable(8)) local_unnamed_addr #0

; Function Attrs: noreturn
declare dso_local void @_ZSt16__throw_bad_castv() local_unnamed_addr #8

declare dso_local void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* noundef nonnull align 8 dereferenceable(570)) local_unnamed_addr #0

declare dso_local i32 @hipMalloc(i8** noundef, i64 noundef) local_unnamed_addr #0

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_reduce_16.cpp() #9 section ".text.startup" {
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* noundef nonnull align 1 dereferenceable(1) @_ZStL8__ioinit)
  %1 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #10
  ret void
}

declare dso_local i32 @__hipRegisterFunction(i8**, i8*, i8*, i8*, i32, i8*, i8*, i8*, i8*, i32*) local_unnamed_addr

declare dso_local i8** @__hipRegisterFatBinary(i8*) local_unnamed_addr

define internal void @__hip_module_ctor() {
  %1 = load i8**, i8*** @__hip_gpubin_handle, align 8
  %2 = icmp eq i8** %1, null
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = tail call i8** @__hipRegisterFatBinary(i8* bitcast ({ i32, i32, i8*, i8* }* @__hip_fatbin_wrapper to i8*))
  store i8** %4, i8*** @__hip_gpubin_handle, align 8
  br label %5

5:                                                ; preds = %3, %0
  %6 = phi i8** [ %4, %3 ], [ %1, %0 ]
  %7 = tail call i32 @__hipRegisterFunction(i8** %6, i8* bitcast (void (float*, float*)** @_Z6reducePfS_ to i8*), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @0, i64 0, i64 0), i8* getelementptr inbounds ([14 x i8], [14 x i8]* @0, i64 0, i64 0), i32 -1, i8* null, i8* null, i8* null, i8* null, i32* null)
  %8 = tail call i32 @atexit(void ()* nonnull @__hip_module_dtor)
  ret void
}

declare dso_local void @__hipUnregisterFatBinary(i8**) local_unnamed_addr

define internal void @__hip_module_dtor() {
  %1 = load i8**, i8*** @__hip_gpubin_handle, align 8
  %2 = icmp eq i8** %1, null
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @__hipUnregisterFatBinary(i8** nonnull %1)
  store i8** null, i8*** @__hip_gpubin_handle, align 8
  br label %4

4:                                                ; preds = %3, %0
  ret void
}

declare dso_local i32 @atexit(void ()*) local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @puts(i8* nocapture noundef readonly) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #2

attributes #0 = { "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nofree nounwind }
attributes #3 = { norecurse uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #5 = { nobuiltin allocsize(0) "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nofree nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #7 = { nobuiltin nounwind "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #8 = { noreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #9 = { uwtable "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #10 = { nounwind }
attributes #11 = { noreturn }
attributes #12 = { builtin allocsize(0) }
attributes #13 = { builtin nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"clang version 15.0.0"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
!9 = !{!10, !10, i64 0}
!10 = !{!"vtable pointer", !6, i64 0}
!11 = !{!12, !4, i64 240}
!12 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !13, i64 0, !4, i64 216, !5, i64 224, !19, i64 225, !4, i64 232, !4, i64 240, !4, i64 248, !4, i64 256}
!13 = !{!"_ZTSSt8ios_base", !14, i64 8, !14, i64 16, !15, i64 24, !16, i64 28, !16, i64 32, !4, i64 40, !17, i64 48, !5, i64 64, !8, i64 192, !4, i64 200, !18, i64 208}
!14 = !{!"long", !5, i64 0}
!15 = !{!"_ZTSSt13_Ios_Fmtflags", !5, i64 0}
!16 = !{!"_ZTSSt12_Ios_Iostate", !5, i64 0}
!17 = !{!"_ZTSNSt8ios_base6_WordsE", !4, i64 0, !14, i64 8}
!18 = !{!"_ZTSSt6locale", !4, i64 0}
!19 = !{!"bool", !5, i64 0}
!20 = !{!21, !5, i64 56}
!21 = !{!"_ZTSSt5ctypeIcE", !22, i64 0, !4, i64 16, !19, i64 24, !4, i64 32, !4, i64 40, !4, i64 48, !5, i64 56, !5, i64 57, !5, i64 313, !5, i64 569}
!22 = !{!"_ZTSNSt6locale5facetE", !8, i64 8}
!23 = !{!5, !5, i64 0}
!24 = !{!25, !25, i64 0}
!25 = !{!"float", !5, i64 0}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.mustprogress"}

; __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu
