; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i32 @say_hello()

define i32 @main() {
  %1 = call i32 @say_hello()
  ret i32 %1
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
