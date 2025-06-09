module {
  llvm.func @say_hello() -> i32 attributes {sym_visibility = "private"}
  llvm.func @main() -> i32 {
    %0 = llvm.call @say_hello() : () -> i32
    llvm.return %0 : i32
  }
}

