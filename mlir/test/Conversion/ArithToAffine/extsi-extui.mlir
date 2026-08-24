module {
  llvm.func @load_extui(%i: i8) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %ext = arith.extsi %i : i8 to i16
    %ext2 = arith.extui %ext : i16 to i32
    %addr = llvm.getelementptr %base[%ext2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}