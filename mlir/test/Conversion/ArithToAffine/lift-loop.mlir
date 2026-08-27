// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// A single load inside a loop with an affine address expression.
// CHECK:       #map = affine_map<()[s0] -> (s0 * 4)>
// CHECK-LABEL: llvm.func @load_in_loop
// The lifted branch uses affine.apply, the fallback uses plain GEP.
// CHECK:       scf.if
// CHECK:       scf.for
// CHECK:         affine.apply
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:       } else {
// CHECK:       scf.for
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
module {
  llvm.func @load_in_loop(%base: !llvm.ptr, %ub: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c0_index = arith.constant 0 : index
    %ub_idx = arith.index_cast %ub : i32 to index
    %sum = scf.for %i = %c0_index to %ub_idx step %c1 iter_args(%acc = %c0) -> i32 {
      %ext = arith.index_cast %i : index to i32
      %addr = llvm.getelementptr %base[%ext] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %addr : !llvm.ptr -> i32
      %next = arith.addi %acc, %v : i32
      scf.yield %next : i32
    }
    llvm.return %sum : i32
  }
}
