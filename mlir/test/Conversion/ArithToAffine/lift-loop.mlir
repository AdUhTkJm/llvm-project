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
    %c1 = arith.constant 1 : i32
    %sum = scf.for %i = %c0 to %ub step %c1 iter_args(%acc = %c0) -> i32 : i32 {
      %addr = llvm.getelementptr %base[%i] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %addr : !llvm.ptr -> i32
      %next = arith.addi %acc, %v : i32
      scf.yield %next : i32
    }
    llvm.return %sum : i32
  }
}
