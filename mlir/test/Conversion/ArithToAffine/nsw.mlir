// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// Same as extsi.mlir, but the `nsw` flag on the addition already guarantees
// that %arg0 + %arg1 does not signed-overflow. The known fact
// -2147483648 <= %arg0 + %arg1 <= 2147483647 is recorded in the reference
// relation, so the synthesized overflow checks are eliminated and the
// condition folds to true.

// CHECK:       #map = affine_map<()[s0, s1] -> ((s0 + s1) * 4)>
// CHECK-LABEL: llvm.func @load_nsw(
// CHECK:         affine.apply
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_nsw(%i: i32, %j: i32) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %sum = arith.addi %i, %j overflow<nsw> : i32
    %ext = arith.extsi %sum : i32 to i64
    %addr = llvm.getelementptr %base[%ext] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
