// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// The sign extension pins the interpretation of the computation of the GEP
// index to signed, so the synthesized overflow checks use the signed bounds
// of i32: 0 <= %arg0 + %arg1 (from the unsigned GEP expression, after GCD
// normalization) and %arg0 + %arg1 <= 2147483647 (signed max of i32).

// CHECK-LABEL: llvm.func @load_extsi(
// CHECK:         affine.apply
// CHECK:         arith.cmpi sge
// CHECK:         arith.constant 2147483647 : i34
// CHECK:         arith.cmpi sge
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_extsi(%i: i32, %j: i32) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %sum = arith.addi %i, %j : i32
    %ext = arith.extsi %sum : i32 to i64
    %addr = llvm.getelementptr %base[%ext] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
