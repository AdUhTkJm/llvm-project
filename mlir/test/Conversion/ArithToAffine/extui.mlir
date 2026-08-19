// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// The zero extension pins the interpretation of the computation of the GEP
// index to unsigned. The unsigned upper-bound check (%arg0 + %arg1 <=
// 4294967295) is redundant given the analyzed ranges of the operands, so
// only the lower-bound check %arg0 + %arg1 >= 0 is synthesized.

// CHECK-LABEL: llvm.func @load_extui(
// CHECK:         affine.apply
// CHECK-NOT:     arith.constant 4294967295
// CHECK:         arith.cmpi sge
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_extui(%i: i32, %j: i32) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %sum = arith.addi %i, %j : i32
    %ext = arith.extui %sum : i32 to i64
    %addr = llvm.getelementptr %base[%ext] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
