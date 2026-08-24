// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// Same as extui.mlir, but the `nuw` flag on the addition records the known
// fact 0 <= (%arg0 mod 2^32) + (%arg1 mod 2^32) < 2^32, from which the rows
// bounding the modulo and its local variables follow. Only the signed-range
// checks on the sum itself remain:
// -2147483648 <= %arg0 + %arg1 <= 2147483647.

// CHECK:       #map = affine_map<()[s0, s1] -> (((s0 + s1) mod 4294967296) * 4)>
// CHECK-LABEL: llvm.func @load_nuw(
// CHECK:         affine.apply
// CHECK-NOT:     arith.divsi
// CHECK:         arith.constant 2147483648 : i34
// CHECK:         arith.cmpi sge
// CHECK:         arith.constant 2147483647 : i34
// CHECK:         arith.cmpi sge
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_nuw(%i: i32, %j: i32) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %sum = arith.addi %i, %j overflow<nuw> : i32
    %ext = arith.extui %sum : i32 to i64
    %addr = llvm.getelementptr %base[%ext] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
