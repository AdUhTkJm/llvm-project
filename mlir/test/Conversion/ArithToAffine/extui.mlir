// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// The zero extension is represented with a modulo by 2^32 in the affine map,
// treating the signed symbols as unsigned. The overflow checks only need to
// guarantee that the addition does not overflow under the signed
// interpretation: -2147483648 <= %arg0 + %arg1 <= 2147483647. The rows
// pinning the modulo's local variable hold by construction, and the bounds
// of the GEP sum are implied by them, so no division is synthesized.

// CHECK:       #map = affine_map<()[s0, s1] -> (((s0 + s1) mod 4294967296) * 4)>
// CHECK-LABEL: llvm.func @load_extui(
// CHECK:         affine.apply
// CHECK-NOT:     arith.divsi
// CHECK:         arith.constant 2147483648 : i34
// CHECK:         arith.cmpi sge
// CHECK:         arith.constant 2147483647 : i34
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
