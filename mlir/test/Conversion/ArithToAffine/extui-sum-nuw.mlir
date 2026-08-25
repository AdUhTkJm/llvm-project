// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// Same as extui-sum.mlir, but the addition carries the `nuw` flag. The known
// fact 0 <= u(a) + u(b) < 2^33 (with u(x) = x mod 2^33) lets the reference
// relation eliminate every constraint row of the addition itself, including
// its local variable definitions. It does not imply the bound on the GEP
// offset, 4 * ((i mod 2^32) + (j mod 2^32)) <= 2^32 - 1 under the signed
// interpretation of i33, so the modulo locals are still materialized and
// this last check remains:
// -2147483648 <= (i mod 2^32) + (j mod 2^32) <= 1073741823,
// i.e. 4*((i mod 2^32) + (j mod 2^32)) <= 2^32 - 1 after GCD tightening.

// CHECK:       #map = affine_map<()[s0, s1] -> ((s0 mod 4294967296 + s1 mod 4294967296) * 4)>
// CHECK-LABEL: llvm.func @load_extui_sum(
// CHECK:         affine.apply
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK:         arith.constant 32 : i35
// CHECK:         arith.shrsi
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK-NOT:     arith.select
// CHECK:         arith.constant 32 : i35
// CHECK:         arith.shrsi
// CHECK-NOT:     arith.select
// CHECK:         arith.constant 1073741823 : i36
// CHECK:         arith.cmpi sge
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_extui_sum(%i: i32, %j: i32) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %a = arith.extui %i : i32 to i33
    %b = arith.extui %j : i32 to i33
    %sum = arith.addi %a, %b overflow<nuw> : i33
    %addr = llvm.getelementptr %base[%sum] : (!llvm.ptr, i33) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
