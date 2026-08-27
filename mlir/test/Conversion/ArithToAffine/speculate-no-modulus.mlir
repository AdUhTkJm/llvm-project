// RUN: mlir-opt -convert-arith-to-affine=speculate-no-modulus %s | FileCheck %s

// With speculation, the zero extensions involve no modulo in the affine maps.
// Instead, the synthesized branches additionally check the operands of the
// zero extensions to be non-negative, under which their signed and unsigned
// interpretations coincide.

// CHECK:       #map = affine_map<()[s0, s1] -> ((s0 + s1) * 4)>
// CHECK:       #map1 = affine_map<()[s0] -> (s0 * 4)>

// The branch only needs to guarantee that the addition does not overflow under
// the signed interpretation: 0 <= s0, 0 <= s1, and
// 4 * (s0 + s1) <= 2^32 - 1 after GCD tightening. Since no local variable for
// a modulo is introduced, no division is synthesized either.

// CHECK-LABEL: llvm.func @load_extui_sum(
// CHECK:         affine.apply
// CHECK-NOT:     arith.shrsi
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK:         arith.constant 0 : i33
// CHECK:         arith.cmpi sge
// CHECK:         arith.constant 0 : i33
// CHECK:         arith.cmpi sge
// CHECK:         arith.constant 1073741823 : i34
// CHECK:         arith.cmpi sge
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_extui_sum(%i: i32, %j: i32) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %a = arith.extui %i : i32 to i33
    %b = arith.extui %j : i32 to i33
    %sum = arith.addi %a, %b : i33
    %addr = llvm.getelementptr %base[%sum] : (!llvm.ptr, i33) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}

// The sign extension passes its operand through unchanged, so the map involves
// no modulo regardless. The zero extension on top of it requires its operand
// to be non-negative; since that operand is the sign extension itself, a
// single comparison remains.

// CHECK-LABEL: llvm.func @load_extsi_extui(
// CHECK:         affine.apply
// CHECK:         arith.constant 0 : i9
// CHECK:         arith.cmpi sge
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
module {
  llvm.func @load_extsi_extui(%i: i8) -> i32 {
    %one = arith.constant 1 : i32
    %base = llvm.alloca %one x i32 : (i32) -> !llvm.ptr
    %ext = arith.extsi %i : i8 to i16
    %ext2 = arith.extui %ext : i16 to i32
    %addr = llvm.getelementptr %base[%ext2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
