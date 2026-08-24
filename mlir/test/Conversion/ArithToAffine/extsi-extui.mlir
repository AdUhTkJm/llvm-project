// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// The sign extension pins the interpretation of the i8 symbol to signed, and
// the zero extension to i32 is represented with a modulo by 2^16. The
// sign-extended i8 value cannot overflow i16, and the bounds of the GEP sum
// are implied by the modulo's defining rows, so the synthesized condition
// folds to true.

// CHECK:       #map = affine_map<()[s0] -> ((s0 mod 65536) * 4)>
// CHECK-LABEL: llvm.func @load_extui(
// CHECK:         affine.apply
// CHECK-NOT:     arith.cmpi
// CHECK:         scf.if
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
