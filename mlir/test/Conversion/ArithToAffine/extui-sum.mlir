// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// The sum of two zero extensions is represented with two modulo locals. The
// upper-bound check of the GEP sum is not implied by the defining rows of
// the locals, so the locals are materialized at runtime as floor divisions
// (divsi/remsi truncated towards zero, corrected by a select), and the check
// (ri + rj <= 1073741823, i.e. 4*(ri + rj) <= 2^32 - 1 after GCD tightening)
// is evaluated with them. The defining rows of the locals themselves hold by
// construction and are not checked.

// CHECK:       #map = affine_map<()[s0, s1] -> ((s0 mod 4294967296 + s1 mod 4294967296) * 4)>
// CHECK-LABEL: llvm.func @load_extui_sum(
// CHECK:         affine.apply
// CHECK:         arith.constant 4294967296 : i35
// CHECK:         arith.divsi
// CHECK:         arith.remsi
// CHECK:         arith.cmpi slt
// CHECK:         arith.select
// CHECK:         arith.constant 4294967296 : i35
// CHECK:         arith.divsi
// CHECK:         arith.remsi
// CHECK:         arith.cmpi slt
// CHECK:         arith.select
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
    %sum = arith.addi %a, %b : i33
    %addr = llvm.getelementptr %base[%sum] : (!llvm.ptr, i33) -> !llvm.ptr, i32
    %v = llvm.load %addr : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}
