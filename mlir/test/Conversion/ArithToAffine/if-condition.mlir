// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// An scf.if whose condition bounds a value contributes constraints that let
// the pass drop runtime overflow-check rows for a load guarded by that
// condition. Each function loads in[i - 1] (i32 element, 4 bytes) with the
// enclosing condition narrowed step by step.

// Shared map for the in[i - 1] load.
// CHECK:       #map = affine_map<()[s0] -> ((s0 - 1) * 4)>

// No enclosing condition: both the lower and upper rows need a runtime check.
// CHECK-LABEL: func.func @baseline
// CHECK:         affine.apply #map
// CHECK:         arith.cmpi sge, %[[LO:.*]], %[[C0:.*]] : i65
// CHECK:         arith.cmpi sge, %[[HI:.*]], %[[C1:.*]] : i65
// CHECK:         scf.if
// CHECK:         llvm.load

// The enclosing condition proves i >= 0: only the upper row still needs a
// runtime check.
// CHECK-LABEL: func.func @partial
// CHECK:         affine.apply #map
// CHECK:         arith.cmpi sge, %[[HI:.*]], %[[C:.*]] : i65
// CHECK-NOT:     i65
// CHECK:         llvm.load

// The enclosing condition proves 0 <= i < 100: both rows are implied, so no
// runtime check is synthesized.
// CHECK-LABEL: func.func @full
// CHECK:         affine.apply #map
// CHECK-NOT:     i65
// CHECK:         llvm.load
module {
  func.func @baseline(%in: !llvm.ptr, %i: i64) -> i32 {
    %c1 = arith.constant 1 : i64
    %iprev = llvm.sub %i, %c1 : i64
    %p = llvm.getelementptr %in[%iprev] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %v = llvm.load %p : !llvm.ptr -> i32
    return %v : i32
  }
  func.func @partial(%in: !llvm.ptr, %i: i64) -> i32 {
    %c0 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cond = arith.cmpi sge, %i, %c0 : i64
    %r = scf.if %cond -> (i32) {
      %c1 = arith.constant 1 : i64
      %iprev = llvm.sub %i, %c1 : i64
      %p = llvm.getelementptr %in[%iprev] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      scf.yield %v : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    return %r : i32
  }
  func.func @full(%in: !llvm.ptr, %i: i64) -> i32 {
    %c0 = arith.constant 0 : i64
    %c100 = arith.constant 100 : i64
    %c0_i32 = arith.constant 0 : i32
    %lo = arith.cmpi sge, %i, %c0 : i64
    %hi = arith.cmpi slt, %i, %c100 : i64
    %cond = arith.andi %lo, %hi : i1
    %r = scf.if %cond -> (i32) {
      %c1 = arith.constant 1 : i64
      %iprev = llvm.sub %i, %c1 : i64
      %p = llvm.getelementptr %in[%iprev] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      scf.yield %v : i32
    } else {
      scf.yield %c0_i32 : i32
    }
    return %r : i32
  }
}
