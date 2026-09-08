// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// Tests for nested loops. The pass computes a single unified lifting plan for
// every outermost loop: the lifting constraints of all the loads in the nest
// (including those of nested loops) are intersected, the symbols defined
// inside the nest (e.g. the induction variables) are substituted by their
// worst-case bounds in terms of the values before the outer loop, and a
// single check is emitted before the outer loop:
//   if (check) { lifted loop nest } else { original loop nest }
// so that the whole nest is raised at most once and the synthesized condition
// never violates dominance.

// CHECK:       #map = affine_map<()[s0] -> (s0 * 4)>
// CHECK:       #map1 = affine_map<()[s0, s1] -> ((s0 * 16 + s1) * 4)>
// CHECK:       #map2 = affine_map<()[s0, s1] -> ((s0 * 4 + s1) * 4)>
// CHECK:       #map3 = affine_map<()[s0, s1, s2] -> (((s0 * 4 + s1) * 4 + s2) * 4)>
// CHECK:       #map4 = affine_map<()[s0, s1] -> ((s0 + s1) * 4)>
// CHECK:       #map5 = affine_map<()[s0] -> ((s0 mod 4294967296) * 4)>

// The inner loop bound depends on the outer induction variable (j < i). The
// constraint on j is substituted by its worst-case bound in terms of the
// values before the outer loop (j <= i - 1 and i's bound in terms of %n), so
// the unified check only constrains %n: 4 * j <= 2^63 - 1 becomes a single
// comparison n <= 2305843009213693952 = 2^61.
// CHECK-LABEL: func.func @triangular
// CHECK:         arith.constant 2305843009213693952 : i65
// CHECK:         arith.extsi %[[N:.*]] : i64 to i65
// CHECK:         arith.cmpi sge, {{.*}} : i65
// CHECK:         scf.if
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               affine.apply #map
// CHECK:               llvm.load
// CHECK:           } else {
// CHECK:             scf.for
// CHECK:             scf.for
// CHECK-NOT:           affine.apply
// CHECK:               llvm.load
module {
  func.func @triangular(%base: !llvm.ptr, %n: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    scf.for %i = %c0 to %n step %c1 : i64 {
      scf.for %j = %c0 to %i step %c1 : i64 {
        %p = llvm.getelementptr %base[%j] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }

// The address 16*i + j mixes both induction variables. Both are defined
// inside the nest, so both are substituted by their worst-case bounds in
// terms of the loop inputs, and the row 64*i + 4*j <= 2^63 - 1 (the GEP
// offset 4*(16*i + j)) is normalized to 16*n + m <= (2^63 - 1) / 4.
// CHECK-LABEL: func.func @flat_2d
// CHECK:         arith.constant 2305843009213693951 : i69
// CHECK:         arith.constant -16 : i69
// CHECK:         arith.extsi %[[N:.*]] : i64 to i69
// CHECK:         arith.extsi %[[M:.*]] : i64 to i69
// CHECK:         arith.cmpi sge, {{.*}} : i69
// CHECK:         scf.if
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               affine.apply #map1
// CHECK:               llvm.load
// CHECK:           } else {
// CHECK:             scf.for
// CHECK:             scf.for
// CHECK-NOT:           affine.apply
// CHECK:               llvm.load
  func.func @flat_2d(%base: !llvm.ptr, %n: i64, %m: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c16 = arith.constant 16 : i64
    scf.for %i = %c0 to %n step %c1 : i64 {
      %off = arith.muli %i, %c16 : i64
      scf.for %j = %c0 to %m step %c1 : i64 {
        %idx = arith.addi %off, %j : i64
        %p = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }

// Small constant trip counts: 4*i + j stays in [0, 15], which cannot
// overflow i64. The constraints are implied by the loop bounds, so the load
// is lifted in place with no runtime check at all.
// CHECK-LABEL: func.func @constant_trip_counts
// CHECK-NOT:     scf.if
// CHECK:         affine.apply #map2
// CHECK-NOT:     scf.if
// CHECK:         llvm.load
  func.func @constant_trip_counts(%base: !llvm.ptr) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c4 = arith.constant 4 : i64
    scf.for %i = %c0 to %c4 step %c1 : i64 {
      scf.for %j = %c0 to %c4 step %c1 : i64 {
        %ij = arith.muli %i, %c4 : i64
        %idx = arith.addi %ij, %j : i64
        %p = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }

// A triple nest with constant trip counts: (4*i + j)*4 + k stays in
// [0, 63], so the load is lifted in place over three symbols, again without
// any runtime check.
// CHECK-LABEL: func.func @triple_nest
// CHECK-NOT:     scf.if
// CHECK:         affine.apply #map3
// CHECK-NOT:     scf.if
// CHECK:         llvm.load
  func.func @triple_nest(%base: !llvm.ptr) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c4 = arith.constant 4 : i64
    scf.for %i = %c0 to %c4 step %c1 : i64 {
      scf.for %j = %c0 to %c4 step %c1 : i64 {
        scf.for %k = %c0 to %c4 step %c1 : i64 {
          %ij = arith.muli %i, %c4 : i64
          %ij2 = arith.addi %ij, %j : i64
          %ij4 = arith.muli %ij2, %c4 : i64
          %idx = arith.addi %ij4, %k : i64
          %p = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i32
          %v = llvm.load %p : !llvm.ptr -> i32
        }
      }
    }
    func.return
  }

// A load directly in the outer loop body plus a load in the inner loop. The
// unified check covers both loads, so a single scf.if surrounds the whole
// nest and the inner loop is lifted together with the outer one: the
// condition i + j <= (2^63 - 1) / 4 (over the substituted bounds n + m)
// guarantees that neither address computation overflows.
// CHECK-LABEL: func.func @loads_at_both_levels
// CHECK:         arith.constant 2305843009213693951 : i66
// CHECK:         scf.if
// CHECK:           scf.for
// CHECK:             affine.apply #map
// CHECK:             llvm.load
// CHECK:             scf.for
// CHECK:               affine.apply #map4
// CHECK:               llvm.load
// CHECK:           } else {
// CHECK:             scf.for
// CHECK:             scf.for
// CHECK-NOT:           scf.if
// CHECK:               llvm.load
  func.func @loads_at_both_levels(%base: !llvm.ptr, %n: i64, %m: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    scf.for %i = %c0 to %n step %c1 : i64 {
      %po = llvm.getelementptr %base[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %vo = llvm.load %po : !llvm.ptr -> i32
      scf.for %j = %c0 to %m step %c1 : i64 {
        %idx = arith.addi %i, %j : i64
        %pi = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %vi = llvm.load %pi : !llvm.ptr -> i32
      }
    }
    func.return
  }

// The product i*m of two symbols is not affine, so nothing is lifted.
// CHECK-LABEL: func.func @non_affine_product
// CHECK-NOT:     affine.apply
// CHECK-NOT:     scf.if
// CHECK:         return
  func.func @non_affine_product(%base: !llvm.ptr, %n: i64, %m: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    scf.for %i = %c0 to %n step %c1 : i64 {
      scf.for %j = %c0 to %m step %c1 : i64 {
        %im = arith.muli %i, %m : i64
        %idx = arith.addi %im, %j : i64
        %p = llvm.getelementptr %base[%idx] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }

// The zero extension of the inner iv introduces a floordiv local variable
// over an inner symbol, whose definition cannot be materialized before the
// outer loop. Hoisting fails and the pass falls back to lifting the load
// individually, with a per-load scf.if inside the inner loop body.
// CHECK-LABEL: func.func @extui_inner_fallback
// CHECK:         scf.for
// CHECK-NOT:     scf.if
// CHECK:           scf.for
// CHECK-NOT:     scf.if
// CHECK:             affine.apply #map5
// CHECK:             scf.if
// CHECK:               llvm.getelementptr {{.*}}i8
// CHECK:             } else {
// CHECK:             llvm.load
  func.func @extui_inner_fallback(%base: !llvm.ptr, %n: i32, %m: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      scf.for %j = %c0 to %m step %c1 : i32 {
        %je = arith.extui %j : i32 to i64
        %p = llvm.getelementptr %base[%je] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }

// i32 arithmetic in a nested loop. Without overflow flags, the unified check
// must guarantee that i + j does not overflow i32. The lower row is implied
// by the loop bounds 0 <= i, j, and the induction variables are substituted
// by their bounds in terms of the loop inputs, so a single comparison
// remains: n + m <= 2^31 - 1.
// CHECK-LABEL: func.func @i32_sum_no_flag
// CHECK:         arith.constant 2147483647 : i34
// CHECK:         arith.extsi %[[N:.*]] : i32 to i34
// CHECK:         arith.extsi %[[M:.*]] : i32 to i34
// CHECK:         arith.cmpi sge, {{.*}} : i34
// CHECK:         scf.if
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               affine.apply #map4
// CHECK:               llvm.load
// CHECK:           } else {
// CHECK:             scf.for
// CHECK:             scf.for
// CHECK-NOT:           affine.apply
// CHECK:               llvm.load
  func.func @i32_sum_no_flag(%base: !llvm.ptr, %n: i32, %m: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      scf.for %j = %c0 to %m step %c1 : i32 {
        %idx = arith.addi %i, %j : i32
        %idx64 = arith.extsi %idx : i32 to i64
        %p = llvm.getelementptr %base[%idx64] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }

// Same computation with the nsw flag: the flag makes the no-overflow
// constraint hold unconditionally (the addition is poison otherwise), so the
// check rows are implied and the load is lifted in place with no scf.if.
// CHECK-LABEL: func.func @i32_sum_nsw
// CHECK-NOT:     scf.if
// CHECK:         affine.apply #map4
// CHECK-NOT:     scf.if
// CHECK:         llvm.load
  func.func @i32_sum_nsw(%base: !llvm.ptr, %n: i32, %m: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %n step %c1 : i32 {
      scf.for %j = %c0 to %m step %c1 : i32 {
        %idx = arith.addi %i, %j overflow<nsw> : i32
        %idx64 = arith.extsi %idx : i32 to i64
        %p = llvm.getelementptr %base[%idx64] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
      }
    }
    func.return
  }
}
