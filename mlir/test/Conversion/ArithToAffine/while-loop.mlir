// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// Tests for do-while style scf.while loops. The pass extracts the symbolic
// constraints of the induction variable manually (scf.while does not
// implement ValueBoundsOpInterface): the induction variable is recognized
// from the scf.condition at the end of the first region, whose operand is
// updated from the before-region argument by a constant step and yielded
// back unchanged through the after region, and the loop condition compares
// the forwarded operand against a bound defined outside the loop. Under the
// counted-loop assumption, the entries of the induction variable are bounded
// by %init and %bound, which lets the unified check hoisted before the loop
// drop the rows they imply.

// Shared map for the base + iv load of the first seven functions.
// CHECK:       #map = affine_map<()[s0] -> (s0 * 4)>

// Constant init and bound: the extracted facts 0 <= iv <= bound - step imply
// both no-overflow rows of the i32 address expression, so the load is lifted
// in place with no runtime check at all.
// CHECK-LABEL: llvm.func @constant_bounds
// CHECK-NOT:     scf.if
// CHECK:         scf.while
// CHECK:           affine.apply #map
// CHECK:           llvm.load
// CHECK-NOT:       scf.if
// CHECK:           scf.condition
// CHECK-NOT:       affine.apply
module {
  llvm.func @constant_bounds(%base: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %sum = scf.while (%iv = %c0) : (i32) -> i32 {
      %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %next = arith.addi %iv, %c1 : i32
      %cond = arith.cmpi ne, %next, %c10 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    llvm.return %sum : i32
  }

// Runtime bounds: the facts init <= iv <= bound - step remain, so a single
// runtime check constrains %init and %bound, and the whole loop is lifted
// speculatively: if (check) { lifted loop } else { original loop }.
// CHECK-LABEL: llvm.func @runtime_bounds
// CHECK:         arith.extsi %[[INIT:.*]] : i32 to i33
// CHECK:         arith.cmpi sge, %{{.*}} : i33
// CHECK:         arith.extsi %[[BOUND:.*]] : i32 to i33
// CHECK:         arith.cmpi sge, %{{.*}} : i33
// CHECK:         scf.if
// CHECK:           scf.while
// CHECK:             affine.apply #map
// CHECK:             llvm.load
// CHECK:         } else {
// CHECK:           scf.while
// CHECK-NOT:         affine.apply
// CHECK:             llvm.load
  llvm.func @runtime_bounds(%base: !llvm.ptr, %init: i32, %bound: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %sum = scf.while (%iv = %init) : (i32) -> i32 {
      %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %next = arith.addi %iv, %c1 : i32
      %cond = arith.cmpi ne, %next, %bound : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    llvm.return %sum : i32
  }

// More than one loop-carried variable: an accumulator is carried alongside
// the induction variable. The induction variable is still identified from
// the scf.condition, the load is lifted in place, and the accumulator, whose
// value is not structurally bounded, yields no facts.
// CHECK-LABEL: llvm.func @multiple_iter_args
// CHECK-NOT:     scf.if
// CHECK:         scf.while
// CHECK:           affine.apply #map
// CHECK:           llvm.load
// CHECK-NOT:       scf.if
// CHECK:           scf.condition
// CHECK-NOT:       affine.apply
  llvm.func @multiple_iter_args(%base: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %sum:2 = scf.while (%iv = %c0, %acc = %c0) : (i32, i32) -> (i32, i32) {
      %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %next_acc = arith.addi %acc, %v : i32
      %next_iv = arith.addi %iv, %c1 : i32
      %cond = arith.cmpi ne, %next_iv, %c10 : i32
      scf.condition(%cond) %next_iv, %next_acc : i32, i32
    } do {
    ^bb0(%iv2: i32, %acc2: i32):
      scf.yield %iv2, %acc2 : i32, i32
    }
    llvm.return %sum#0 : i32
  }

// A signed comparison instead of equality: the loop continues while
// next < bound, so the entries are bounded by init <= iv < bound. The rows
// are implied by the constant bounds, so the load is lifted in place.
// CHECK-LABEL: llvm.func @signed_condition
// CHECK-NOT:     scf.if
// CHECK:         affine.apply #map
// CHECK:         llvm.load
  llvm.func @signed_condition(%base: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %sum = scf.while (%iv = %c0) : (i32) -> i32 {
      %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %next = arith.addi %iv, %c1 : i32
      %cond = arith.cmpi slt, %next, %c10 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    llvm.return %sum : i32
  }

// The load address depends on the forwarded operand computed in the body
// rather than the argument itself; the value is still affine over the
// symbols. The address 4*(iv + 1) stays within the constant bounds, so the
// load is lifted in place.
// CHECK-LABEL: llvm.func @forwarded_operand
// CHECK-NOT:     scf.if
// CHECK:         affine.apply #map1
// CHECK:         llvm.load
  llvm.func @forwarded_operand(%base: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %sum = scf.while (%iv = %c0) : (i32) -> i32 {
      %next = arith.addi %iv, %c1 : i32
      %p = llvm.getelementptr %base[%next] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %cond = arith.cmpi ne, %next, %c10 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    llvm.return %sum : i32
  }

// A decreasing do-while: next = iv - 1 and the loop continues while
// next > 0, so the entries are bounded by bound + step < iv <= init. The
// lower row is implied by the constant bound, so only the upper row
// constrains %init.
// CHECK-LABEL: llvm.func @decreasing_loop
// CHECK:         scf.if
// CHECK:           scf.while
// CHECK:             affine.apply #map
// CHECK:             llvm.load
// CHECK:         } else {
// CHECK:           scf.while
// CHECK-NOT:         affine.apply
// CHECK:             llvm.load
  llvm.func @decreasing_loop(%base: !llvm.ptr, %init: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %sum = scf.while (%iv = %init) : (i32) -> i32 {
      %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %next = arith.subi %iv, %c1 : i32
      %cond = arith.cmpi sgt, %next, %c0 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    llvm.return %sum : i32
  }

// A non-constant step: no directional fact is derivable, but the loop
// condition still bounds the entries from above (next < bound), which
// combined with the constant init implies the no-overflow rows, so the load
// is lifted in place.
// CHECK-LABEL: llvm.func @symbolic_step
// CHECK-NOT:     scf.if
// CHECK:         affine.apply #map
// CHECK:         llvm.load
  llvm.func @symbolic_step(%base: !llvm.ptr, %step: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c10 = arith.constant 10 : i32
    %sum = scf.while (%iv = %c0) : (i32) -> i32 {
      %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, i32
      %v = llvm.load %p : !llvm.ptr -> i32
      %next = arith.addi %iv, %step : i32
      %cond = arith.cmpi slt, %next, %c10 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    llvm.return %sum : i32
  }

// A while nested in a for loop: the initial value of the induction variable
// (%i) is defined inside the outer loop, so hoisting the check before the
// outer loop fails. The nest falls back to processing the while loop
// separately, hoisting the check before it inside the for body.
// CHECK-LABEL: llvm.func @while_in_for
// CHECK:         scf.for
// CHECK-NOT:       scf.if
// CHECK:           scf.if
// CHECK:             scf.while
// CHECK:               affine.apply #map
// CHECK:               llvm.load
// CHECK:           } else {
// CHECK:             scf.while
// CHECK-NOT:           affine.apply
// CHECK:               llvm.load
  llvm.func @while_in_for(%base: !llvm.ptr, %n: i64, %m: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    scf.for %i = %c0 to %n step %c1 : i64 {
      %r = scf.while (%iv = %i) : (i64) -> i64 {
        %p = llvm.getelementptr %base[%iv] : (!llvm.ptr, i64) -> !llvm.ptr, i32
        %v = llvm.load %p : !llvm.ptr -> i32
        %next = arith.addi %iv, %c1 : i64
        %cond = arith.cmpi ne, %next, %m : i64
        scf.condition(%cond) %next : i64
      } do {
      ^bb0(%iter: i64):
        scf.yield %iter : i64
      }
    }
    llvm.return
  }

// A load after the loop: the while result is the exit value of the
// induction variable, but it is not traceable to a symbol without bounds,
// so the load is not lifted at all.
// CHECK-LABEL: llvm.func @load_after_loop
// CHECK-NOT:     scf.if
// CHECK-NOT:     affine.apply
// CHECK:         llvm.load
  llvm.func @load_after_loop(%base: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %r = scf.while (%iv = %c0) : (i32) -> i32 {
      %next = arith.addi %iv, %c1 : i32
      %cond = arith.cmpi ne, %next, %c10 : i32
      scf.condition(%cond) %next : i32
    } do {
    ^bb0(%iter: i32):
      scf.yield %iter : i32
    }
    %p = llvm.getelementptr %base[%r] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %v = llvm.load %p : !llvm.ptr -> i32
    llvm.return %v : i32
  }
}