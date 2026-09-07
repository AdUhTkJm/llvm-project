// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// Three loads inside a loop: i-1, i, i+1. All three should be lifted into
// the then branch with affine.apply; the else branch keeps original GEPs.
// CHECK:       #map = affine_map<()[s0] -> ((s0 - 1) * 4)>
// CHECK:       #map1 = affine_map<()[s0] -> (s0 * 4)>
// CHECK:       #map2 = affine_map<()[s0] -> ((s0 + 1) * 4)>
// CHECK-LABEL: func.func @f
// CHECK:       scf.if
// Then: lifted loads via affine.apply
// CHECK:       scf.for
// CHECK:         affine.apply #map
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         affine.apply #map1
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         affine.apply #map2
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:       } else {
// Else: original loads via plain GEP
// CHECK:       scf.for
// CHECK-NOT:     affine.apply
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
// CHECK:         llvm.getelementptr
// CHECK:         llvm.load
module {
  func.func @f(%lb: i64, %ub: i64, %in: !llvm.ptr, %out: !llvm.ptr) {
    %one = arith.constant 1: i64
    scf.for %i = %lb to %ub step %one : i64 { 
      %c1 = arith.constant 1 : i64
      %c3 = arith.constant 3 : i32
      %i_prev = llvm.sub %i, %c1 : i64
      %ptr_prev = llvm.getelementptr %in[%i_prev] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v_prev = llvm.load %ptr_prev : !llvm.ptr -> i32

      %ptr_curr = llvm.getelementptr %in[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v_curr = llvm.load %ptr_curr : !llvm.ptr -> i32

      %i_next = llvm.add %i, %c1 : i64
      %ptr_next = llvm.getelementptr %in[%i_next] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %v_next = llvm.load %ptr_next : !llvm.ptr -> i32

      %sum1 = llvm.add %v_prev, %v_curr : i32
      %sum2 = llvm.add %sum1, %v_next : i32
      %avg = llvm.sdiv %sum2, %c3 : i32

      %ptr_out = llvm.getelementptr %out[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %avg, %ptr_out : i32, !llvm.ptr
    }
    func.return
  }
}
