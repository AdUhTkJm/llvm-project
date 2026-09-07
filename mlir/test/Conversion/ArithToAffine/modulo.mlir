// RUN: mlir-opt -convert-arith-to-affine %s | FileCheck %s

// 5-point stencil: every load is lifted in place via affine.apply. The
// user's interior/boundary scf.if is preserved; no runtime overflow check
// is synthesized because the loop bounds prove all constraints.
// CHECK:       #map = affine_map<()[s0] -> ((s0 - 1024) * 4)>
// CHECK:       #map1 = affine_map<()[s0] -> ((s0 + 1024) * 4)>
// CHECK:       #map2 = affine_map<()[s0] -> ((s0 - 1) * 4)>
// CHECK:       #map3 = affine_map<()[s0] -> ((s0 + 1) * 4)>
// CHECK:       #map4 = affine_map<()[s0] -> (s0 * 4)>
// CHECK-LABEL: func.func @apply_5point_stencil
// CHECK:       scf.for
// CHECK:       scf.if
// CHECK:         affine.apply #map
// CHECK-NOT:     i65
// CHECK:         llvm.load
// CHECK:         affine.apply #map1
// CHECK-NOT:     i65
// CHECK:         llvm.load
// CHECK:         affine.apply #map2
// CHECK-NOT:     i65
// CHECK:         llvm.load
// CHECK:         affine.apply #map3
// CHECK-NOT:     i65
// CHECK:         llvm.load
// CHECK:       } else {
// CHECK:         affine.apply #map4
// CHECK-NOT:     i65
// CHECK:         llvm.load
module {
  func.func @apply_5point_stencil(
    %in_ptr: !llvm.ptr, 
    %out_ptr: !llvm.ptr
  ) {
    // Constants
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c1023 = arith.constant 1023 : i64
    %c1024 = arith.constant 1024 : i64
    %c1048576 = arith.constant 1048576 : i64
    %cst_025 = arith.constant 0.250000e+00 : f32

    // 1D Loop matching C++ ij iteration
    scf.for %ij = %c0 to %c1048576 step %c1 : i64 {
      // Recover 2D indices (i = ij / 1024, j = ij % 1024)
      %i = arith.divui %ij, %c1024 : i64
      %j = arith.remui %ij, %c1024 : i64

      // Boundary condition checks
      %i_gt_0 = arith.cmpi ugt, %i, %c0 : i64
      %i_lt_1023 = arith.cmpi ult, %i, %c1023 : i64
      %j_gt_0 = arith.cmpi ugt, %j, %c0 : i64
      %j_lt_1023 = arith.cmpi ult, %j, %c1023 : i64

      %cond_i = arith.andi %i_gt_0, %i_lt_1023 : i1
      %cond_j = arith.andi %j_gt_0, %j_lt_1023 : i1
      %is_interior = arith.andi %cond_i, %cond_j : i1

      scf.if %is_interior {
        // Offset indices
        %idx_top    = arith.subi %ij, %c1024 : i64
        %idx_bottom = arith.addi %ij, %c1024 : i64
        %idx_left   = arith.subi %ij, %c1 : i64
        %idx_right  = arith.addi %ij, %c1 : i64

        // Pointer arithmetic via llvm.getelementptr (GEP)
        %ptr_top    = llvm.getelementptr %in_ptr[%idx_top] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %ptr_bottom = llvm.getelementptr %in_ptr[%idx_bottom] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %ptr_left   = llvm.getelementptr %in_ptr[%idx_left] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %ptr_right  = llvm.getelementptr %in_ptr[%idx_right] : (!llvm.ptr, i64) -> !llvm.ptr, f32

        // Explicit pointer loads
        %val_top    = llvm.load %ptr_top : !llvm.ptr -> f32
        %val_bottom = llvm.load %ptr_bottom : !llvm.ptr -> f32
        %val_left   = llvm.load %ptr_left : !llvm.ptr -> f32
        %val_right  = llvm.load %ptr_right : !llvm.ptr -> f32

        // Accumulate floating point arithmetic
        %sum_tb = arith.addf %val_top, %val_bottom : f32
        %sum_tbl = arith.addf %sum_tb, %val_left : f32
        %sum_all = arith.addf %sum_tbl, %val_right : f32
        %res = arith.mulf %sum_all, %cst_025 : f32

        // Compute destination pointer and store result
        %ptr_out = llvm.getelementptr %out_ptr[%ij] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        llvm.store %res, %ptr_out : f32, !llvm.ptr
      } else {
        // Boundary copy using GEP + Load + Store
        %ptr_in_curr  = llvm.getelementptr %in_ptr[%ij] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        %ptr_out_curr = llvm.getelementptr %out_ptr[%ij] : (!llvm.ptr, i64) -> !llvm.ptr, f32
        
        %val = llvm.load %ptr_in_curr : !llvm.ptr -> f32
        llvm.store %val, %ptr_out_curr : f32, !llvm.ptr
      }
    }
    return
  }
}