// RUN: mlir-opt --convert-arith-to-affine %s | FileCheck %s

// A load whose address is a raw pointer (not an integer/index computation)
// cannot be lifted: the pointer must not become a symbol of the affine
// expression. Previously this crashed in getSignedBounds via the index-cast
// that tidyResult would create for the pointer-typed symbol.
module {
  // CHECK-LABEL: llvm.func @load_of_raw_pointer
  // CHECK:         llvm.load %arg0 : !llvm.ptr -> f64
  llvm.func @load_of_raw_pointer(%arg0: !llvm.ptr) {
    %0 = llvm.load %arg0 : !llvm.ptr -> f64
    llvm.return
  }
}
