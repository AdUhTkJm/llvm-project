//===- LLVMToArith.h - LLVM to Arith Pass entrypoint ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMTOARITH_LLVMTOARITH_H_
#define MLIR_CONVERSION_LLVMTOARITH_LLVMTOARITH_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTLLVMTOARITHPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert basic integer operations from the
/// LLVM dialect (add, sub, mul, sdiv, udiv, icmp) to the Arith dialect.
void populateLLVMToArithConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMTOARITH_LLVMTOARITH_H_
