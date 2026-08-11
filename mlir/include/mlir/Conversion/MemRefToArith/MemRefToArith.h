//===- MemRefToArith.h - MemRef to Arith Pass entrypoint ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTMEMREFTOARITH_CONVERTMEMREFTOARITH_H_
#define MLIR_CONVERSION_CONVERTMEMREFTOARITH_CONVERTMEMREFTOARITH_H_

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;

#define GEN_PASS_DECL_CONVERTMEMREFTOARITH
#include "mlir/Conversion/Passes.h.inc"

void populateMemRefToArithConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTMEMREFTOARITH_CONVERTMEMREFTOARITH_H_
