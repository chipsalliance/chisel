//===- ValueMapper.h - Support for mapping SSA values -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides support for mapping SSA values between two domains.
// Provided a BackedgeBuilder, the ValueMapper supports mappings between
// GraphRegions, creating Backedges in cases of 'get'ing mapped values which are
// yet to be 'set'.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_VALUEMAPPER_H
#define CIRCT_SUPPORT_VALUEMAPPER_H

#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <functional>
#include <variant>

namespace circt {

/// The ValueMapper class facilitates the definition and connection of SSA
/// def-use chains between two location - a 'from' location (defining
/// use-def chains) and a 'to' location (where new operations are created based
/// on the 'from' location).´
class ValueMapper {
public:
  using TypeTransformer = llvm::function_ref<mlir::Type(mlir::Type)>;
  static mlir::Type identity(mlir::Type t) { return t; };
  explicit ValueMapper(BackedgeBuilder *bb = nullptr) : bb(bb) {}

  // Get the mapped value of value 'from'. If no mapping has been registered, a
  // new backedge is created. The type of the mapped value may optionally be
  // modified through the 'typeTransformer'.
  mlir::Value get(mlir::Value from,
                  TypeTransformer typeTransformer = ValueMapper::identity);
  llvm::SmallVector<mlir::Value>
  get(mlir::ValueRange from,
      TypeTransformer typeTransformer = ValueMapper::identity);

  // Set the mapped value of 'from' to 'to'. If 'from' is already mapped to a
  // backedge, replaces that backedge with 'to'. If 'replace' is not set, and a
  // (non-backedge) mapping already exists, an assert is thrown.
  void set(mlir::Value from, mlir::Value to, bool replace = false);
  void set(mlir::ValueRange from, mlir::ValueRange to, bool replace = false);

private:
  BackedgeBuilder *bb = nullptr;
  llvm::DenseMap<mlir::Value, std::variant<mlir::Value, Backedge>> mapping;
};

} // namespace circt

#endif // CIRCT_SUPPORT_VALUEMAPPER_H
