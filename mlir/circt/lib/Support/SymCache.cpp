//===- SymCache.cpp - Declare Symbol Cache ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a Symbol Cache.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/SymCache.h"

using namespace mlir;
using namespace circt;

namespace circt {

/// Virtual method anchor.
SymbolCacheBase::~SymbolCacheBase() {}

void SymbolCacheBase::addDefinitions(mlir::Operation *top) {
  for (auto &region : top->getRegions())
    for (auto &block : region.getBlocks())
      for (auto symOp : block.getOps<mlir::SymbolOpInterface>())
        addSymbol(symOp);
}
} // namespace circt
