//===- MSFTOpInterfaces.cpp - Implement MSFT OpInterfaces -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

using namespace circt;
using namespace msft;

LogicalResult circt::msft::verifyDynInstData(Operation *op) {
  auto inst = dyn_cast<DynamicInstanceOp>(op->getParentOp());
  FlatSymbolRefAttr globalRef =
      cast<DynInstDataOpInterface>(op).getGlobalRefSym();

  if (inst && globalRef)
    return op->emitOpError("cannot both have a global ref symbol and be a "
                           "child of a dynamic instance op");
  if (!inst && !globalRef)
    return op->emitOpError("must have either a global ref symbol of belong to "
                           "a dynamic instance op");
  return success();
}

namespace circt {
namespace msft {
#include "circt/Dialect/MSFT/MSFTOpInterfaces.cpp.inc"
} // namespace msft
} // namespace circt
