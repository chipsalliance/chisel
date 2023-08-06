//===- MSFTOpInterfaces.h - Microsoft OpInterfaces --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTOPINTERFACES_H
#define CIRCT_DIALECT_MSFT_MSFTOPINTERFACES_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"

namespace circt {
namespace msft {
LogicalResult verifyDynInstData(Operation *);
class InstanceHierarchyOp;
} // namespace msft
} // namespace circt
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_MSFT_MSFTOPINTERFACES_H
