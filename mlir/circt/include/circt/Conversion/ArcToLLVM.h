//===- ArcToLLVM.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_ARCTOLLVM_H
#define CIRCT_CONVERSION_ARCTOLLVM_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
std::unique_ptr<OperationPass<ModuleOp>> createLowerArcToLLVMPass();
} // namespace circt

#endif // CIRCT_CONVERSION_ARCTOLLVM_H
