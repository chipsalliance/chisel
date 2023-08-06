//===- MooreToCore.h - Moore to Core pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the MooreToCore pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_MOORETOCORE_H
#define CIRCT_CONVERSION_MOORETOCORE_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

/// Create an Moore to Comb/HW/LLHD conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMooreToCorePass();

} // namespace circt

#endif // CIRCT_CONVERSION_MOORETOCORE_H
