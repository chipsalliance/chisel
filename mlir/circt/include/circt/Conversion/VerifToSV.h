//===- VerifToSV.h - Verif to SV pass entry point ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the VerifToSV pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_VERIFTOSV_H
#define CIRCT_CONVERSION_VERIFTOSV_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {
namespace hw {
class HWModuleOp;
} // namespace hw

/// Create the Verif to SV conversion pass.
std::unique_ptr<OperationPass<hw::HWModuleOp>> createLowerVerifToSVPass();

} // namespace circt

#endif // CIRCT_CONVERSION_VERIFTOSV_H
