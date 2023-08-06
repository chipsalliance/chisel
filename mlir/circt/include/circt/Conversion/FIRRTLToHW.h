//===- FIRRTLToHW.h - FIRRTL to HW conversion pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the FIRRTL dialect to
// HW and SV dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FIRRTLTOHW_FIRRTLTOHW_H
#define CIRCT_CONVERSION_FIRRTLTOHW_FIRRTLTOHW_H

#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

std::unique_ptr<mlir::Pass> createLowerFIRRTLToHWPass(
    bool enableAnnotationWarning = false, bool emitChiselAssertsAsSVA = false,
    bool disableMemRandomization = false, bool disableRegRandomization = false);

} // namespace circt

#endif // CIRCT_CONVERSION_FIRRTLTOHW_FIRRTLTOHW_H
