//===- FSMToSV.h - FSM to SV conversions ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_FSMTOSV_FSMTOSV_H
#define CIRCT_CONVERSION_FSMTOSV_FSMTOSV_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createConvertFSMToSVPass();
} // namespace circt

#endif // CIRCT_CONVERSION_FSMTOSV_FSMTOSV_H
