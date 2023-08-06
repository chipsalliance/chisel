//===- CalyxToHW.h - Calyx to HW conversion pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Calyx dialect to the
// HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_CALYXTOHW_CALYXTOHW_H
#define CIRCT_CONVERSION_CALYXTOHW_CALYXTOHW_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace calyx {

/// Returns a hw.module.extern operation describing the Verilog module which a
/// ComponentOp eventually results in.
hw::HWModuleExternOp getExternHWModule(OpBuilder &builder, ComponentOp op);

} // namespace calyx

std::unique_ptr<mlir::Pass> createCalyxToHWPass();

} // namespace circt

#endif // CIRCT_CONVERSION_CALYXTOHW_CALYXTOHW_H
