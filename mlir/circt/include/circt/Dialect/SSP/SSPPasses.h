//===- SSPPasses.h - SSP pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPPASSES_H
#define CIRCT_DIALECT_SSP_SSPPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace circt {
namespace ssp {

std::unique_ptr<mlir::Pass> createPrintPass();
std::unique_ptr<mlir::Pass> createRoundtripPass();
std::unique_ptr<mlir::Pass> createSchedulePass();

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/SSP/SSPPasses.h.inc"

} // namespace ssp
} // namespace circt

#endif // CIRCT_DIALECT_SSP_SSPPASSES_H
