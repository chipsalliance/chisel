//===- PassRegistration.cpp - Register LLHD transformation passes ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/LLHD/Transforms/Passes.h.inc"
} // namespace

void circt::llhd::initLLHDTransformationPasses() { registerPasses(); }
