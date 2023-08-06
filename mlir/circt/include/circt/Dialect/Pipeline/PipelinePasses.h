//===- PipelinePasses.h - Pipeline pass entry points ------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_H
#define CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace circt {
namespace pipeline {

std::unique_ptr<mlir::Pass> createExplicitRegsPass();
std::unique_ptr<mlir::Pass> createScheduleLinearPipelinePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Pipeline/PipelinePasses.h.inc"

} // namespace pipeline
} // namespace circt

#endif // CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_H
