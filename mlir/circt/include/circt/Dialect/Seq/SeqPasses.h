//===- SeqPasses.h - Seq pass entry points ----------------------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_SEQ_SEQPASSES_H
#define CIRCT_DIALECT_SEQ_SEQPASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace seq {

#define GEN_PASS_DECL_LOWERSEQTOSV
#define GEN_PASS_DECL_LOWERSEQFIRRTLTOSV
#define GEN_PASS_DECL_EXTERNALIZECLOCKGATE
#include "circt/Dialect/Seq/SeqPasses.h.inc"

std::unique_ptr<mlir::Pass>
createSeqLowerToSVPass(std::optional<bool> lowerToAlwaysFF = {});
std::unique_ptr<mlir::Pass> createLowerSeqFIRRTLInitToSV();
std::unique_ptr<mlir::Pass>
createSeqFIRRTLLowerToSVPass(const LowerSeqFIRRTLToSVOptions &options = {});
std::unique_ptr<mlir::Pass> createLowerSeqHLMemPass();
std::unique_ptr<mlir::Pass>
createExternalizeClockGatePass(const ExternalizeClockGateOptions &options = {});
std::unique_ptr<mlir::Pass> createLowerFirMemPass();
std::unique_ptr<mlir::Pass> createLowerSeqFIFOPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

#endif // CIRCT_DIALECT_SEQ_SEQPASSES_H
