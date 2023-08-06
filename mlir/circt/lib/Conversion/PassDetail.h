//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace arith {
class ArithDialect;
} // namespace arith

namespace cf {
class ControlFlowDialect;
} // namespace cf

namespace memref {
class MemRefDialect;
} // namespace memref

namespace scf {
class SCFDialect;
} // namespace scf

namespace LLVM {
class LLVMDialect;
} // namespace LLVM

namespace func {
class FuncDialect;
class FuncOp;
} // namespace func

namespace emitc {
class EmitCDialect;
} // namespace emitc
} // namespace mlir

namespace circt {

namespace arc {
class ArcDialect;
} // namespace arc

namespace dc {
class DCDialect;
} // namespace dc

namespace fsm {
class FSMDialect;
} // namespace fsm

namespace calyx {
class CalyxDialect;
class ComponentOp;
} // namespace calyx

namespace firrtl {
class FIRRTLDialect;
class CircuitOp;
class FModuleOp;
} // namespace firrtl

namespace handshake {
class HandshakeDialect;
class FuncOp;
} // namespace handshake

namespace esi {
class ESIDialect;
} // namespace esi

namespace moore {
class MooreDialect;
} // namespace moore

namespace llhd {
class LLHDDialect;
} // namespace llhd

namespace ltl {
class LTLDialect;
} // namespace ltl

namespace loopschedule {
class LoopScheduleDialect;
} // namespace loopschedule

namespace comb {
class CombDialect;
} // namespace comb

namespace hw {
class HWDialect;
class HWModuleOp;
} // namespace hw

namespace hwarith {
class HWArithDialect;
} // namespace hwarith

namespace pipeline {
class PipelineDialect;
} // namespace pipeline

namespace seq {
class SeqDialect;
} // namespace seq

namespace sv {
class SVDialect;
} // namespace sv

namespace fsm {
class FSMDialect;
} // namespace fsm

namespace systemc {
class SystemCDialect;
} // namespace systemc

namespace verif {
class VerifDialect;
} // namespace verif

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CONVERSION_PASSDETAIL_H
