//===- InitAllDialects.h - CIRCT Dialects Registration ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialects and
// passes to the system.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_INITALLDIALECTS_H_
#define CIRCT_INITALLDIALECTS_H_

#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Interop/InteropDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/Pipeline/PipelineDialect.h"
#include "circt/Dialect/SSP/SSPDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/SystemC/SystemCDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "mlir/IR/Dialect.h"

namespace circt {

// Add all the MLIR dialects to the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  registry.insert<
    arc::ArcDialect,
    calyx::CalyxDialect,
    chirrtl::CHIRRTLDialect,
    comb::CombDialect,
    dc::DCDialect,
    esi::ESIDialect,
    firrtl::FIRRTLDialect,
    fsm::FSMDialect,
    handshake::HandshakeDialect,
    hw::HWDialect,
    hwarith::HWArithDialect,
    interop::InteropDialect,
    ibis::IbisDialect,
    llhd::LLHDDialect,
    loopschedule::LoopScheduleDialect,
    ltl::LTLDialect,
    moore::MooreDialect,
    msft::MSFTDialect,
    om::OMDialect,
    pipeline::PipelineDialect,
    seq::SeqDialect,
    ssp::SSPDialect,
    sv::SVDialect,
    systemc::SystemCDialect,
    verif::VerifDialect
  >();
  // clang-format on
}

} // namespace circt

#endif // CIRCT_INITALLDIALECTS_H_
