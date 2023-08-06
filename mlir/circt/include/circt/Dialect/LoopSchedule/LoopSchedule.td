//===- LoopSchedule.td - LoopSchedule Definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LOOP_SCHEDULE_DIALECT_TD
#define CIRCT_LOOP_SCHEDULE_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def LoopSchedule_Dialect : Dialect {
  let name = "loopschedule";
  let cppNamespace = "::circt::loopschedule";
  let summary = "Representation of scheduled loops";
}

include "circt/Dialect/LoopSchedule/LoopScheduleOps.td"

#endif // CIRCT_LOOP_SCHEDULE_DIALECT_TD 
