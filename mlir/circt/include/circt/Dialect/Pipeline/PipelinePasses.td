//===-- PipelinePasses.td - Pipeline pass definition file --*- tablegen -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the Pipeline dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD
#define CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD

include "mlir/Pass/PassBase.td"

def ExplicitRegs : Pass<"pipeline-explicit-regs", "pipeline::ScheduledPipelineOp"> {
  let summary = "Makes stage registers explicit.";
  let description = [{
    Makes all stage-crossing def-use chains into explicit registers.
  }];
  let dependentDialects = ["hw::HWDialect"];
  let constructor = "circt::pipeline::createExplicitRegsPass()";
}

def ScheduleLinearPipeline : InterfacePass<"pipeline-schedule-linear", "circt::hw::HWModuleLike"> {
  let summary = "Schedules `pipeline.unscheduled` operations.";
  let description = [{
    Schedules `pipeline.unscheduled` operations based on operator latencies.
    This pass runs with `HWModuleLike` granularity, since it is necessary to
    erase the `pipeline.unscheduled` operation in favor of creating a
    `pipeline.scheduled` operation.
  }];
  let dependentDialects = ["hw::HWDialect"];
  let constructor = "circt::pipeline::createScheduleLinearPipelinePass()";
}


#endif // CIRCT_DIALECT_PIPELINE_PIPELINEPASSES_TD
