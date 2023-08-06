//===- LoopScheduleOps.td - LoopSchedule Op Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LoopSchedule Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCIT_LOOP_SCHEDULE_OPS_TD
#define CIRCIT_LOOP_SCHEDULE_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "circt/Dialect/LoopSchedule/LoopSchedule.td"

class LoopScheduleOp<string mnemonic, list<Trait> traits = []> :
  Op<LoopSchedule_Dialect, mnemonic, traits>;

def LoopSchedulePipelineOp : LoopScheduleOp<"pipeline", []> {
  let summary = "LoopSchedule dialect pipeline-loop.";
  let description = [{
    The `loopschedule.pipeline` operation represents a statically scheduled
    pipeline stucture that executes while a condition is true. For more details,
    see: https://llvm.discourse.group/t/rfc-representing-pipelined-loops/4171.

    A pipeline captures the result of scheduling, and is not generally safe to
    transform, besides lowering to hardware dialects. For more discussion about
    relaxing this, see: https://github.com/llvm/circt/issues/2204.

    This is the top-level operation representing a high-level pipeline. It is
    not isolated from above, but could be if this is helpful. A pipeline
    contains two regions: `condition` and `stages`.

    The pipeline may accept an optional `iter_args`, similar to the SCF dialect,
    for representing loop-carried values like induction variables or reductions.
    When the pipeline starts execution, the registers indicated as `iter_args`
    by `pipeline.terminator` should be initialized to the initial
    values specified in the `iter_args` section here. The `iter_args` relate to
    the initiation interval of the loop. The maximum distance in stages between
    where an `iter_arg` is used and where that `iter_arg` is registered must be
    less than the loop's initiation interval. For example, with II=1, each
    `iter_arg` must be used and registered in the same stage.

    The single-block `condition` region dictates the condition under which the
    pipeline should execute. It has a `register` terminator, and the
    pipeline initiates new iterations while the registered value is `true : i1`.
    It may access SSA values dominating the pipeline, as well as `iter_args`,
    which are block arguments. The body of the block may only contain
    "combinational" operations, which are currently defined to be simple
    arithmetic, comparisons, and selects from the `Standard` dialect.

    The single-block `stages` region wraps `loopschedule.pipeline.stage`
    operations. It has a `loopschedule.terminator` terminator, which can
    both return results from the pipeline and register `iter_args`. Stages may
    access SSA values dominating the pipeline, as well as `iter_args`, which are
    block arguments.
  }];

  let arguments = (ins
    I64Attr:$II,
    OptionalAttr<I64Attr>:$tripCount,
    Variadic<AnyType>:$iterArgs
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$condition,
    SizedRegion<1>:$stages
  );

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$II,
                   "std::optional<IntegerAttr>": $tripCount,
                   "mlir::ValueRange":$iterArgs)>
  ];

  let extraClassDeclaration = [{
    Block &getCondBlock() { return getCondition().front(); }
    Block &getStagesBlock() { return getStages().front(); }
  }];
}

def LoopSchedulePipelineStageOp : LoopScheduleOp<"pipeline.stage",
    [HasParent<"LoopSchedulePipelineOp">]> {
  let summary = "LoopSchedule dialect pipeline stage.";
  let description = [{
    This operation has a single-block region which dictates the operations that
    may occur concurrently.

    It has a `start` attribute, which indicates the start cycle for this stage.

    It may have an optional `when` predicate, which supports conditional
    execution for each stage. This is in addition to the `condition` region that
    controls the execution of the whole pipeline. A stage with a `when`
    predicate should only execute when the predicate is `true : i1`, and push a
    bubble through the pipeline otherwise.

    It has a `register` terminator, which passes the concurrently
    computed values forward to the next stage.

    Any stage may access `iter_args`. If a stage accesses an `iter_arg` after
    the stage in which it is defined, it is up to lowering passes to preserve
    this value until the last stage that needs it.

    Other than `iter_args`, stages may only access SSA values dominating the
    pipeline or SSA values computed by any previous stage. This ensures the
    stages capture the coarse-grained schedule of the pipeline and how values
    feed forward and backward.
  }];

  let arguments = (ins
    SI64Attr:$start,
    Optional<I1>:$when
  );

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region
    SizedRegion<1>:$body
  );

  let assemblyFormat = [{
    `start` `=` $start (`when` $when^)? $body (`:` qualified(type($results))^)? attr-dict
  }];

  let hasVerifier = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::IntegerAttr":$start)>
  ];

  let extraClassDeclaration = [{
    Block &getBodyBlock() { return getBody().front(); }
    unsigned getStageNumber();
  }];
}

def LoopScheduleRegisterOp : LoopScheduleOp<"register",
    [ParentOneOf<["LoopSchedulePipelineOp", "LoopSchedulePipelineStageOp"]>, Terminator]> {
  let summary = "LoopSchedule dialect loop register.";
  let description = [{
    The `loopschedule.register` terminates a pipeline stage and
    "registers" the values specified as operands. These values become the
    results of the stage.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = [{
    $operands (`:` qualified(type($operands))^)? attr-dict
  }];

  let hasVerifier = 1;
}

def LoopScheduleTerminatorOp : LoopScheduleOp<"terminator",
    [HasParent<"LoopSchedulePipelineOp">, Terminator, AttrSizedOperandSegments]> {
  let summary = "LoopSchedule dialect terminator.";
  let description = [{
    The `loopschedule.terminator` operation represents the terminator of
    a `loopschedule.pipeline`.

    The `results` section accepts a variadic list of values which become the
    pipeline’s return values. These must be results of a stage, and their types
    must match the pipeline's return types. The results need not be defined in
    the final stage, and it is up to lowering passes to preserve these values
    until the final stage is complete.

    The `iter_args` section accepts a variadic list of values which become the
    next iteration’s `iter_args`. These may be the results of any stage, and
    their types must match the pipeline's `iter_args` types.
  }];

  let arguments = (ins
    Variadic<AnyType>:$iter_args,
    Variadic<AnyType>:$results
  );

  let assemblyFormat = [{
    `iter_args` `(` $iter_args `)` `,`
    `results` `(` $results `)` `:`
    functional-type($iter_args, $results) attr-dict
  }];

  let hasVerifier = 1;
}

#endif // CIRCIT_LOOP_SCHEDULE_OPS_TD
