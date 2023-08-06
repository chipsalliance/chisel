//===- PipelineOps.td - Pipeline dialect operations --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIPELINE_PIPELINEOPS_TD
#define CIRCT_DIALECT_PIPELINE_PIPELINEOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

include "circt/Dialect/Pipeline/PipelineDialect.td"


class PipelineBase<string mnemonic, list<Trait> traits = []> : 
  Op<Pipeline_Dialect, mnemonic, !listconcat(traits, [
      IsolatedFromAbove,
      Pure,
      RegionKindInterface,
      AttrSizedOperandSegments])> {

  let arguments = (ins
    OptionalAttr<StrAttr>:$name, Variadic<AnyType>:$inputs, Variadic<AnyType>:$extInputs, Optional<I1>:$stall,
    I1:$clock, I1:$reset, I1:$go, StrArrayAttr:$inputNames,
    OptionalAttr<StrArrayAttr>:$extInputNames, StrArrayAttr:$outputNames
  );
  let results = (outs Variadic<AnyType>:$dataOutputs, I1:$done);
  let hasCustomAssemblyFormat = 1;

  /// Additional class declarations inside the pipeline op.
  code extraModuleClassDeclaration = ?;

  // The entry block of the pipeline contains a lot of block arguments - the
  // indices are kept away from the user and only "hardcoded" here, as well as
  // in the printer/parser.
  // Order:
  // 1. Inputs
  // 2. External inputs (opt)
  // 3. Stall (opt)
  // 4. Clock
  // 5. Reset
  // 6. Go
  let extraClassDeclaration = extraModuleClassDeclaration # [{
    // Returns the entry stage of this pipeline.
    Block* getEntryStage() {
      Region* region = &getOperation()->getRegion(0);
      return &region->front();
    }

    // Returns the inner clock signal argument of this pipeline.
    BlockArgument getInnerClock() {
      Block* entryStage = getEntryStage();
      return entryStage->getArgument(entryStage->getNumArguments() - 3);
    }

    // Returns the inner reset signal argument of this pipeline.
    BlockArgument getInnerReset() {
      Block* entryStage = getEntryStage();
      return getEntryStage()->getArgument(entryStage->getNumArguments() - 2);
    }

    // Returns the inner go signal argument of this pipeline.
    BlockArgument getInnerGo() {
      Block* entryStage = getEntryStage();
      return getEntryStage()->getArgument(entryStage->getNumArguments() - 1);
    }

    // Returns true if this pipeline has a stall signal.
    bool hasStall() {
      return static_cast<bool>(getStall());
    }

    // Returns the stall signal argument of this pipeline, if it exists.
    BlockArgument getInnerStall() {
      assert(hasStall() && "pipeline has no stall signal");
      Block* entryStage = getEntryStage();
      return getEntryStage()->getArgument(entryStage->getNumArguments() - 4);
    }

    llvm::ArrayRef<BlockArgument> getInnerInputs() {
      return getEntryStage()->getArguments().take_front(getInputs().size());
    }

    llvm::ArrayRef<BlockArgument> getInnerExtInputs() {
      return getEntryStage()->getArguments().slice(
          getInputs().size(), getExtInputs().size()
      );
    }
  }];
}

def UnscheduledPipelineOp : PipelineBase<"unscheduled", [
    RegionKindInterface,
    HasOnlyGraphRegion,
    SingleBlockImplicitTerminator<"ReturnOp">
  ]> {
  
  let summary = "unscheduled pipeline operation";
  let description = [{
    The "pipeline.unscheduled" operation represents a pipeline that has not yet
    been scheduled. It contains a single block representing a graph region of
    operations to-be-scheduled into a pipeline.
    Mainly serves as a container and entrypoint for scheduling.

    The interface of a `pipeline.unscheduled` is similar to that of a
    `pipeline.scheduled`. Please refer to this op for further documentation
    about the interface signals.
  }];

  let regions = (region SizedRegion<1>: $body);
  let extraModuleClassDeclaration = "";
}

def ScheduledPipelineOp : PipelineBase<"scheduled", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Scheduled pipeline operation";
  let description = [{
    The `pipeline.scheduled` operation represents a scheduled pipeline.
    The pipeline contains a single block representing a graph region.

    A `pipeline.scheduled` operation can exist in multiple phases, mainly
    pertaining to when registers have been materialized (made explicit).
    For an in-depth explanation, please refer to the Pipeline dialect rationale.

    A `pipeline.scheduled` supports a `stall` input. This signal is intended to
    connect to all stages within the pipeline, and is used to stall the entirety
    of the pipeline. It is lowering defined how stages choose to use this signal,
    although in the common case, a `stall` signal would typically connect to
    the clock-enable input of the stage-separating registers.

    The `go` input is used to start the pipeline. This value is fed through
    the stages as the current stage enable/next stage valid signal.
    Note: the op is currently only designed for pipelines with II=1. For
    pipelines with II>1, a user must themselves maintain state about when
    the pipeline is ready to accept new inputs. We plan to add support for
    head-of-pipeline backpressure in the future.
  }];

  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "TypeRange":$dataOutputs, "ValueRange":$inputs, "ValueRange":$extInputs,
      "ArrayAttr":$inputNames, "ArrayAttr":$outputNames, "ArrayAttr":$extInputNames,
      "Value":$clock, "Value":$reset, "Value":$go, CArg<"Value", "{}">:$stall, CArg<"StringAttr", "{}">:$name)>
  ];


  code extraModuleClassDeclaration = [{
    static mlir::RegionKind getRegionKind(unsigned index) {
      return mlir::RegionKind::SSACFG;
    }

    // Returns all of the stages in this pipeline.
    llvm::iplist<mlir::Block>& getStages() {
      return getRegion().getBlocks();
    }

    void getAsmBlockArgumentNames(mlir::Region &region,
                        mlir::OpAsmSetValueNameFn setNameFn);

    // Returns all of the stages in this pipeline. The stages are ordered
    // with respect to their position in the pipeline as determined by the
    // control flow of `pipeline.stage` operations.
    // Stages are ordered from first (entry) to last (exit).
    llvm::SmallVector<Block*> getOrderedStages();

    // Returns a map of stages to their index in the pipeline - this is
    // with respect to 'getOrderedStages'.
    llvm::DenseMap<Block*, unsigned> getStageMap();

    // Gets the n'th stage of this pipeline
    Block* getStage(unsigned n) {
      auto& blocks = getRegion().getBlocks();
      assert(n < blocks.size() && "Stage index out of bounds");
      return &*std::next(blocks.begin(), n);
    }

    // Returns the last stage in the pipeline.
    Block* getLastStage();

    // Adds a new stage to this pipeline. It is the users responsibility to
    // modify other stages to point to this new stage.
    Block* addStage();

    // Returns true if the pipeline has entered the register materialized phase.
    // This enables certain invariants such as "all values used within a stage
    // must be defined within a stage".
    bool isMaterialized();

    // Returns the data arguments for a stage. The stage enable signal is _not_ part
    // of the returned values.
    llvm::ArrayRef<BlockArgument> getStageDataArgs(Block* stage) {
      if(stage == getEntryStage())
        return getInnerInputs();

      // Data arguments for all non-entry stages are all block arguments
      // except for the last block arguments (which is the stage enable signal).
      return stage->getArguments().drop_back();
    }

    // Returns the enable signal of the given pipeline stage. This is always
    // the last block argument of a stage for anything but the entry stage.
    Value getStageEnableSignal(size_t stageIdx) {
      return getStageEnableSignal(getStage(stageIdx));
    }

    // Returns the enable signal for the given stage. The enable signal is always
    // the last signal in the stage argument list.
    Value getStageEnableSignal(Block* stage) {
      if(stage != getEntryStage())
        return stage->getArguments().back();
      return getInnerGo();
    }
  }];
}


def StageOp : Op<Pipeline_Dialect, "stage", [
    AttrSizedOperandSegments,
    HasParent<"ScheduledPipelineOp">,
    Pure,
    Terminator
  ]> {
  let summary = "Pipeline stage terminator.";
  let description = [{
    The `pipeline.stage` operation represents a stage terminator. It is used
    to communicate:
    1. which stage (block) to transition to next
    2. which registers to build at this stage boundary
    3. which values to pass through to the next stage without registering
    4. An optional hierarchy of boolean values to be used for clock gates for
       each register.
      - The implicit '!stalled' gate will always be the first signal in the
        hierarchy. Further signals are added to the hierarchy from left to
        right.


    Example:
    ```mlir
    pipeline.stage ^bb1 regs(%a : i32 gated by [%foo, %bar], %b : i1) pass(%c : i32)
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$registers,
    Variadic<AnyType>:$passthroughs,
    Variadic<I1>:$clockGates,
    I64ArrayAttr:$clockGatesPerRegister,
    OptionalAttr<StrArrayAttr>:$registerNames,
    OptionalAttr<StrArrayAttr>:$passthroughNames);
  let successors = (successor AnySuccessor:$nextStage);
  let results = (outs);
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    $nextStage
    custom<StageRegisters>($registers, type($registers), $clockGates, $clockGatesPerRegister, $registerNames)
    custom<Passthroughs>($passthroughs, type($passthroughs), $passthroughNames)  
    attr-dict
  }];

  let extraClassDeclaration = [{
    // Set the destination stage.
    void setNextStage(Block *block) {
      setSuccessor(block);
    }

    // Returns the list of clock gates for the given register.
    ValueRange getClockGatesForReg(unsigned regIdx);

    // Returns the register name for a given register index.
    StringAttr getRegisterName(unsigned regIdx) {
      if(auto names = getOperation()->getAttrOfType<ArrayAttr>("registerNames")) {
        auto name = names[regIdx].cast<StringAttr>();
        if(!name.strref().empty())
          return name;
      }

      return {};
    }

    // Returns the passthrough name for a given passthrough index.
    StringAttr getPassthroughName(unsigned passthroughIdx) {
      if(auto names = getOperation()->getAttrOfType<ArrayAttr>("passthroughNames")) {
        auto name = names[passthroughIdx].cast<StringAttr>();
        if(!name.strref().empty())
          return name;
      }

      return {};
    }


  }];

  let builders = [
    OpBuilder<(ins "Block*":$dest, "ValueRange":$registers, "ValueRange":$passthroughs)>,
    // Clock gate builder, which takes a mapping between the registers and
    // and their clock gate hierarchy.
    OpBuilder<(ins "Block*":$dest, "ValueRange":$registers, "ValueRange":$passthroughs,
      "const llvm::DenseMap<Value, llvm::SmallVector<Value>>&":$clockGates,
      CArg<"mlir::ArrayAttr", "{}">:$registerNames, CArg<"mlir::ArrayAttr", "{}">:$passthroughNames)>
  ];
}

def ReturnOp : Op<Pipeline_Dialect, "return", [
    Terminator,
    ParentOneOf<["UnscheduledPipelineOp", "ScheduledPipelineOp"]>
  ]> {
  let summary = "Pipeline dialect return.";
  let description = [{
    The "return" operation represents a terminator of a `pipeline.pipeline`.
  }];

  let hasVerifier = 1;
  let arguments = (ins Variadic<AnyType>:$inputs);
  let builders = [OpBuilder<(ins), [{ return; }]>];
  let assemblyFormat = [{ ($inputs^)? attr-dict (`:` type($inputs)^)? }];
}

def LatencyOp : Op<Pipeline_Dialect, "latency", [
    SingleBlockImplicitTerminator<"LatencyReturnOp">,
    RegionKindInterface,
    HasOnlyGraphRegion,
    ParentOneOf<["UnscheduledPipelineOp", "ScheduledPipelineOp"]>
  ]> {
  
  let summary = "Pipeline dialect latency operation.";
  let description = [{
    The `pipeline.latency` operation represents an operation for wrapping
    multi-cycle operations. The operation declares a single block
    wherein any operation may be placed within. The operation is not
    `IsolatedFromAbove` meaning that the operation can reference values
    defined outside of the operation (subject to the materialization
    phase of the parent pipeline).
  }];

  let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<1>]>:$latency);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "unsigned":$latency)>
  ];
  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    attr-dict $latency `->` `(` type($results) `)` $body
  }];

  let extraClassDeclaration = [{
    // Returns the body block of the latency operation.
    Block* getBodyBlock() {
      return &getBody().front();
    }
  }];
}

def LatencyReturnOp : Op<Pipeline_Dialect, "latency.return", [
    Terminator,
    HasParent<"LatencyOp">
  ]> {
  let summary = "Pipeline latency return operation.";
  let description = [{
    The `pipeline.latency.return` operation represents a terminator of a
    `pipeline.latency` operation.
  }];

  let hasVerifier = 1;
  let arguments = (ins Variadic<AnyType>:$inputs);
  let builders = [OpBuilder<(ins), [{ return; }]>];
  let assemblyFormat = [{ ($inputs^)? attr-dict (`:` type($inputs)^)? }];
}


#endif // CIRCT_DIALECT_PIPELINE_PIPELINEOPS_TD
