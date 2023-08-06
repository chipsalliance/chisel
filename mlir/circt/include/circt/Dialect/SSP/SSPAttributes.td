//===- SSPAttributes.td - SSP attribute definitions --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SSP (static scheduling problem) dialect attributes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect attributes
//===----------------------------------------------------------------------===//

def DependenceAttr : AttrDef<SSPDialect, "Dependence"> {
  let summary = "Internal representation of dependence edges.";
  let description = [{
    An attribute to uniformly model def-use and auxiliary
    [dependences](https://circt.llvm.org/docs/Scheduling/#components) as well as
    to attach [properties](https://circt.llvm.org/docs/Scheduling/#properties)
    to them. This attribute is an implementation detail of the `ssp.OperationOp`
    and as such is supposed to be hidden by the custom parser/printer.
  }];

  let parameters = (ins "unsigned":$operandIdx,
                        OptionalParameter<"::mlir::FlatSymbolRefAttr">:$sourceRef,
                        OptionalParameter<"::mlir::ArrayAttr">:$properties);

  let mnemonic = "dependence";
  let assemblyFormat = [{
    `<` $operandIdx (`,` $sourceRef^)? (`:` $properties^)? `>`
  }];
}

def DependenceArrayAttr
  : TypedArrayAttrBase<DependenceAttr, "dependence array attribute">;

//===----------------------------------------------------------------------===//
// Property definitions for in-tree problems
//===----------------------------------------------------------------------===//

include "PropertyBase.td"

// Problem
def LinkedOperatorTypeProp : OperationProperty<SSPDialect,
  "LinkedOperatorType", "::mlir::SymbolRefAttr", "::circt::scheduling::Problem"> {
  let mnemonic = "opr";
  let unwrapValue = [{ getValue().getLeafReference() }];
  let wrapValue = [{ ::mlir::FlatSymbolRefAttr::get(ctx, value) }];
}
def StartTimeProp : OperationProperty<SSPDialect,
  "StartTime", "unsigned", "::circt::scheduling::Problem"> {
  let mnemonic = "t";
}
def LatencyProp : OperatorTypeProperty<SSPDialect,
  "Latency", "unsigned", "::circt::scheduling::Problem"> {
  let mnemonic = "latency";
}

// CyclicProblem
def DistanceProp : DependenceProperty<SSPDialect,
  "Distance", "unsigned", "::circt::scheduling::CyclicProblem"> {
  let mnemonic = "dist";
}
def InitiationIntervalProp : InstanceProperty<SSPDialect,
  "InitiationInterval", "unsigned", "::circt::scheduling::CyclicProblem"> {
  let mnemonic = "II";
}

// ChainingProblem
let
  unwrapValue = [{ (float) getValue().getValueAsDouble() }],
  wrapValue = [{ ::mlir::FloatAttr::get(::mlir::Float32Type::get(ctx), value) }]
in {
  def StartTimeInCycleProp : OperationProperty<SSPDialect,
    "StartTimeInCycle", "::mlir::FloatAttr", "::circt::scheduling::ChainingProblem"> {
    let mnemonic = "z";
  }
  def IncomingDelayProp : OperatorTypeProperty<SSPDialect,
    "IncomingDelay", "::mlir::FloatAttr", "::circt::scheduling::ChainingProblem"> {
    let mnemonic = "incDelay";
  }
  def OutgoingDelayProp : OperatorTypeProperty<SSPDialect,
    "OutgoingDelay", "::mlir::FloatAttr", "::circt::scheduling::ChainingProblem"> {
    let mnemonic = "outDelay";
  }
}

// SharedOperatorsProblem
def LimitProp : OperatorTypeProperty<SSPDialect,
  "Limit", "unsigned", "::circt::scheduling::SharedOperatorsProblem"> {
  let mnemonic = "limit";
}
