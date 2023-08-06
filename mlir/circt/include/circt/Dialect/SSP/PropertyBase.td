//===- PropertyBase.td - Base classes for SSP properties ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines base classes for attributes representing properties in the
// SSP (static scheduling problem) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PROPERTYBASE_TD
#define PROPERTYBASE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// Base class for MLIR attributes representing properties (in the scheduling
// infra's sense) on the SSP dialect's ops. The general idea here is that all
// property attributes inherit from one of the subclasses below, so that their
// implementations gain `setInProblem` and `getFromProblem` methods which then
// directly call the corresponding getter/setters on the concrete problem
// class.
class PropertyBase<Dialect dialect,
                   string name, string type,
                   string problem,
                   list<Trait> traits = []>
  : AttrDef<dialect, name, traits> {
  
  string propertyName = name;
  string propertyType = type;
  string problemClassName = problem;

  // Unwrap the data in this attribute's `$value` parameter in order to pass it 
  // to the corresponding setter on the problem class.
  code unwrapValue = [{ getValue() }];
  // Wrap the `value` returned by the getter on the problem class in order to
  // store it in this attribute's `$value` parameter.
  code wrapValue = [{ value }];

  let summary = "Models the `" # propertyName # "` property in `" # problemClassName # "`.";

  let parameters = (ins propertyType:$value);
  let mnemonic = propertyName;
  let assemblyFormat = [{ `<` $value `>` }];
}

class OperationProperty<Dialect dialect, string name, string type, string problem,
                        list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob, ::mlir::Operation *op) {
      prob.set}] # propertyName # [{(op, }] # unwrapValue # [{);
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob,
                                            ::mlir::Operation *op, ::mlir::MLIRContext *ctx) {
      if (auto optValue = prob.get}] # propertyName # [{(op)) {
        auto value = *optValue;
        return }] # cppClassName # [{::get(ctx, }] # wrapValue # [{);
      }
      return {};
    }
  }];
}

class OperatorTypeProperty<Dialect dialect, string name, string type, string problem,
                           list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob,
                      ::circt::scheduling::Problem::OperatorType opr) {
      prob.set}] # propertyName # [{(opr, }] # unwrapValue # [{);
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob,
                                            ::circt::scheduling::Problem::OperatorType opr,
                                            ::mlir::MLIRContext *ctx) {
      if (auto optValue = prob.get}] # propertyName # [{(opr)) {
        auto value = *optValue;
        return }] # cppClassName # [{::get(ctx, }] # wrapValue # [{);
      }
      return {};
    }
  }];
}

class DependenceProperty<Dialect dialect, string name, string type, string problem,
                         list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob,
                      ::circt::scheduling::Problem::Dependence dep) {
      prob.set}] # propertyName # [{(dep, }] # unwrapValue # [{);
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob,
                                            ::circt::scheduling::Problem::Dependence dep,
                                            ::mlir::MLIRContext *ctx) {
      if (auto optValue = prob.get}] # propertyName # [{(dep)) {
        auto value = *optValue;
        return }] # cppClassName # [{::get(ctx, }] # wrapValue # [{);
      }
      return {};
    }
  }];
}

class InstanceProperty<Dialect dialect, string name, string type, string problem,
                       list<Trait> traits = []>
    : PropertyBase<dialect, name, type, problem, traits> {
  let extraClassDeclaration = [{
    void setInProblem(}] # problemClassName # [{ &prob) {
      prob.set}] # propertyName # [{(}] # unwrapValue # [{);
    }
    static ::mlir::Attribute getFromProblem(}] # problemClassName # [{ &prob,
                                            ::mlir::MLIRContext *ctx) {
      if (auto optValue = prob.get}] # propertyName # [{()) {
        auto value = *optValue;
        return }] # cppClassName # [{::get(ctx, }] # wrapValue # [{);
      }
      return {};
    }
  }];
}

#endif // PROPERTYBASE_TD
