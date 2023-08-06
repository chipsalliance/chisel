//===- Evaluator.h - Object Model dialect evaluator -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model dialect declaration.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
#define CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H

#include "circt/Dialect/OM/OMOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace circt {
namespace om {

/// A value of an object in memory. It is either a composite Object, or a
/// primitive Attribute. Further refinement is expected.
using ObjectValue = std::variant<std::shared_ptr<struct Object>, Attribute>;

/// The fields of a composite Object, currently represented as a map. Further
/// refinement is expected.
using ObjectFields = SmallDenseMap<StringAttr, ObjectValue>;

/// An Evaluator, which is constructed with an IR module and can instantiate
/// Objects. Further refinement is expected.
struct Evaluator {
  /// Construct an Evaluator with an IR module.
  Evaluator(ModuleOp mod);

  /// Instantiate an Object with its class name and actual parameters.
  FailureOr<std::shared_ptr<Object>>
  instantiate(StringAttr className, ArrayRef<ObjectValue> actualParams);

  /// Get the Module this Evaluator is built from.
  mlir::ModuleOp getModule();

private:
  /// Evaluate a Value in a Class body according to the small expression grammar
  /// described in the rationale document. The actual parameters are the values
  /// supplied at the current instantiation of the Class being evaluated.
  FailureOr<ObjectValue> evaluateValue(Value value,
                                       ArrayRef<ObjectValue> actualParams);

  /// Evaluator dispatch functions for the small expression grammar.
  FailureOr<ObjectValue> evaluateParameter(BlockArgument formalParam,
                                           ArrayRef<ObjectValue> actualParams);
  FailureOr<ObjectValue> evaluateConstant(ConstantOp op,
                                          ArrayRef<ObjectValue> actualParams);
  FailureOr<ObjectValue>
  evaluateObjectInstance(ObjectOp op, ArrayRef<ObjectValue> actualParams);
  FailureOr<ObjectValue>
  evaluateObjectField(ObjectFieldOp op, ArrayRef<ObjectValue> actualParams);

  /// The symbol table for the IR module the Evaluator was constructed with.
  /// Used to look up class definitions.
  SymbolTable symbolTable;

  /// Object storage. Currently used for memoizing calls to evaluateValue.
  /// Further refinement is expected.
  mlir::DenseMap<Value, std::shared_ptr<Object>> objects;
};

/// A composite Object, which has a type and fields.
/// Enables the shared_from_this functionality so Object pointers can be passed
/// through the CAPI and unwrapped back into C++ smart pointers with the
/// appropriate reference count.
struct Object : std::enable_shared_from_this<Object> {
  /// Get the type of the Object.
  mlir::Type getType();

  /// Get a field of the Object by name.
  FailureOr<ObjectValue> getField(StringAttr name);

  /// Get all the field names of the Object.
  ArrayAttr getFieldNames();

private:
  /// Allow the instantiate method as a friend to construct Objects.
  friend FailureOr<std::shared_ptr<Object>>
      Evaluator::instantiate(StringAttr, ArrayRef<ObjectValue>);

  /// Construct an Object of the given Class with the given fields.
  Object(ClassOp cls, ObjectFields &fields);

  /// The Class of the Object.
  ClassOp cls;

  /// The fields of the Object.
  ObjectFields fields;
};

/// Helper to enable printing objects in Diagnostics.
static inline mlir::Diagnostic &operator<<(mlir::Diagnostic &diag,
                                           const ObjectValue &objectValue) {
  if (auto *object = std::get_if<std::shared_ptr<Object>>(&objectValue))
    diag << *object;
  if (auto *attribute = std::get_if<Attribute>(&objectValue))
    diag << *attribute;
  return diag;
}

} // namespace om
} // namespace circt

#endif // CIRCT_DIALECT_OM_EVALUATOR_EVALUATOR_H
