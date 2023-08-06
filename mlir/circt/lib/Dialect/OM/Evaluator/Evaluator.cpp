//===- Evaluator.cpp - Object Model dialect evaluator ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Object Model dialect Evaluator.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt::om;

/// Construct an Evaluator with an IR module.
circt::om::Evaluator::Evaluator(ModuleOp mod) : symbolTable(mod) {}

/// Get the Module this Evaluator is built from.
ModuleOp circt::om::Evaluator::getModule() {
  return cast<ModuleOp>(symbolTable.getOp());
}

/// Instantiate an Object with its class name and actual parameters.
FailureOr<std::shared_ptr<Object>>
circt::om::Evaluator::instantiate(StringAttr className,
                                  ArrayRef<ObjectValue> actualParams) {
  ClassOp cls = symbolTable.lookup<ClassOp>(className);
  if (!cls)
    return symbolTable.getOp()->emitError("unknown class name ") << className;

  auto formalParamNames = cls.getFormalParamNames().getAsRange<StringAttr>();
  auto formalParamTypes = cls.getBodyBlock()->getArgumentTypes();

  // Verify the actual parameters are the right size and types for this class.
  if (actualParams.size() != formalParamTypes.size()) {
    auto error = cls.emitError("actual parameter list length (")
                 << actualParams.size() << ") does not match formal "
                 << "parameter list length (" << formalParamTypes.size() << ")";
    error.attachNote() << "actual parameters: " << actualParams;
    error.attachNote(cls.getLoc()) << "formal parameters: " << formalParamTypes;
    return error;
  }

  // Verify the actual parameter types match.
  for (auto [actualParam, formalParamName, formalParamType] :
       llvm::zip(actualParams, formalParamNames, formalParamTypes)) {
    Type actualParamType;
    if (auto *attr = std::get_if<Attribute>(&actualParam))
      if (auto typedActualParam = attr->dyn_cast_or_null<TypedAttr>())
        actualParamType = typedActualParam.getType();
    if (auto *object = std::get_if<std::shared_ptr<Object>>(&actualParam))
      actualParamType = object->get()->getType();

    if (!actualParamType)
      return cls.emitError("actual parameter for ")
             << formalParamName << " is null";

    if (actualParamType != formalParamType) {
      auto error = cls.emitError("actual parameter for ")
                   << formalParamName << " has invalid type";
      error.attachNote() << "actual parameter: " << actualParam;
      error.attachNote() << "format parameter type: " << formalParamType;
      return error;
    }
  }

  // Instantiate the fields.
  ObjectFields fields;
  for (auto field : cls.getOps<ClassFieldOp>()) {
    StringAttr name = field.getSymNameAttr();
    Value value = field.getValue();

    FailureOr<ObjectValue> result = evaluateValue(value, actualParams);
    if (failed(result))
      return failure();

    fields[name] = result.value();
  }

  // Allocate the Object. Further refinement is expected.
  auto *object = new Object(cls, fields);

  return success(std::shared_ptr<Object>(object));
}

/// Evaluate a Value in a Class body according to the semantics of the IR. The
/// actual parameters are the values supplied at the current instantiation of
/// the Class being evaluated.
FailureOr<ObjectValue>
circt::om::Evaluator::evaluateValue(Value value,
                                    ArrayRef<ObjectValue> actualParams) {
  return TypeSwitch<Value, FailureOr<ObjectValue>>(value)
      .Case([&](BlockArgument arg) {
        return evaluateParameter(arg, actualParams);
      })
      .Case([&](OpResult result) {
        return TypeSwitch<Operation *, FailureOr<ObjectValue>>(
                   result.getDefiningOp())
            .Case([&](ConstantOp op) {
              return evaluateConstant(op, actualParams);
            })
            .Case([&](ObjectOp op) {
              return evaluateObjectInstance(op, actualParams);
            })
            .Case([&](ObjectFieldOp op) {
              return evaluateObjectField(op, actualParams);
            })
            .Default([&](Operation *op) {
              auto error = op->emitError("unable to evaluate value");
              error.attachNote() << "value: " << value;
              return error;
            });
      });
}

/// Evaluator dispatch function for parameters.
FailureOr<ObjectValue>
circt::om::Evaluator::evaluateParameter(BlockArgument formalParam,
                                        ArrayRef<ObjectValue> actualParams) {
  return success(actualParams[formalParam.getArgNumber()]);
}

/// Evaluator dispatch function for constants.
FailureOr<ObjectValue>
circt::om::Evaluator::evaluateConstant(ConstantOp op,
                                       ArrayRef<ObjectValue> actualParams) {
  return success(op.getValue());
}

/// Evaluator dispatch function for Object instances.
FailureOr<ObjectValue> circt::om::Evaluator::evaluateObjectInstance(
    ObjectOp op, ArrayRef<ObjectValue> actualParams) {
  // First, check if we have already evaluated this object, and return it if so.
  auto existingInstance = objects.find(op);
  if (existingInstance != objects.end())
    return success(existingInstance->second);

  // If we need to instantiate a new object, evaluate values for all of its
  // actual parameters. Note that this is eager evaluation, which precludes
  // creating cycles in the object model. Further refinement is expected.
  SmallVector<ObjectValue> objectParams;
  for (auto param : op.getActualParams()) {
    FailureOr<ObjectValue> result = evaluateValue(param, actualParams);
    if (failed(result))
      return result;
    objectParams.push_back(result.value());
  }

  // Instantiate and return the new Object, saving the instance for later.
  auto newInstance = instantiate(op.getClassNameAttr(), objectParams);
  if (succeeded(newInstance))
    objects[op.getResult()] = newInstance.value();
  return newInstance;
}

/// Evaluator dispatch function for Object fields.
FailureOr<ObjectValue>
circt::om::Evaluator::evaluateObjectField(ObjectFieldOp op,
                                          ArrayRef<ObjectValue> actualParams) {
  // Evaluate the Object itself, in case it hasn't been evaluated yet.
  FailureOr<ObjectValue> currentObjectResult =
      evaluateValue(op.getObject(), actualParams);
  if (failed(currentObjectResult))
    return currentObjectResult;

  std::shared_ptr<Object> currentObject =
      std::get<std::shared_ptr<Object>>(currentObjectResult.value());

  // Iteratively access nested fields through the path until we reach the final
  // field in the path.
  ObjectValue finalField;
  for (auto field : op.getFieldPath().getAsRange<FlatSymbolRefAttr>()) {
    auto currentField = currentObject->getField(field.getAttr());
    finalField = currentField.value();
    if (auto *nextObject = std::get_if<std::shared_ptr<Object>>(&finalField))
      currentObject = *nextObject;
  }

  // Return the field being accessed.
  return finalField;
}

/// Construct an Object of the given type with the given fields.
circt::om::Object::Object(ClassOp cls, ObjectFields &fields)
    : cls(cls), fields(fields) {}

/// Get the type of the Object.
Type circt::om::Object::getType() {
  return ClassType::get(cls.getContext(),
                        FlatSymbolRefAttr::get(cls.getNameAttr()));
}

/// Get a field of the Object by name.
FailureOr<ObjectValue> circt::om::Object::getField(StringAttr name) {
  auto field = fields.find(name);
  if (field == fields.end())
    return cls.emitError("field ") << name << " does not exist";
  return success(fields[name]);
}

/// Get an ArrayAttr with the names of the fields in the Object. Sort the fields
/// so there is always a stable order.
ArrayAttr circt::om::Object::getFieldNames() {
  SmallVector<Attribute> fieldNames;
  for (auto &f : fields)
    fieldNames.push_back(f.first);

  llvm::sort(fieldNames, [](Attribute a, Attribute b) {
    return cast<StringAttr>(a).getValue() < cast<StringAttr>(b).getValue();
  });

  return ArrayAttr::get(cls.getContext(), fieldNames);
}
