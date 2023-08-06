//===- OM.cpp - C Interface for the OM Dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements a C Interface for the OM Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/OM.h"
#include "circt/Dialect/OM/Evaluator/Evaluator.h"
#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"

using namespace mlir;
using namespace circt::om;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(OM, om, OMDialect)

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Is the Type a ClassType.
bool omTypeIsAClassType(MlirType type) { return unwrap(type).isa<ClassType>(); }

//===----------------------------------------------------------------------===//
// Evaluator data structures.
//===----------------------------------------------------------------------===//

DEFINE_C_API_PTR_METHODS(OMEvaluator, circt::om::Evaluator)

/// Define our own wrap and unwrap instead of using the usual macro. This is To
/// handle the std::shared_ptr reference counts appropriately. We want to always
/// create *new* shared pointers to the Object when we wrap it for C, to
/// increment the reference count. We want to use the shared_from_this
/// functionality to ensure it is unwrapped into C++ with the correct reference
/// count.

static inline OMObject wrap(std::shared_ptr<Object> object) {
  return OMObject{static_cast<void *>(
      (new std::shared_ptr<Object>(std::move(object)))->get())};
}

static inline std::shared_ptr<Object> unwrap(OMObject c) {
  return static_cast<Object *>(c.ptr)->shared_from_this();
}

//===----------------------------------------------------------------------===//
// Evaluator API.
//===----------------------------------------------------------------------===//

/// Construct an Evaluator with an IR module.
OMEvaluator omEvaluatorNew(MlirModule mod) {
  // Just allocate and wrap the Evaluator.
  return wrap(new Evaluator(unwrap(mod)));
}

/// Use the Evaluator to Instantiate an Object from its class name and actual
/// parameters.
OMObject omEvaluatorInstantiate(OMEvaluator evaluator, MlirAttribute className,
                                intptr_t nActualParams,
                                MlirAttribute const *actualParams) {
  // Unwrap the Evaluator.
  Evaluator *cppEvaluator = unwrap(evaluator);

  // Unwrap the className, which the client must supply as a StringAttr.
  StringAttr cppClassName = unwrap(className).cast<StringAttr>();

  // Unwrap the actual parameters, which the client must supply as Attributes.
  SmallVector<Attribute> actualParamsTmp;
  SmallVector<ObjectValue> cppActualParams(
      unwrapList(nActualParams, actualParams, actualParamsTmp));

  // Invoke the Evaluator to instantiate the Object.
  FailureOr<std::shared_ptr<Object>> result =
      cppEvaluator->instantiate(cppClassName, cppActualParams);

  // If instantiation failed, return a null Object. A Diagnostic will be emitted
  // in this case.
  if (failed(result))
    return OMObject();

  // Wrap and return the Object.
  return wrap(result.value());
}

/// Get the Module the Evaluator is built from.
MlirModule omEvaluatorGetModule(OMEvaluator evaluator) {
  // Just unwrap the Evaluator, get the Module, and wrap it.
  return wrap(unwrap(evaluator)->getModule());
}

//===----------------------------------------------------------------------===//
// Object API.
//===----------------------------------------------------------------------===//

/// Query if the Object is null.
bool omEvaluatorObjectIsNull(OMObject object) {
  // Just check if the Object shared pointer is null.
  return !object.ptr;
}

/// Get the Type from an Object, which will be a ClassType.
MlirType omEvaluatorObjectGetType(OMObject object) {
  return wrap(unwrap(object)->getType());
}

/// Get an ArrayAttr with the names of the fields in an Object.
MlirAttribute omEvaluatorObjectGetFieldNames(OMObject object) {
  return wrap(unwrap(object)->getFieldNames());
}

/// Get a field from an Object, which must contain a field of that name.
OMObjectValue omEvaluatorObjectGetField(OMObject object, MlirAttribute name) {
  // Unwrap the Object and get the field of the name, which the client must
  // supply as a StringAttr.
  FailureOr<ObjectValue> result =
      unwrap(object)->getField(unwrap(name).cast<StringAttr>());

  // If getField failed, return a null ObjectValue. A Diagnostic will be emitted
  // in this case.
  if (failed(result))
    return OMObjectValue();

  // If the field is an Object, return an ObjectValue with the Object set.
  if (auto *object = std::get_if<std::shared_ptr<Object>>(&result.value()))
    return OMObjectValue{MlirAttribute(), wrap(*object)};

  // If the field is an Attribute, return an ObjectValue with the Primitive set.
  if (auto *primitive = std::get_if<Attribute>(&result.value()))
    return OMObjectValue{wrap(*primitive), OMObject()};

  // This case should never be hit, but return a null ObjectValue that is
  // neither an Object nor a Primitive.
  return OMObjectValue();
}

//===----------------------------------------------------------------------===//
// ObjectValue API.
//===----------------------------------------------------------------------===//

// Query if the ObjectValue is null.
bool omEvaluatorObjectValueIsNull(OMObjectValue objectValue) {
  // Check if both Object and Attribute are null.
  return !omEvaluatorObjectValueIsAObject(objectValue) &&
         !omEvaluatorObjectValueIsAPrimitive(objectValue);
}

/// Query if the ObjectValue is an Object.
bool omEvaluatorObjectValueIsAObject(OMObjectValue objectValue) {
  // Check if the Object is non-null.
  return !omEvaluatorObjectIsNull(objectValue.object);
}

/// Get the Object from an  ObjectValue, which must contain an Object.
OMObject omEvaluatorObjectValueGetObject(OMObjectValue objectValue) {
  // Assert the Object is non-null, and return it.
  assert(omEvaluatorObjectValueIsAObject(objectValue));
  return objectValue.object;
}

/// Query if the ObjectValue is a Primitive.
bool omEvaluatorObjectValueIsAPrimitive(OMObjectValue objectValue) {
  // Check if the Attribute is non-null.
  return !mlirAttributeIsNull(objectValue.primitive);
}

/// Get the Primitive from an  ObjectValue, which must contain a Primitive.
MlirAttribute omEvaluatorObjectValueGetPrimitive(OMObjectValue objectValue) {
  // Assert the Attribute is non-null, and return it.
  assert(omEvaluatorObjectValueIsAPrimitive(objectValue));
  return objectValue.primitive;
}

//===----------------------------------------------------------------------===//
// ReferenceAttr API.
//===----------------------------------------------------------------------===//

bool omAttrIsAReferenceAttr(MlirAttribute attr) {
  return unwrap(attr).isa<ReferenceAttr>();
}

MlirAttribute omReferenceAttrGetInnerRef(MlirAttribute referenceAttr) {
  return wrap(
      (Attribute)unwrap(referenceAttr).cast<ReferenceAttr>().getInnerRef());
}

//===----------------------------------------------------------------------===//
// ListAttr API.
//===----------------------------------------------------------------------===//

bool omAttrIsAListAttr(MlirAttribute attr) {
  return unwrap(attr).isa<ListAttr>();
}

intptr_t omListAttrGetNumElements(MlirAttribute attr) {
  auto listAttr = llvm::cast<ListAttr>(unwrap(attr));
  return static_cast<intptr_t>(listAttr.getElements().size());
}

MlirAttribute omListAttrGetElement(MlirAttribute attr, intptr_t pos) {
  auto listAttr = llvm::cast<ListAttr>(unwrap(attr));
  return wrap(listAttr.getElements()[pos]);
}
