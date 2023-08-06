//===-- circt-c/Dialect/OM.h - C API for OM dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// OM dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_OM_H
#define CIRCT_C_DIALECT_OM_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(OM, om);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Is the Type a ClassType.
MLIR_CAPI_EXPORTED bool omTypeIsAClassType(MlirType type);

//===----------------------------------------------------------------------===//
// Evaluator data structures.
//===----------------------------------------------------------------------===//

/// A value type for use in C APIs that just wraps a pointer to an Evaluator.
/// This is in line with the usual MLIR DEFINE_C_API_STRUCT.
struct OMEvaluator {
  void *ptr;
};

// clang-tidy doesn't respect extern "C".
// see https://github.com/llvm/llvm-project/issues/35272.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct OMEvaluator OMEvaluator;

/// A value type for use in C APIs that just wraps a pointer to an Object.
/// This is in line with the usual MLIR DEFINE_C_API_STRUCT.
struct OMObject {
  void *ptr;
};

// clang-tidy doesn't respect extern "C".
// see https://github.com/llvm/llvm-project/issues/35272.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct OMObject OMObject;

/// A value type for use in C APIs that represents an ObjectValue.
/// Because ObjectValue is a std::variant, which doesn't work well with C APIs,
/// we use a struct with both fields, one of which will always be null.
struct OMObjectValue {
  MlirAttribute primitive;
  OMObject object;
};

// clang-tidy doesn't respect extern "C".
// see https://github.com/llvm/llvm-project/issues/35272.
// NOLINTNEXTLINE(modernize-use-using)
typedef struct OMObjectValue OMObjectValue;

//===----------------------------------------------------------------------===//
// Evaluator API.
//===----------------------------------------------------------------------===//

/// Construct an Evaluator with an IR module.
MLIR_CAPI_EXPORTED OMEvaluator omEvaluatorNew(MlirModule mod);

/// Use the Evaluator to Instantiate an Object from its class name and actual
/// parameters.
MLIR_CAPI_EXPORTED OMObject omEvaluatorInstantiate(
    OMEvaluator evaluator, MlirAttribute className, intptr_t nActualParams,
    MlirAttribute const *actualParams);

/// Get the Module the Evaluator is built from.
MLIR_CAPI_EXPORTED MlirModule omEvaluatorGetModule(OMEvaluator evaluator);

//===----------------------------------------------------------------------===//
// Object API.
//===----------------------------------------------------------------------===//

/// Query if the Object is null.
MLIR_CAPI_EXPORTED bool omEvaluatorObjectIsNull(OMObject object);

/// Get the Type from an Object, which will be a ClassType.
MLIR_CAPI_EXPORTED MlirType omEvaluatorObjectGetType(OMObject object);

/// Get a field from an Object, which must contain a field of that name.
MLIR_CAPI_EXPORTED OMObjectValue omEvaluatorObjectGetField(OMObject object,
                                                           MlirAttribute name);

/// Get all the field names from an Object, can be empty if object has no
/// fields.
MLIR_CAPI_EXPORTED MlirAttribute
omEvaluatorObjectGetFieldNames(OMObject object);

//===----------------------------------------------------------------------===//
// ObjectValue API.
//===----------------------------------------------------------------------===//

// Query if the ObjectValue is null.
MLIR_CAPI_EXPORTED bool omEvaluatorObjectValueIsNull(OMObjectValue objectValue);

/// Query if the ObjectValue is an Object.
MLIR_CAPI_EXPORTED bool
omEvaluatorObjectValueIsAObject(OMObjectValue objectValue);

/// Get the Object from an  ObjectValue, which must contain an Object.
MLIR_CAPI_EXPORTED OMObject
omEvaluatorObjectValueGetObject(OMObjectValue objectValue);

/// Query if the ObjectValue is a Primitive.
MLIR_CAPI_EXPORTED bool
omEvaluatorObjectValueIsAPrimitive(OMObjectValue objectValue);

/// Get the Primitive from an  ObjectValue, which must contain a Primitive.
MLIR_CAPI_EXPORTED MlirAttribute
omEvaluatorObjectValueGetPrimitive(OMObjectValue objectValue);

//===----------------------------------------------------------------------===//
// ReferenceAttr API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool omAttrIsAReferenceAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute omReferenceAttrGetInnerRef(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// ListAttr API
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool omAttrIsAListAttr(MlirAttribute attr);

MLIR_CAPI_EXPORTED intptr_t omListAttrGetNumElements(MlirAttribute attr);

MLIR_CAPI_EXPORTED MlirAttribute omListAttrGetElement(MlirAttribute attr,
                                                      intptr_t pos);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_OM_H
