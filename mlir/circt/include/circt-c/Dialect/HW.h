//===-- circt-c/Dialect/HW.h - C API for HW dialect ---------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// HW dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_HW_H
#define CIRCT_C_DIALECT_HW_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

struct HWStructFieldInfo {
  MlirIdentifier name;
  MlirType type;
};
typedef struct HWStructFieldInfo HWStructFieldInfo;

//===----------------------------------------------------------------------===//
// Dialect API.
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(HW, hw);

//===----------------------------------------------------------------------===//
// Type API.
//===----------------------------------------------------------------------===//

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
MLIR_CAPI_EXPORTED int64_t hwGetBitWidth(MlirType);

/// Return true if the specified type can be used as an HW value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType or unknown types from other
/// dialects.
MLIR_CAPI_EXPORTED bool hwTypeIsAValueType(MlirType);

/// If the type is an HW array
MLIR_CAPI_EXPORTED bool hwTypeIsAArrayType(MlirType);

/// If the type is an HW inout.
MLIR_CAPI_EXPORTED bool hwTypeIsAInOut(MlirType type);

/// If the type is an HW struct.
MLIR_CAPI_EXPORTED bool hwTypeIsAStructType(MlirType);

/// If the type is an HW type alias.
MLIR_CAPI_EXPORTED bool hwTypeIsATypeAliasType(MlirType);

/// If the type is an HW int.
MLIR_CAPI_EXPORTED bool hwTypeIsAIntType(MlirType);

/// Creates a fixed-size HW array type in the context associated with element
MLIR_CAPI_EXPORTED MlirType hwArrayTypeGet(MlirType element, size_t size);

/// returns the element type of an array type
MLIR_CAPI_EXPORTED MlirType hwArrayTypeGetElementType(MlirType);

/// returns the size of an array type
MLIR_CAPI_EXPORTED intptr_t hwArrayTypeGetSize(MlirType);

/// Creates an HW inout type in the context associated with element.
MLIR_CAPI_EXPORTED MlirType hwInOutTypeGet(MlirType element);

/// Returns the element type of an inout type.
MLIR_CAPI_EXPORTED MlirType hwInOutTypeGetElementType(MlirType);

/// Creates an HW struct type in the context associated with the elements.
MLIR_CAPI_EXPORTED MlirType hwStructTypeGet(MlirContext ctx,
                                            intptr_t numElements,
                                            HWStructFieldInfo const *elements);

MLIR_CAPI_EXPORTED MlirType hwStructTypeGetField(MlirType structType,
                                                 MlirStringRef fieldName);

MLIR_CAPI_EXPORTED MlirType hwParamIntTypeGet(MlirAttribute parameter);

MLIR_CAPI_EXPORTED MlirAttribute hwParamIntTypeGetWidthAttr(MlirType);

MLIR_CAPI_EXPORTED HWStructFieldInfo
hwStructTypeGetFieldNum(MlirType structType, unsigned idx);
MLIR_CAPI_EXPORTED intptr_t hwStructTypeGetNumFields(MlirType structType);

MLIR_CAPI_EXPORTED MlirType hwTypeAliasTypeGet(MlirStringRef scope,
                                               MlirStringRef name,
                                               MlirType innerType);

MLIR_CAPI_EXPORTED MlirType hwTypeAliasTypeGetCanonicalType(MlirType typeAlias);

MLIR_CAPI_EXPORTED MlirType hwTypeAliasTypeGetInnerType(MlirType typeAlias);

MLIR_CAPI_EXPORTED MlirStringRef hwTypeAliasTypeGetName(MlirType typeAlias);

MLIR_CAPI_EXPORTED MlirStringRef hwTypeAliasTypeGetScope(MlirType typeAlias);

//===----------------------------------------------------------------------===//
// Attribute API.
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool hwAttrIsAInnerSymAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerSymAttrGet(MlirAttribute symName);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerSymAttrGetSymName(MlirAttribute);

MLIR_CAPI_EXPORTED bool hwAttrIsAInnerRefAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerRefAttrGet(MlirAttribute moduleName,
                                                   MlirAttribute innerSym);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerRefAttrGetName(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwInnerRefAttrGetModule(MlirAttribute);

MLIR_CAPI_EXPORTED bool hwAttrIsAGlobalRefAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwGlobalRefAttrGet(MlirAttribute symName);

MLIR_CAPI_EXPORTED bool hwAttrIsAParamDeclAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclAttrGet(MlirStringRef name,
                                                    MlirType type,
                                                    MlirAttribute value);
MLIR_CAPI_EXPORTED MlirStringRef hwParamDeclAttrGetName(MlirAttribute decl);
MLIR_CAPI_EXPORTED MlirType hwParamDeclAttrGetType(MlirAttribute decl);
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclAttrGetValue(MlirAttribute decl);

MLIR_CAPI_EXPORTED bool hwAttrIsAParamDeclRefAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwParamDeclRefAttrGet(MlirContext ctx,
                                                       MlirStringRef cName);
MLIR_CAPI_EXPORTED MlirStringRef hwParamDeclRefAttrGetName(MlirAttribute decl);
MLIR_CAPI_EXPORTED MlirType hwParamDeclRefAttrGetType(MlirAttribute decl);

MLIR_CAPI_EXPORTED bool hwAttrIsAParamVerbatimAttr(MlirAttribute);
MLIR_CAPI_EXPORTED MlirAttribute hwParamVerbatimAttrGet(MlirAttribute text);

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_HW_H
