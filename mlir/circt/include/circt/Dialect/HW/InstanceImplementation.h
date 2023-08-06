//===- InstanceImplementation.h - Instance-like Op utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utility functions for implementing instance-like
// operations, in particular, parsing, and printing common to instance-like
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_INSTANCEIMPLEMENTATION_H
#define CIRCT_DIALECT_HW_INSTANCEIMPLEMENTATION_H

#include "circt/Support/LLVM.h"
#include <functional>

namespace circt {
namespace hw {
// Forward declarations.
class HWSymbolCache;

namespace instance_like_impl {

/// Whenever the nested function returns true, a note referring to the
/// referenced module is attached to the error.
using EmitErrorFn =
    std::function<void(std::function<bool(InFlightDiagnostic &)>)>;

/// Convenience function to query the input and result names of a HW module (as
/// determined by 'hw::isAnyModule').
std::pair<ArrayAttr, ArrayAttr> getHWModuleArgAndResultNames(Operation *module);

/// Return a pointer to the referenced module operation.
Operation *getReferencedModule(const HWSymbolCache *cache,
                               Operation *instanceOp,
                               mlir::FlatSymbolRefAttr moduleName);

/// Verify that the instance refers to a valid HW module as determined by the
/// 'hw::isAnyModule' function.
LogicalResult verifyReferencedModule(Operation *instanceOp,
                                     SymbolTableCollection &symbolTable,
                                     mlir::FlatSymbolRefAttr moduleName,
                                     Operation *&module);

/// Stores a resolved version of each type in @param types wherein any parameter
/// reference has been evaluated based on the set of provided @param parameters
/// in @param resolvedTypes
LogicalResult resolveParametricTypes(Location loc, ArrayAttr parameters,
                                     ArrayRef<Type> types,
                                     SmallVectorImpl<Type> &resolvedTypes,
                                     const EmitErrorFn &emitError);

/// Verify that the list of inputs of the instance and the module match in terms
/// of length, names, and types.
LogicalResult verifyInputs(ArrayAttr argNames, ArrayAttr moduleArgNames,
                           TypeRange inputTypes,
                           ArrayRef<Type> moduleInputTypes,
                           const EmitErrorFn &emitError);

/// Verify that the list of outputs of the instance and the module match in
/// terms of length, names, and types.
LogicalResult verifyOutputs(ArrayAttr resultNames, ArrayAttr moduleResultNames,
                            TypeRange resultTypes,
                            ArrayRef<Type> moduleResultTypes,
                            const EmitErrorFn &emitError);

/// Verify that the parameter lists of the instance and the module match in
/// terms of length, names, and types.
LogicalResult verifyParameters(ArrayAttr parameters, ArrayAttr moduleParameters,
                               const EmitErrorFn &emitError);

/// Combines verifyReferencedModule, verifyInputs, verifyOutputs, and
/// verifyParameters. It is only allowed to call this function when the instance
/// refers to a HW module. The @param parameters attribute may be null in which
/// case not parameters are verified.
LogicalResult verifyInstanceOfHWModule(
    Operation *instance, FlatSymbolRefAttr moduleRef, OperandRange inputs,
    TypeRange results, ArrayAttr argNames, ArrayAttr resultNames,
    ArrayAttr parameters, SymbolTableCollection &symbolTable);

/// Check that all the parameter values specified to the instance are
/// structurally valid.
LogicalResult verifyParameterStructure(ArrayAttr parameters,
                                       ArrayAttr moduleParameters,
                                       const EmitErrorFn &emitError);

/// Return the name at the specified index of the ArrayAttr or null if it cannot
/// be determined.
StringAttr getName(ArrayAttr names, size_t idx);

/// Change the name at the specified index of the @param oldNames ArrayAttr to
/// @param name
ArrayAttr updateName(ArrayAttr oldNames, size_t i, StringAttr name);

/// Suggest a name for each result value based on the saved result names
/// attribute.
void getAsmResultNames(OpAsmSetValueNameFn setNameFn, StringRef instanceName,
                       ArrayAttr resultNames, ValueRange results);

} // namespace instance_like_impl
} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_INSTANCEIMPLEMENTATION_H
