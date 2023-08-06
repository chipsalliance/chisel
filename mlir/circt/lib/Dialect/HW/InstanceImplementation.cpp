//===- InstanceImplementation.cpp - Utilities for instance-like ops -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/InstanceImplementation.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"

using namespace circt;
using namespace circt::hw;

std::pair<ArrayAttr, ArrayAttr>
instance_like_impl::getHWModuleArgAndResultNames(Operation *module) {
  assert(isAnyModule(module) && "Can only reference a module");

  return {module->getAttrOfType<ArrayAttr>("argNames"),
          module->getAttrOfType<ArrayAttr>("resultNames")};
}

Operation *
instance_like_impl::getReferencedModule(const HWSymbolCache *cache,
                                        Operation *instanceOp,
                                        mlir::FlatSymbolRefAttr moduleName) {
  if (cache)
    if (auto *result = cache->getDefinition(moduleName))
      return result;

  auto topLevelModuleOp = instanceOp->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(moduleName.getValue());
}

LogicalResult instance_like_impl::verifyReferencedModule(
    Operation *instanceOp, SymbolTableCollection &symbolTable,
    mlir::FlatSymbolRefAttr moduleName, Operation *&module) {
  module = symbolTable.lookupNearestSymbolFrom(instanceOp, moduleName);
  if (module == nullptr)
    return instanceOp->emitError("Cannot find module definition '")
           << moduleName.getValue() << "'";

  // It must be some sort of module.
  if (!hw::isAnyModule(module))
    return instanceOp->emitError("symbol reference '")
           << moduleName.getValue() << "' isn't a module";

  return success();
}

LogicalResult instance_like_impl::resolveParametricTypes(
    Location loc, ArrayAttr parameters, ArrayRef<Type> types,
    SmallVectorImpl<Type> &resolvedTypes, const EmitErrorFn &emitError) {
  for (auto type : types) {
    auto expectedType = evaluateParametricType(loc, parameters, type);
    if (failed(expectedType)) {
      emitError([&](auto &diag) {
        diag << "failed to resolve parametric input of instantiated module";
        return true;
      });
      return failure();
    }

    resolvedTypes.push_back(*expectedType);
  }

  return success();
}

LogicalResult instance_like_impl::verifyInputs(ArrayAttr argNames,
                                               ArrayAttr moduleArgNames,
                                               TypeRange inputTypes,
                                               ArrayRef<Type> moduleInputTypes,
                                               const EmitErrorFn &emitError) {
  // Check operand types first.
  if (moduleInputTypes.size() != inputTypes.size()) {
    emitError([&](auto &diag) {
      diag << "has a wrong number of operands; expected "
           << moduleInputTypes.size() << " but got " << inputTypes.size();
      return true;
    });
    return failure();
  }

  if (argNames.size() != inputTypes.size()) {
    emitError([&](auto &diag) {
      diag << "has a wrong number of input port names; expected "
           << inputTypes.size() << " but got " << argNames.size();
      return true;
    });
    return failure();
  }

  for (size_t i = 0; i != inputTypes.size(); ++i) {
    auto expectedType = moduleInputTypes[i];
    auto operandType = inputTypes[i];

    if (operandType != expectedType) {
      emitError([&](auto &diag) {
        diag << "operand type #" << i << " must be " << expectedType
             << ", but got " << operandType;
        return true;
      });
      return failure();
    }

    if (argNames[i] != moduleArgNames[i]) {
      emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << moduleArgNames[i]
             << ", but got " << argNames[i];
        return true;
      });
      return failure();
    }
  }

  return success();
}

LogicalResult instance_like_impl::verifyOutputs(
    ArrayAttr resultNames, ArrayAttr moduleResultNames, TypeRange resultTypes,
    ArrayRef<Type> moduleResultTypes, const EmitErrorFn &emitError) {
  // Check result types and labels.
  if (moduleResultTypes.size() != resultTypes.size()) {
    emitError([&](auto &diag) {
      diag << "has a wrong number of results; expected "
           << moduleResultTypes.size() << " but got " << resultTypes.size();
      return true;
    });
    return failure();
  }

  if (resultNames.size() != resultTypes.size()) {
    emitError([&](auto &diag) {
      diag << "has a wrong number of results port labels; expected "
           << resultTypes.size() << " but got " << resultNames.size();
      return true;
    });
    return failure();
  }

  for (size_t i = 0; i != resultTypes.size(); ++i) {
    auto expectedType = moduleResultTypes[i];
    auto resultType = resultTypes[i];

    if (resultType != expectedType) {
      emitError([&](auto &diag) {
        diag << "result type #" << i << " must be " << expectedType
             << ", but got " << resultType;
        return true;
      });
      return failure();
    }

    if (resultNames[i] != moduleResultNames[i]) {
      emitError([&](auto &diag) {
        diag << "result label #" << i << " must be " << moduleResultNames[i]
             << ", but got " << resultNames[i];
        return true;
      });
      return failure();
    }
  }

  return success();
}

LogicalResult
instance_like_impl::verifyParameters(ArrayAttr parameters,
                                     ArrayAttr moduleParameters,
                                     const EmitErrorFn &emitError) {
  // Check parameters match up.
  auto numParameters = parameters.size();
  if (numParameters != moduleParameters.size()) {
    emitError([&](auto &diag) {
      diag << "expected " << moduleParameters.size() << " parameters but had "
           << numParameters;
      return true;
    });
    return failure();
  }

  for (size_t i = 0; i != numParameters; ++i) {
    auto param = parameters[i].cast<ParamDeclAttr>();
    auto modParam = moduleParameters[i].cast<ParamDeclAttr>();

    auto paramName = param.getName();
    if (paramName != modParam.getName()) {
      emitError([&](auto &diag) {
        diag << "parameter #" << i << " should have name " << modParam.getName()
             << " but has name " << paramName;
        return true;
      });
      return failure();
    }

    if (param.getType() != modParam.getType()) {
      emitError([&](auto &diag) {
        diag << "parameter " << paramName << " should have type "
             << modParam.getType() << " but has type " << param.getType();
        return true;
      });
      return failure();
    }

    // All instance parameters must have a value.  Specify the same value as
    // a module's default value if you want the default.
    if (!param.getValue()) {
      emitError([&](auto &diag) {
        diag << "parameter " << paramName << " must have a value";
        return false;
      });
      return failure();
    }
  }

  return success();
}

LogicalResult instance_like_impl::verifyInstanceOfHWModule(
    Operation *instance, FlatSymbolRefAttr moduleRef, OperandRange inputs,
    TypeRange results, ArrayAttr argNames, ArrayAttr resultNames,
    ArrayAttr parameters, SymbolTableCollection &symbolTable) {
  // Verify that we reference some kind of HW module and get the module on
  // success.
  Operation *module;
  if (failed(instance_like_impl::verifyReferencedModule(instance, symbolTable,
                                                        moduleRef, module)))
    return failure();

  // Emit an error message on the instance, with a note indicating which module
  // is being referenced. The error message on the instance is added by the
  // verification function this lambda is passed to.
  EmitErrorFn emitError =
      [&](const std::function<bool(InFlightDiagnostic & diag)> &fn) {
        auto diag = instance->emitOpError();
        if (fn(diag))
          diag.attachNote(module->getLoc()) << "module declared here";
      };

  // Check that input types are consistent with the referenced module.
  auto [modArgNames, modResultNames] =
      instance_like_impl::getHWModuleArgAndResultNames(module);

  ArrayRef<Type> resolvedModInputTypesRef = getModuleType(module).getInputs();
  SmallVector<Type> resolvedModInputTypes;
  if (parameters) {
    if (failed(instance_like_impl::resolveParametricTypes(
            instance->getLoc(), parameters, getModuleType(module).getInputs(),
            resolvedModInputTypes, emitError)))
      return failure();
    resolvedModInputTypesRef = resolvedModInputTypes;
  }
  if (failed(instance_like_impl::verifyInputs(
          argNames, modArgNames, inputs.getTypes(), resolvedModInputTypesRef,
          emitError)))
    return failure();

  // Check that result types are consistent with the referenced module.
  ArrayRef<Type> resolvedModResultTypesRef = getModuleType(module).getResults();
  SmallVector<Type> resolvedModResultTypes;
  if (parameters) {
    if (failed(instance_like_impl::resolveParametricTypes(
            instance->getLoc(), parameters, getModuleType(module).getResults(),
            resolvedModResultTypes, emitError)))
      return failure();
    resolvedModResultTypesRef = resolvedModResultTypes;
  }
  if (failed(instance_like_impl::verifyOutputs(
          resultNames, modResultNames, results, resolvedModResultTypesRef,
          emitError)))
    return failure();

  if (parameters) {
    // Check that the parameters are consistent with the referenced module.
    ArrayAttr modParameters = module->getAttrOfType<ArrayAttr>("parameters");
    if (failed(instance_like_impl::verifyParameters(parameters, modParameters,
                                                    emitError)))
      return failure();
  }

  return success();
}

LogicalResult
instance_like_impl::verifyParameterStructure(ArrayAttr parameters,
                                             ArrayAttr moduleParameters,
                                             const EmitErrorFn &emitError) {
  // Check that all the parameter values specified to the instance are
  // structurally valid.
  for (auto param : parameters) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto value = paramAttr.getValue();
    // The SymbolUses verifier which checks that this exists may not have been
    // run yet. Let it issue the error.
    if (!value)
      continue;

    auto typedValue = value.dyn_cast<mlir::TypedAttr>();
    if (!typedValue) {
      emitError([&](auto &diag) {
        diag << "parameter " << paramAttr
             << " should have a typed value; has value " << value;
        return false;
      });
      return failure();
    }

    if (typedValue.getType() != paramAttr.getType()) {
      emitError([&](auto &diag) {
        diag << "parameter " << paramAttr << " should have type "
             << paramAttr.getType() << "; has type " << typedValue.getType();
        return false;
      });
      return failure();
    }

    if (failed(checkParameterInContext(value, moduleParameters, emitError)))
      return failure();
  }
  return success();
}

StringAttr instance_like_impl::getName(ArrayAttr names, size_t idx) {
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

ArrayAttr instance_like_impl::updateName(ArrayAttr oldNames, size_t i,
                                         StringAttr name) {
  SmallVector<Attribute> newNames(oldNames.begin(), oldNames.end());
  if (newNames[i] == name)
    return oldNames;
  newNames[i] = name;
  return ArrayAttr::get(oldNames.getContext(), oldNames);
}

void instance_like_impl::getAsmResultNames(OpAsmSetValueNameFn setNameFn,
                                           StringRef instanceName,
                                           ArrayAttr resultNames,
                                           ValueRange results) {
  // Provide default names for instance results.
  std::string name = instanceName.str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = resultNames.size(); i != e; ++i) {
    auto resName = getName(resultNames, i);
    name.resize(baseNameLen);
    if (resName && !resName.getValue().empty())
      name += resName.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(results[i], name);
  }
}
