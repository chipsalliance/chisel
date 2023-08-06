//===- LegalizeAnonEnums.cpp - Legalizes anonymous enumerations -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces all anonymous enumeration with typedecls in the output
// Verilog.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "ExportVerilogInternals.h"
#include "circt/Conversion/ExportVerilog.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseSet.h"

using namespace circt;
using namespace hw;
using namespace sv;

namespace {
struct LegalizeAnonEnums : public LegalizeAnonEnumsBase<LegalizeAnonEnums> {
  /// Creates a TypeScope on demand for anonymous enumerations.
  TypeScopeOp getTypeScope() {
    auto topLevel = getOperation();
    if (!typeScope) {
      auto builder = OpBuilder::atBlockBegin(&topLevel.getRegion().front());
      typeScope = builder.create<TypeScopeOp>(topLevel.getLoc(), "Enums");
      typeScope.getBodyRegion().push_back(new Block());
      mlir::SymbolTable symbolTable(topLevel);
      symbolTable.insert(typeScope);
    }
    return typeScope;
  }

  /// Helper to create TypeDecls and TypeAliases for EnumTypes;
  Type getEnumTypeDecl(EnumType type) {
    auto &typeAlias = enumTypeAliases[type];
    if (typeAlias)
      return typeAlias;
    auto *context = &getContext();
    auto loc = UnknownLoc::get(context);
    auto typeScope = getTypeScope();
    auto builder = OpBuilder::atBlockEnd(&typeScope.getRegion().front());
    auto declName = StringAttr::get(context, "enum" + Twine(enumCount++));
    builder.create<TypedeclOp>(loc, declName, TypeAttr::get(type), nullptr);
    auto symRef = SymbolRefAttr::get(typeScope.getSymNameAttr(),
                                     FlatSymbolRefAttr::get(declName));
    typeAlias = TypeAliasType::get(symRef, type);
    return typeAlias;
  }

  /// Process a type, replacing any anonymous enumerations contained within.
  Type processType(Type type) {
    auto *context = &getContext();
    if (auto structType = dyn_cast<StructType>(type)) {
      bool changed = false;
      SmallVector<StructType::FieldInfo> fields;
      for (auto &element : structType.getElements()) {
        if (auto newFieldType = processType(element.type)) {
          changed = true;
          fields.push_back({element.name, newFieldType});
        } else {
          fields.push_back(element);
        }
      }
      if (changed)
        return StructType::get(context, fields);
      return {};
    }

    if (auto arrayType = dyn_cast<ArrayType>(type)) {
      if (auto newElementType = processType(arrayType.getElementType()))
        return ArrayType::get(newElementType, arrayType.getSize());
      return {};
    }

    if (auto unionType = dyn_cast<UnionType>(type)) {
      bool changed = false;
      SmallVector<UnionType::FieldInfo> fields;
      for (const auto &element : unionType.getElements()) {
        if (auto newFieldType = processType(element.type)) {
          fields.push_back({element.name, newFieldType, element.offset});
          changed = true;
        } else {
          fields.push_back(element);
        }
      }
      if (changed)
        return UnionType::get(context, fields);
      return {};
    }

    if (auto typeAlias = dyn_cast<TypeAliasType>(type)) {
      // Enum type aliases have already been handled.
      if (isa<EnumType>(typeAlias.getInnerType()))
        return {};
      // Otherwise recursively update the type alias.
      return processType(typeAlias.getInnerType());
    }

    if (auto inoutType = dyn_cast<InOutType>(type)) {
      if (auto newType = processType(inoutType.getElementType()))
        return InOutType::get(newType);
      return {};
    }

    // EnumTypes must be changed into TypeAlias.
    if (auto enumType = dyn_cast<EnumType>(type))
      return getEnumTypeDecl(enumType);

    if (auto funcType = dyn_cast<FunctionType>(type)) {
      bool changed = false;
      SmallVector<Type> inputs;
      for (auto &type : funcType.getInputs()) {
        if (auto newType = processType(type)) {
          inputs.push_back(newType);
          changed = true;
        } else {
          inputs.push_back(type);
        }
      }
      SmallVector<Type> results;
      for (auto &type : funcType.getResults()) {
        if (auto newType = processType(type)) {
          results.push_back(newType);
          changed = true;
        } else {
          results.push_back(type);
        }
      }
      if (changed)
        return FunctionType::get(context, inputs, results);
      return {};
    }

    // Default case is that it is not an aggregate type.
    return {};
  };

  void runOnOperation() override {
    enumCount = 0;
    typeScope = {};

    // Perform the actual walk looking for anonymous enumeration types.
    getOperation().walk([&](Operation *op) {
      // If this is a constant operation, make sure to update the constant
      // to reference the typedef, otherwise we will emit the wrong constant.
      // Theoretically we should be searching all attributes on every operation
      // for EnumFieldAttrs.
      if (auto enumConst = dyn_cast<EnumConstantOp>(op)) {
        auto fieldAttr = enumConst.getField();
        if (auto newType = processType(fieldAttr.getType().getValue()))
          enumConst.setFieldAttr(
              EnumFieldAttr::get(op->getLoc(), fieldAttr.getField(), newType));
      }

      // Update the operation signature if it is function-like.
      if (auto funcLike = dyn_cast<mlir::FunctionOpInterface>(op))
        if (auto newType = processType(funcLike.getFunctionType()))
          funcLike.setFunctionTypeAttr(TypeAttr::get(newType));

      // Update all operations results.
      for (auto result : op->getResults())
        if (auto newType = processType(result.getType()))
          result.setType(newType);

      // Update all block arguments.
      for (auto &region : op->getRegions())
        for (auto &block : region.getBlocks())
          for (auto arg : block.getArguments())
            if (auto newType = processType(arg.getType()))
              arg.setType(newType);
    });

    enumTypeAliases.clear();
  }

  TypeScopeOp typeScope;
  unsigned enumCount;
  DenseMap<Type, Type> enumTypeAliases;
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::createLegalizeAnonEnumsPass() {
  return std::make_unique<LegalizeAnonEnums>();
}
