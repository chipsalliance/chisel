//===- APIUtilities.cpp - ESI general-purpose API utilities ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities and classes applicable to all ESI API generators.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/APIUtilities.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Support/IndentedOstream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

namespace circt {
namespace esi {

/// Returns true if the type is currently supported.
// NOLINTNEXTLINE(misc-no-recursion)
static bool isSupported(Type type, bool outer = false) {
  return llvm::TypeSwitch<::mlir::Type, bool>(type)
      .Case([](IntegerType t) { return t.getWidth() <= 64; })
      .Case([](hw::ArrayType t) { return isSupported(t.getElementType()); })
      .Case([outer](hw::StructType t) {
        // We don't yet support structs containing structs.
        if (!outer)
          return false;
        // A struct is supported if all of its elements are.
        for (auto field : t.getElements()) {
          if (!isSupported(field.type))
            return false;
        }
        return true;
      })
      .Default([](Type) { return false; });
}

bool ESIAPIType::isSupported() const {
  return circt::esi::isSupported(type, true);
}

ESIAPIType::ESIAPIType(Type typeArg) : type(innerType(typeArg)) {
  TypeSwitch<Type>(type)
      .Case([this](IntegerType t) {
        fieldTypes.push_back(
            FieldInfo{StringAttr::get(t.getContext(), "i"), t});
      })
      .Case([this](hw::ArrayType t) {
        fieldTypes.push_back(
            FieldInfo{StringAttr::get(t.getContext(), "l"), t});
      })
      .Case([this](hw::StructType t) {
        fieldTypes.append(t.getElements().begin(), t.getElements().end());
      })
      .Default([](Type) {});
}

bool ESIAPIType::operator==(const ESIAPIType &that) const {
  return type == that.type;
}

/// Write a valid Capnp name for 'type'.
// NOLINTNEXTLINE(misc-no-recursion)
static void emitName(Type type, uint64_t id, llvm::raw_ostream &os) {
  llvm::TypeSwitch<Type>(type)
      .Case([&os](IntegerType intTy) {
        std::string intName;
        llvm::raw_string_ostream(intName) << intTy;
        // Capnp struct names must start with an uppercase character.
        intName[0] = toupper(intName[0]);
        os << intName;
      })
      .Case([&os](hw::ArrayType arrTy) {
        os << "ArrayOf" << arrTy.getSize() << 'x';
        emitName(arrTy.getElementType(), 0, os);
      })
      .Case([&os](NoneType) { os << "None"; })
      .Case([&os, id](hw::StructType t) { os << "Struct" << id; })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
      });
}

/// For now, the name is just the type serialized. This works only because we
/// only support ints.
StringRef ESIAPIType::name() const {
  if (cachedName.empty()) {
    llvm::raw_string_ostream os(cachedName);
    emitName(type, typeID(), os);
    cachedName = os.str();
  }
  return cachedName;
}

// We compute a deterministic hash based on the type. Since llvm::hash_value
// changes from execution to execution, we don't use it.
uint64_t ESIAPIType::typeID() const {
  if (cachedID)
    return *cachedID;

  // Get the MLIR asm type, padded to a multiple of 64 bytes.
  std::string typeName;
  llvm::raw_string_ostream osName(typeName);
  osName << type;
  size_t overhang = osName.tell() % 64;
  if (overhang != 0)
    osName.indent(64 - overhang);
  osName.flush();
  const char *typeNameC = typeName.c_str();

  uint64_t hash = esiApiVersion;
  for (size_t i = 0, e = typeName.length() / 64; i < e; ++i)
    hash =
        llvm::hashing::detail::hash_33to64_bytes(&typeNameC[i * 64], 64, hash);

  // Capnp IDs always have a '1' high bit.
  cachedID = hash | 0x8000000000000000;
  return *cachedID;
}

} // namespace esi
} // namespace circt
