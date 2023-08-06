//===- APIUtilities.h - ESI general-purpose API utilities -------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_ESI_APIUTILITIES_H
#define CIRCT_DIALECT_ESI_APIUTILITIES_H

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/MapVector.h"

#include <memory>

namespace circt {
namespace esi {

/// Every time we implement a breaking change in the schema generation,
/// increment this number. It is a seed for all the schema hashes.
constexpr uint64_t esiApiVersion = 1;

// Base type for all Cosim-implementing type emitters.
class ESIAPIType {
public:
  using FieldInfo = hw::StructType::FieldInfo;

  ESIAPIType(mlir::Type);
  virtual ~ESIAPIType() = default;
  bool operator==(const ESIAPIType &) const;

  /// Get the type back.
  mlir::Type getType() const { return type; }

  /// Returns true if the type is currently supported.
  virtual bool isSupported() const;

  llvm::ArrayRef<FieldInfo> getFields() const { return fieldTypes; }

  // API-safe name for this type which should work with most languages.
  StringRef name() const;

  // Capnproto-safe type id for this type.
  uint64_t typeID() const;

protected:
  /// Cosim requires that everything be contained in a struct. ESI doesn't so
  /// we wrap non-struct types in a struct.
  llvm::SmallVector<FieldInfo> fieldTypes;

  mlir::Type type;
  mutable std::string cachedName;
  mutable std::optional<uint64_t> cachedID;
};

} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_APIUTILITIES_H
