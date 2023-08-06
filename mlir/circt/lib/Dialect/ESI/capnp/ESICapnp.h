//===- ESICapnp.h - ESI Cap'nProto library utilies --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ESI utility code which requires libcapnp and libcapnpc.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
#define CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H

#include "circt/Dialect/ESI/APIUtilities.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Format.h"

#include <memory>

namespace mlir {
class Type;
struct LogicalResult;
class Value;
class OpBuilder;
} // namespace mlir
namespace llvm {
class raw_ostream;
class StringRef;
} // namespace llvm

namespace circt {
namespace esi {
namespace capnp {

/// Emit an ID in capnp format.
inline llvm::raw_ostream &emitCapnpID(llvm::raw_ostream &os, int64_t id) {
  return os << "@" << llvm::format_hex(id, /*width=*/16 + 2);
}

namespace detail {
struct CapnpTypeSchemaImpl;
} // namespace detail

/// Generate and reason about a Cap'nProto schema for a particular MLIR type.
class CapnpTypeSchema : public ESIAPIType {
public:
  CapnpTypeSchema(mlir::Type);

  using ESIAPIType::operator==;

  /// Size in bits of the capnp message.
  size_t size() const;

  /// Write out the schema in its entirety.
  mlir::LogicalResult write(llvm::raw_ostream &os) const;

  /// Build an HW/SV dialect capnp encoder for this type.
  mlir::Value buildEncoder(mlir::OpBuilder &, mlir::Value clk,
                           mlir::Value valid, mlir::Value rawData) const;
  /// Build an HW/SV dialect capnp decoder for this type.
  mlir::Value buildDecoder(mlir::OpBuilder &, mlir::Value clk,
                           mlir::Value valid, mlir::Value capnpData) const;

  /// Write out the name and ID in capnp schema format.
  void writeMetadata(llvm::raw_ostream &os) const;

private:
  /// The implementation of this. Separate to hide the details and avoid having
  /// to include the capnp headers in this header.
  std::shared_ptr<detail::CapnpTypeSchemaImpl> s;

  /// Cache of the decode/encode modules;
  static llvm::SmallDenseMap<Type, hw::HWModuleOp> decImplMods;
  static llvm::SmallDenseMap<Type, hw::HWModuleOp> encImplMods;
};

} // namespace capnp
} // namespace esi
} // namespace circt

#endif // CIRCT_DIALECT_ESI_CAPNP_ESICAPNP_H
