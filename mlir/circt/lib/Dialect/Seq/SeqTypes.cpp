//===- SeqTypes.cpp - Seq types code defs ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation logic for Seq data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace seq;

//===----------------------------------------------------------------------===//
/// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Seq/SeqTypes.cpp.inc"

void SeqDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Seq/SeqTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// HLMemType
//===----------------------------------------------------------------------===//

HLMemType HLMemType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                               Type elementType) const {
  return HLMemType::get(elementType.getContext(), shape.value_or(getShape()),
                        elementType);
}

llvm::SmallVector<Type> HLMemType::getAddressTypes() const {
  auto *ctx = getContext();
  llvm::SmallVector<Type> addressTypes;
  for (auto dim : getShape())
    addressTypes.push_back(IntegerType::get(ctx, llvm::Log2_64_Ceil(dim)));
  return addressTypes;
}

Type HLMemType::parse(mlir::AsmParser &odsParser) {
  llvm::SmallVector<int64_t> shape;
  Type elementType;
  if (odsParser.parseLess() ||
      odsParser.parseDimensionList(shape, /*allowDynamic=*/false,
                                   /*withTrailingX=*/true) ||
      odsParser.parseType(elementType) || odsParser.parseGreater())
    return {};

  return HLMemType::get(odsParser.getContext(), shape, elementType);
}

void HLMemType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << '<';
  for (auto dim : getShape())
    odsPrinter << dim << 'x';
  odsPrinter << getElementType();
  odsPrinter << '>';
}

LogicalResult
HLMemType::verify(llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<int64_t> shape, Type elementType) {
  if (shape.empty())
    return emitError() << "shape must have at least one dimension.";
  return success();
}
