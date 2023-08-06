//===- SVAttributes.h - Declare SV dialect attributes ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVATTRIBUTES_H
#define CIRCT_DIALECT_SV_SVATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "circt/Dialect/SV/SVEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SV/SVAttributes.h.inc"

namespace circt {
namespace sv {

/// Helper functions to handle SV attributes.
bool hasSVAttributes(mlir::Operation *op);

/// Return all the SV attributes of an operation, or null if there are none. All
/// array elements are `SVAttributeAttr`.
mlir::ArrayAttr getSVAttributes(mlir::Operation *op);

/// Set the SV attributes of an operation. If `attrs` is null or contains no
/// attributes, the attribute is removed from the op.
void setSVAttributes(mlir::Operation *op, mlir::ArrayAttr attrs);

/// Set the SV attributes of an operation. If `attrs` is empty the attribute is
/// removed from the op. The attributes are deduplicated such that only one copy
/// of each attribute is kept on the operation.
void setSVAttributes(mlir::Operation *op,
                     mlir::ArrayRef<SVAttributeAttr> attrs);

/// Check if an op contains a specific SV attribute.
inline bool hasSVAttribute(mlir::Operation *op, SVAttributeAttr attr) {
  return llvm::is_contained(getSVAttributes(op), attr);
}

/// Modify the list of SV attributes of an operation. This function offers a
/// read-modify-write interface where the callback can modify the list of
/// attributes how it sees fit. Returns true if any modifications occurred.
bool modifySVAttributes(
    mlir::Operation *op,
    llvm::function_ref<void(llvm::SmallVectorImpl<SVAttributeAttr> &)>
        modifyCallback);

/// Add a list of SV attributes to an operation. The attributes are deduplicated
/// such that only one copy of each attribute is kept on the operation. Returns
/// the number of attributes that were added.
unsigned addSVAttributes(mlir::Operation *op,
                         llvm::ArrayRef<SVAttributeAttr> attrs);

/// Remove the SV attributes from an operation for which `removeCallback`
/// returns true. Returns the number of attributes actually removed.
unsigned
removeSVAttributes(mlir::Operation *op,
                   llvm::function_ref<bool(SVAttributeAttr)> removeCallback);

/// Remove a list of SV attributes from an operation. Returns the number of
/// attributes actually removed.
unsigned removeSVAttributes(mlir::Operation *op,
                            llvm::ArrayRef<SVAttributeAttr> attrs);

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVATTRIBUTES_H
