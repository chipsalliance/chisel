//===- CalyxHelpers.h - Calyx helper methods --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various helper methods for building Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXHELPERS_H
#define CIRCT_DIALECT_CALYX_CALYXHELPERS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"

#include <memory>

namespace circt {
namespace calyx {

/// Creates a RegisterOp, with input and output port bit widths defined by
/// `width`.
calyx::RegisterOp createRegister(Location loc, OpBuilder &builder,
                                 ComponentOp component, size_t width,
                                 Twine prefix);

/// A helper function to create constants in the HW dialect.
hw::ConstantOp createConstant(Location loc, OpBuilder &builder,
                              ComponentOp component, size_t width,
                              size_t value);

// Returns whether this operation is a leaf node in the Calyx control.
// TODO(github.com/llvm/circt/issues/1679): Add Invoke.
bool isControlLeafNode(Operation *op);

// Creates a DictionaryAttr containing a unit attribute 'name'. Used for
// defining mandatory port attributes for calyx::ComponentOp's.
DictionaryAttr getMandatoryPortAttr(MLIRContext *ctx, StringRef name);

// Adds the mandatory Calyx component I/O ports (->[clk, reset, go], [done]->)
// to ports.
void addMandatoryComponentPorts(PatternRewriter &rewriter,
                                SmallVectorImpl<calyx::PortInfo> &ports);

// Returns the bit width for the given dimension. This will always be greater
// than zero. See: https://github.com/llvm/circt/issues/2660
unsigned handleZeroWidth(int64_t dim);

/// Updates the guard of each assignment within a group with `op`.
template <typename Op>
static void updateGroupAssignmentGuards(OpBuilder &builder, GroupOp &group,
                                        Op &op) {
  group.walk([&](AssignOp assign) {
    if (assign.getGuard())
      // If the assignment is guarded already, take the bitwise & of the current
      // guard and the group's go signal.
      assign->setOperand(2, builder.create<comb::AndOp>(
                                group.getLoc(), assign.getGuard(), op, false));
    else
      // Otherwise, just insert it as the guard.
      assign->insertOperands(2, {op});
  });
}

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXHELPERS_H
