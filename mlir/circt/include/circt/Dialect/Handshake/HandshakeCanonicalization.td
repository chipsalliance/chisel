//===- HandshakeCanonicalization.td - Handshake canonicalization -*- tablegen
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The canonicalization patterns for the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKECANONICALIZATION_TD
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKECANONICALIZATION_TD

include "circt/Dialect/Handshake/Handshake.td"

def HasOneOperand : Constraint<CPred<"$_self.size() == 1">, "has one operand">;
def HasOneResult : Constraint<CPred<"$_self.size() == 1">, "has one result">;

def EliminateSimpleMergesPattern : Pat<(MergeOp $arg), (replaceWithValue $arg),
                                       [(HasOneOperand:$arg)]>;

def EliminateSimpleBranchesPattern: Pat<(BranchOp $a), (replaceWithValue $a)>;

def EliminateSimpleForksPattern : Pat<(ForkOp:$op $arg),
                                      (replaceWithValue $arg), [(HasOneResult
                                                               : $op)]>;

def EliminateSunkConstantsPattern
    : Pat<(SinkOp:$sink (ConstantOp $value, $ctrl)),
          (replaceWithValue $sink)>;

def EliminateSunkBuffersPattern
    : Pat<(SinkOp (BufferOp $operand, $size, $bufferType, $initValues)),
          (SinkOp $operand)>;

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKECANONICALIZATION_TD
