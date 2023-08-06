//===- APInt.h - CIRCT Lowering Options -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for working around limitations of upstream LLVM APInts.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_APINT_H
#define CIRCT_SUPPORT_APINT_H

#include "circt/Support/LLVM.h"

namespace circt {

/// A safe version of APInt::sext that will NOT assert on zero-width
/// signed APSInts.  Instead of asserting, this will zero extend.
APInt sextZeroWidth(APInt value, unsigned width);

/// A safe version of APSInt::extOrTrunc that will NOT assert on zero-width
/// signed APSInts.  Instead of asserting, this will zero extend.
APSInt extOrTruncZeroWidth(APSInt value, unsigned width);

} // namespace circt

#endif // CIRCT_SUPPORT_APINT_H
