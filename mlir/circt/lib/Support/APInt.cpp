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

#include "circt/Support/APInt.h"
#include "llvm/ADT/APSInt.h"

using namespace circt;

APInt circt::sextZeroWidth(APInt value, unsigned width) {
  return value.getBitWidth() ? value.sext(width) : value.zext(width);
}

APSInt circt::extOrTruncZeroWidth(APSInt value, unsigned width) {
  return value.getBitWidth()
             ? value.extOrTrunc(width)
             : APSInt(value.zextOrTrunc(width), value.isUnsigned());
}
