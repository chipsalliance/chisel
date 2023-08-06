//===- SeqOpInterfaces.cpp - Implement the Seq op interfaces --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Seq operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace llvm;
using namespace circt::seq;

#include "circt/Dialect/Seq/SeqOpInterfaces.cpp.inc"
