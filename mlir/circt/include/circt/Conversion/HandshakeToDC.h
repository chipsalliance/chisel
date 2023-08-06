//===- HandshakeToDC.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Handshake dialect to
// CIRCT RTL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HANDSHAKETODC_H
#define CIRCT_CONVERSION_HANDSHAKETODC_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

std::unique_ptr<mlir::Pass> createHandshakeToDCPass();

namespace handshake {

// Converts 't' into a valid HW type. This is strictly used for converting
// 'index' types into a fixed-width type.
Type toValidType(Type t);

} // namespace handshake
} // namespace circt

#endif // CIRCT_CONVERSION_HANDSHAKETODC_H
