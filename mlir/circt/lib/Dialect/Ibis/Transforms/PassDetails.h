//===- PassDetails.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_IBIS_TRANSFORMS_PASSDETAILS_H
#define DIALECT_IBIS_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/DC/DCDialect.h"
#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/SV/SVDialect.h"

#include "mlir/Pass/Pass.h"

namespace circt {
namespace sandpiper {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Ibis/Ibis.h.inc"

} // namespace sandpiper
} // namespace circt

#endif // DIALECT_IBIS_TRANSFORMS_PASSDETAILS_H
