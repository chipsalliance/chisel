//===- HWEmissionPatterns.h - HW Dialect Emission Patterns ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the HW dialect for registration.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_HWEMISSIONPATTERNS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_HWEMISSIONPATTERNS_H

#include "../EmissionPatternSupport.h"

namespace circt {
namespace ExportSystemC {
void populateHWEmitters(OpEmissionPatternSet &patterns, MLIRContext *context);
} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_HWEMISSIONPATTERNS_H
