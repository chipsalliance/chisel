//===- SystemCEmissionPatterns.h - SystemC Dialect Emission Patterns ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the systemc dialect for registration.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_SYSTEMCEMISSIONPATTERNS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_SYSTEMCEMISSIONPATTERNS_H

#include "../EmissionPatternSupport.h"

namespace circt {
namespace ExportSystemC {

/// Register SystemC operation emission patterns.
void populateSystemCOpEmitters(OpEmissionPatternSet &patterns,
                               MLIRContext *context);

/// Register SystemC type emission patterns.
void populateSystemCTypeEmitters(TypeEmissionPatternSet &patterns);

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_SYSTEMCEMISSIONPATTERNS_H
