//===- BuiltinEmissionPatterns.h - Builtin Dialect Emission Patterns ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the builtin dialect for registration.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_BUILTINEMISSIONPATTERNS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_BUILTINEMISSIONPATTERNS_H

#include "../EmissionPatternSupport.h"

namespace circt {
namespace ExportSystemC {

/// Register Builtin operation emission patterns.
void populateBuiltinOpEmitters(OpEmissionPatternSet &patterns,
                               MLIRContext *context);

/// Register Builtin type emission patterns.
void populateBuiltinTypeEmitters(TypeEmissionPatternSet &patterns);

/// Register Builtin attribute emission patterns.
void populateBuiltinAttrEmitters(AttrEmissionPatternSet &patterns);

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_BUILTINEMISSIONPATTERNS_H
