//===- EmitCEmissionPatterns.h - EmitC Dialect Emission Patterns ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This exposes the emission patterns of the emitc dialect for registration.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_EMITCEMISSIONPATTERNS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_EMITCEMISSIONPATTERNS_H

#include "../EmissionPatternSupport.h"

namespace circt {
namespace ExportSystemC {

/// Register EmitC operation emission patterns.
void populateEmitCOpEmitters(OpEmissionPatternSet &patterns,
                             MLIRContext *context);

/// Register EmitC type emission patterns.
void populateEmitCTypeEmitters(TypeEmissionPatternSet &patterns);

/// Register EmitC attribute emission patterns.
void populateEmitCAttrEmitters(AttrEmissionPatternSet &patterns);

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_PATTERNS_EMITCEMISSIONPATTERNS_H
