//===- EmissionPatternSupport.h - Emission Pattern forward declarations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Forward declarations and shorthands for various emission pattern classes
// such that emission pattern implementations of dialects don't have to
// #include all the template classes.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPATTERNSUPPORT_H
#define CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPATTERNSUPPORT_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace ExportSystemC {

// Forward declarations.
template <typename Ty>
class EmissionPatternSet;
template <typename PatternTy, typename KeyTy>
class FrozenEmissionPatternSet;
struct OpEmissionPatternBase;
struct TypeEmissionPatternBase;
struct AttrEmissionPatternBase;

using OpEmissionPatternSet = EmissionPatternSet<OpEmissionPatternBase>;
using TypeEmissionPatternSet = EmissionPatternSet<TypeEmissionPatternBase>;
using AttrEmissionPatternSet = EmissionPatternSet<AttrEmissionPatternBase>;

using FrozenOpEmissionPatternSet =
    FrozenEmissionPatternSet<OpEmissionPatternBase, OperationName>;
using FrozenTypeEmissionPatternSet =
    FrozenEmissionPatternSet<TypeEmissionPatternBase, TypeID>;
using FrozenAttrEmissionPatternSet =
    FrozenEmissionPatternSet<AttrEmissionPatternBase, TypeID>;

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_EMISSIONPATTERNSUPPORT_H
