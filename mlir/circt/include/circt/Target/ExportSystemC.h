//===- ExportSystemC.h - SystemC Exporter -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the SystemC emitter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_EXPORTSYSTEMC_H
#define CIRCT_TARGET_EXPORTSYSTEMC_H

#include "circt/Support/LLVM.h"

namespace circt {
namespace ExportSystemC {

LogicalResult exportSystemC(ModuleOp module, llvm::raw_ostream &os);

LogicalResult exportSplitSystemC(ModuleOp module, StringRef directory);

void registerExportSystemCTranslation();

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_H
