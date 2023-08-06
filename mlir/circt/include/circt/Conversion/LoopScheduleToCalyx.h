//===- LoopScheduleToCalyx.h - LoopSchedule to Calyx pass entry point -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the LoopScheduleToCalyx pass
// constructor.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LOOPSCHEDULETOCALYX_H
#define CIRCT_CONVERSION_LOOPSCHEDULETOCALYX_H

#include "circt/Dialect/LoopSchedule/LoopScheduleDialect.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace circt {

/// Create a LoopSchedule to Calyx conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> createLoopScheduleToCalyxPass();

} // namespace circt

#endif // CIRCT_CONVERSION_LOOPSCHEDULETOCALYX_H
