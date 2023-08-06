//===- HWToSV.h - HW to SystemC pass entry point --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the HWToSV pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HWTOSV_H
#define CIRCT_CONVERSION_HWTOSV_H

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
} // namespace mlir

namespace circt {
namespace hw {
class HWModuleOp;
} // namespace hw
} // namespace circt

namespace circt {
std::unique_ptr<mlir::OperationPass<hw::HWModuleOp>> createLowerHWToSVPass();
} // namespace circt

#endif // CIRCT_CONVERSION_HWTOSV_H
