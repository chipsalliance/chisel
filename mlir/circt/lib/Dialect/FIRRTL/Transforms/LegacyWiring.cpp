//===- LegacyWiring- legacy Wiring annotation resolver --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the legacy Wiring annotation resolver.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/LogicalResult.h"

using namespace circt;
using namespace firrtl;

/// Consume SourceAnnotation and SinkAnnotation, storing into state
LogicalResult circt::firrtl::applyWiring(const AnnoPathValue &target,
                                         DictionaryAttr anno,
                                         ApplyState &state) {
  auto clazz = anno.getAs<StringAttr>("class").getValue();
  auto *context = anno.getContext();
  ImplicitLocOpBuilder builder(target.ref.getOp()->getLoc(), context);

  // Convert target to Value
  Value targetValue;
  if (auto portTarget = target.ref.dyn_cast<PortAnnoTarget>()) {
    auto portNum = portTarget.getImpl().getPortNo();
    if (auto module = dyn_cast<FModuleOp>(portTarget.getOp())) {
      if (clazz == wiringSourceAnnoClass) {
        builder.setInsertionPointToStart(module.getBodyBlock());
      } else if (clazz == wiringSinkAnnoClass) {
        builder.setInsertionPointToEnd(module.getBodyBlock());
      }
      targetValue = getValueByFieldID(builder, module.getArgument(portNum),
                                      target.fieldIdx);
    } else if (auto ext = dyn_cast<FExtModuleOp>(portTarget.getOp())) {
      InstanceOp inst;
      if (target.instances.empty()) {
        auto paths = state.instancePathCache.getAbsolutePaths(ext);
        if (paths.size() > 1) {
          mlir::emitError(state.circuit.getLoc())
              << "cannot resolve a unique instance path from the "
                 "external module target "
              << target.ref;
          return failure();
        }
        inst = cast<InstanceOp>(paths[0].back());
      } else {
        inst = cast<InstanceOp>(target.instances.back());
      }
      state.wiringProblemInstRefs.insert(inst);
      builder.setInsertionPointAfter(inst);
      targetValue =
          getValueByFieldID(builder, inst->getResult(portNum), target.fieldIdx);
    } else {
      return mlir::emitError(state.circuit.getLoc())
             << "Annotation has invalid target: " << anno;
    }
  } else if (auto opResult = target.ref.dyn_cast<OpAnnoTarget>()) {
    if (target.isOpOfType<WireOp, RegOp, RegResetOp>()) {
      auto *targetBase = opResult.getOp();
      builder.setInsertionPointAfter(targetBase);
      targetValue =
          getValueByFieldID(builder, targetBase->getResult(0), target.fieldIdx);
    } else {
      return mlir::emitError(state.circuit.getLoc())
             << "Annotation targets non-wireable operation: " << anno;
    }
  } else {
    return mlir::emitError(state.circuit.getLoc())
           << "Annotation has invalid target: " << anno;
  }

  // Get pin field
  auto pin = anno.getAs<StringAttr>("pin");
  if (!pin) {
    return mlir::emitError(state.circuit.getLoc())
           << "Annotation does not have an associated pin name: " << anno;
  }

  // Handle difference between sinks and sources
  if (clazz == wiringSourceAnnoClass) {
    if (state.legacyWiringProblems.find(pin) !=
        state.legacyWiringProblems.end()) {
      // Check if existing problem can be updated
      if (state.legacyWiringProblems[pin].source) {
        return mlir::emitError(state.circuit.getLoc())
               << "More than one " << wiringSourceAnnoClass
               << " defined for pin " << pin;
      }
    }
    state.legacyWiringProblems[pin].source = targetValue;
  } else if (clazz == wiringSinkAnnoClass) {
    state.legacyWiringProblems[pin].sinks.push_back(targetValue);
  }

  return success();
}
