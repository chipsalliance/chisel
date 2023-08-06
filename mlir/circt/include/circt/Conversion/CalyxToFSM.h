//===- CalyxToFSM.h - Calyx to FSM conversion pass --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which lowers a Calyx control schedule to an FSM
// representation.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_CALYXTOFSM_CALYXTOFSM_H
#define CIRCT_CONVERSION_CALYXTOFSM_CALYXTOFSM_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

namespace calyxToFSM {
// Entry and exit state names of the Calyx control program in the FSM.
static constexpr std::string_view sEntryStateName = "fsm_entry";
static constexpr std::string_view sExitStateName = "fsm_exit";
static constexpr std::string_view sGroupDoneInputs =
    "calyx.fsm_group_done_inputs";
static constexpr std::string_view sGroupGoOutputs =
    "calyx.fsm_group_go_outputs";
static constexpr std::string_view sSSAInputIndices = "calyx.fsm_ssa_inputs";
static constexpr std::string_view sFSMTopLevelGoIndex =
    "calyx.fsm_top_level_go";
static constexpr std::string_view sFSMTopLevelDoneIndex =
    "calyx.fsm_top_level_done";

} // namespace calyxToFSM

std::unique_ptr<mlir::Pass> createCalyxToFSMPass();
std::unique_ptr<mlir::Pass> createMaterializeCalyxToFSMPass();
std::unique_ptr<mlir::Pass> createRemoveGroupsFromFSMPass();

} // namespace circt

#endif // CIRCT_CONVERSION_CALYXTOFSM_CALYXTOFSM_H
