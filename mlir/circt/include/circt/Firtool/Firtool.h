//===- Firtool.h - Definitions for the firtool pipeline setup ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This library parses options for firtool and sets up its pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_FIRTOOL_FIRTOOL_H
#define CIRCT_FIRTOOL_FIRTOOL_H

#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace circt {
namespace firtool {
struct FirtoolOptions {
  llvm::cl::OptionCategory &category;

  llvm::cl::opt<circt::firrtl::PreserveAggregate::PreserveMode>
      preserveAggregate{
          "preserve-aggregate", llvm::cl::desc("Specify input file format:"),
          llvm::cl::values(
              clEnumValN(circt::firrtl::PreserveAggregate::None, "none",
                         "Preserve no aggregate"),
              clEnumValN(circt::firrtl::PreserveAggregate::OneDimVec, "1d-vec",
                         "Preserve only 1d vectors of ground type"),
              clEnumValN(circt::firrtl::PreserveAggregate::Vec, "vec",
                         "Preserve only vectors"),
              clEnumValN(circt::firrtl::PreserveAggregate::All, "all",
                         "Preserve vectors and bundles")),
          llvm::cl::init(circt::firrtl::PreserveAggregate::None),
          llvm::cl::cat(category)};

  llvm::cl::opt<firrtl::PreserveValues::PreserveMode> preserveMode{
      "preserve-values",
      llvm::cl::desc("Specify the values which can be optimized away"),
      llvm::cl::values(clEnumValN(firrtl::PreserveValues::None, "none",
                                  "Preserve no values"),
                       clEnumValN(firrtl::PreserveValues::Named, "named",
                                  "Preserve values with meaningful names"),
                       clEnumValN(firrtl::PreserveValues::All, "all",
                                  "Preserve all values")),
      llvm::cl::init(firrtl::PreserveValues::None), llvm::cl::cat(category)};

  // Build mode options.
  enum BuildMode { BuildModeDebug, BuildModeRelease };
  llvm::cl::opt<BuildMode> buildMode{
      "O", llvm::cl::desc("Controls how much optimization should be performed"),
      llvm::cl::values(clEnumValN(BuildModeDebug, "debug",
                                  "Compile with only necessary optimizations"),
                       clEnumValN(BuildModeRelease, "release",
                                  "Compile with optimizations")),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool> disableOptimization{
      "disable-opt", llvm::cl::desc("Disable optimizations"),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool> exportChiselInterface{
      "export-chisel-interface",
      llvm::cl::desc("Generate a Scala Chisel interface to the top level "
                     "module of the firrtl circuit"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<std::string> chiselInterfaceOutDirectory{
      "chisel-interface-out-dir",
      llvm::cl::desc(
          "The output directory for generated Chisel interface files"),
      llvm::cl::init(""), llvm::cl::cat(category)};

  llvm::cl::opt<bool> vbToBV{
      "vb-to-bv",
      llvm::cl::desc("Transform vectors of bundles to bundles of vectors"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> dedup{
      "dedup", llvm::cl::desc("Deduplicate structurally identical modules"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> grandCentralInstantiateCompanionOnly{
      "grand-central-instantiate-companion",
      llvm::cl::desc("Run Grand Central in a mode where the companion module "
                     "is instantiated and not bound in and the interface is "
                     "dropped.  This is intended for situations where there is "
                     "useful assertion logic inside the companion, but you "
                     "don't care about the actual interface."),
      llvm::cl::init(false), llvm::cl::Hidden, llvm::cl::cat(category)};

  llvm::cl::opt<bool> disableAggressiveMergeConnections{
      "disable-aggressive-merge-connections",
      llvm::cl::desc(
          "Disable aggressive merge connections (i.e. merge all field-level "
          "connections into bulk connections)"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> emitOMIR{
      "emit-omir", llvm::cl::desc("Emit OMIR annotations to a JSON file"),
      llvm::cl::init(true), llvm::cl::cat(category)};

  llvm::cl::opt<std::string> omirOutFile{
      "output-omir", llvm::cl::desc("File name for the output omir"),
      llvm::cl::init(""), llvm::cl::cat(category)};

  llvm::cl::opt<bool> lowerMemories{
      "lower-memories",
      llvm::cl::desc("Lower memories to have memories with masks as an "
                     "array with one memory per ground type"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<std::string> blackBoxRootPath{
      "blackbox-path",
      llvm::cl::desc(
          "Optional path to use as the root of black box annotations"),
      llvm::cl::value_desc("path"), llvm::cl::init(""),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool> replSeqMem{
      "repl-seq-mem",
      llvm::cl::desc("Replace the seq mem for macro replacement and emit "
                     "relevant metadata"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<std::string> replSeqMemFile{
      "repl-seq-mem-file", llvm::cl::desc("File name for seq mem metadata"),
      llvm::cl::init(""), llvm::cl::cat(category)};

  llvm::cl::opt<bool> extractTestCode{
      "extract-test-code", llvm::cl::desc("Run the extract test code pass"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> ignoreReadEnableMem{
      "ignore-read-enable-mem",
      llvm::cl::desc("Ignore the read enable signal, instead of "
                     "assigning X on read disable"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  enum class RandomKind { None, Mem, Reg, All };

  llvm::cl::opt<RandomKind> disableRandom{
      llvm::cl::desc(
          "Disable random initialization code (may break semantics!)"),
      llvm::cl::values(
          clEnumValN(RandomKind::Mem, "disable-mem-randomization",
                     "Disable emission of memory randomization code"),
          clEnumValN(RandomKind::Reg, "disable-reg-randomization",
                     "Disable emission of register randomization code"),
          clEnumValN(RandomKind::All, "disable-all-randomization",
                     "Disable emission of all randomization code")),
      llvm::cl::init(RandomKind::None), llvm::cl::cat(category)};

  llvm::cl::opt<std::string> outputAnnotationFilename{
      "output-annotation-file",
      llvm::cl::desc("Optional output annotation file"),
      llvm::cl::CommaSeparated, llvm::cl::value_desc("filename"),
      llvm::cl::cat(category)};

  llvm::cl::opt<bool> enableAnnotationWarning{
      "warn-on-unprocessed-annotations",
      llvm::cl::desc(
          "Warn about annotations that were not removed by lower-to-hw"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> addMuxPragmas{
      "add-mux-pragmas",
      llvm::cl::desc("Annotate mux pragmas for memory array access"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> emitChiselAssertsAsSVA{
      "emit-chisel-asserts-as-sva",
      llvm::cl::desc("Convert all chisel asserts into SVA"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> emitSeparateAlwaysBlocks{
      "emit-separate-always-blocks",
      llvm::cl::desc(
          "Prevent always blocks from being merged and emit constructs into "
          "separate always blocks whenever possible"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> etcDisableInstanceExtraction{
      "etc-disable-instance-extraction",
      llvm::cl::desc("Disable extracting instances only that feed test code"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> etcDisableRegisterExtraction{
      "etc-disable-register-extraction",
      llvm::cl::desc("Disable extracting registers that only feed test code"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> etcDisableModuleInlining{
      "etc-disable-module-inlining",
      llvm::cl::desc("Disable inlining modules that only feed test code"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> addVivadoRAMAddressConflictSynthesisBugWorkaround{
      "add-vivado-ram-address-conflict-synthesis-bug-workaround",
      llvm::cl::desc(
          "Add a vivado specific SV attribute (* ram_style = "
          "\"distributed\" *) to unpacked array registers as a workaronud "
          "for a vivado synthesis bug that incorrectly modifies "
          "address conflict behavivor of combinational memories"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  //===----------------------------------------------------------------------===
  // External Clock Gate Options
  //===----------------------------------------------------------------------===

  seq::ExternalizeClockGateOptions clockGateOpts;

  llvm::cl::opt<std::string, true> ckgModuleName{
      "ckg-name", llvm::cl::desc("Clock gate module name"),
      llvm::cl::location(clockGateOpts.moduleName),
      llvm::cl::init("EICG_wrapper"), llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgInputName{
      "ckg-input", llvm::cl::desc("Clock gate input port name"),
      llvm::cl::location(clockGateOpts.inputName), llvm::cl::init("in"),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgOutputName{
      "ckg-output", llvm::cl::desc("Clock gate output port name"),
      llvm::cl::location(clockGateOpts.outputName), llvm::cl::init("out"),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgEnableName{
      "ckg-enable", llvm::cl::desc("Clock gate enable port name"),
      llvm::cl::location(clockGateOpts.enableName), llvm::cl::init("en"),
      llvm::cl::cat(category)};

  llvm::cl::opt<std::string, true> ckgTestEnableName{
      "ckg-test-enable",
      llvm::cl::desc("Clock gate test enable port name (optional)"),
      llvm::cl::location(clockGateOpts.testEnableName),
      llvm::cl::init("test_en"), llvm::cl::cat(category)};

  bool isRandomEnabled(RandomKind kind) const {
    return disableRandom != RandomKind::All && disableRandom != kind;
  }

  firrtl::PreserveValues::PreserveMode getPreserveMode() const {
    if (!buildMode.getNumOccurrences())
      return preserveMode;
    switch (buildMode) {
    case BuildModeDebug:
      return firrtl::PreserveValues::Named;
    case BuildModeRelease:
      return firrtl::PreserveValues::None;
    }
    llvm_unreachable("unknown build mode");
  }

  FirtoolOptions(llvm::cl::OptionCategory &category) : category(category) {}
};

LogicalResult populateCHIRRTLToLowFIRRTL(mlir::PassManager &pm,
                                         const FirtoolOptions &opt,
                                         ModuleOp module,
                                         StringRef inputFilename);

LogicalResult populateLowFIRRTLToHW(mlir::PassManager &pm,
                                    const FirtoolOptions &opt);

LogicalResult populateHWToSV(mlir::PassManager &pm, const FirtoolOptions &opt);

} // namespace firtool
} // namespace circt

#endif // CIRCT_FIRTOOL_FIRTOOL_H
