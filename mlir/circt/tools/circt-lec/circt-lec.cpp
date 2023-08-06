//===- circt-lec.cpp - The circt-lec driver ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file initiliazes the 'circt-lec' tool, which interfaces with a logical
/// engine to allow its user to check whether two input circuit descriptions
/// are equivalent, and when not provides a counterexample as for why.
///
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "circt/Support/Version.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-lec Options");

static cl::opt<std::string> moduleName1(
    "c1",
    cl::desc("Specify a named module for the first circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> moduleName2(
    "c2",
    cl::desc("Specify a named module for the second circuit of the comparison"),
    cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> fileName1(cl::Positional, cl::Required,
                                      cl::desc("<input file>"),
                                      cl::cat(mainCategory));

static cl::opt<std::string> fileName2(cl::Positional, cl::desc("[input file]"),
                                      cl::cat(mainCategory));

static cl::opt<bool>
    verbose("v", cl::init(false),
            cl::desc("Print extensive execution progress information"),
            cl::cat(mainCategory));

// The following options are stored externally for their value to be accessible
// to other components of the tool.
bool statisticsOpt;
static cl::opt<bool, true> statistics(
    "s", cl::location(statisticsOpt), cl::init(false),
    cl::desc("Print statistics about the logical engine's execution"),
    cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

/// This functions initializes the various components of the tool and
/// orchestrates the work to be done. It first parses the input files, then it
/// traverses their IR to export the logical constraints from the given circuit
/// description to an internal circuit representation, lastly, these will be
/// compared and solved for equivalence.
static LogicalResult executeLEC(MLIRContext &context) {
  // Parse the provided input files.
  if (verbose)
    lec::outs() << "Parsing input file\n";
  OwningOpRef<ModuleOp> file1 = parseSourceFile<ModuleOp>(fileName1, &context);
  if (!file1)
    return failure();

  OwningOpRef<ModuleOp> file2;
  if (!fileName2.empty()) {
    if (verbose)
      lec::outs() << "Parsing second input file\n";
    file2 = parseSourceFile<ModuleOp>(fileName2, &context);
    if (!file2)
      return failure();
  } else if (verbose)
    lec::outs() << "Second input file not specified\n";

  // Initiliaze the constraints solver and the circuits to be compared.
  Solver s(&context, statisticsOpt);
  Solver::Circuit *c1 = s.addCircuit(moduleName1);
  Solver::Circuit *c2 = s.addCircuit(moduleName2);

  // Initialize a logic exporter for the first circuit then run it on the
  // top-level module of the first input file.
  if (verbose)
    lec::outs() << "Analyzing the first circuit\n";
  auto exporter = std::make_unique<LogicExporter>(moduleName1, c1);
  ModuleOp m = file1.get();
  if (failed(exporter->run(m)))
    return failure();

  // Repeat the same procedure for the second circuit.
  if (verbose)
    lec::outs() << "Analyzing the second circuit\n";
  auto exporter2 = std::make_unique<LogicExporter>(moduleName2, c2);
  // In case a second input file was not specified, the first input file will
  // be used instead.
  ModuleOp m2 = fileName2.empty() ? m : file2.get();
  if (failed(exporter2->run(m2)))
    return failure();

  // The logical constraints have been exported to their respective circuit
  // representations and can now be solved for equivalence.
  if (verbose)
    lec::outs() << "Solving constraints\n";
  return s.solve();
}

/// The entry point for the `circt-lec` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeLEC` function to do the actual work.
int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  registerMLIRContextCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-lec - logical equivalence checker\n\n"
      "\tThis tool compares two input circuit descriptions to determine whether"
      " they are logically equivalent.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  registry.insert<circt::comb::CombDialect, circt::hw::HWDialect>();
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  if (verbose)
    lec::outs() << "Starting execution\n";
  exit(failed(executeLEC(context)));
}
