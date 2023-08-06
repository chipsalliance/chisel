//===- firtool.cpp - The firtool utility for working with .fir files ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements 'firtool', which composes together a variety of
// libraries in a way that is convenient to work with as a user.
//
//===----------------------------------------------------------------------===//

#include "circt/Firtool/Firtool.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/LoweringOptionsParser.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

/// Allow the user to specify the input file format.  This can be used to
/// override the input, and can be used to specify ambiguous cases like standard
/// input.
enum InputFormatKind { InputUnspecified, InputFIRFile, InputMLIRFile };

static cl::OptionCategory mainCategory("firtool Options");

static cl::opt<InputFormatKind> inputFormat(
    "format", cl::desc("Specify input file format:"),
    cl::values(
        clEnumValN(InputUnspecified, "autodetect", "Autodetect input format"),
        clEnumValN(InputFIRFile, "fir", "Parse as .fir file"),
        clEnumValN(InputMLIRFile, "mlir", "Parse as .mlir or .mlirbc file")),
    cl::init(InputUnspecified), cl::cat(mainCategory));

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::init("-"), cl::cat(mainCategory));

static cl::opt<std::string> outputFilename(
    "o", cl::desc("Output filename, or directory for split output"),
    cl::value_desc("filename"), cl::init("-"), cl::cat(mainCategory));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::list<std::string> includeDirs(
    "include-dir",
    cl::desc("Directory to search in when resolving source references"),
    cl::value_desc("directory"), cl::cat(mainCategory));

static cl::alias includeDirsShort(
    "I", cl::desc("Alias for --include-dir.  Example: -I<directory>"),
    cl::aliasopt(includeDirs), cl::Prefix, cl::NotHidden,
    cl::cat(mainCategory));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false), cl::Hidden, cl::cat(mainCategory));

static cl::opt<bool> disableAnnotationsClassless(
    "disable-annotation-classless",
    cl::desc("Ignore annotations without a class when parsing"),
    cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> disableAnnotationsUnknown(
    "disable-annotation-unknown",
    cl::desc("Ignore unknown annotations when parsing"), cl::init(false),
    cl::cat(mainCategory));

static cl::opt<bool> lowerAnnotationsNoRefTypePorts(
    "lower-annotations-no-ref-type-ports",
    cl::desc("Create real ports instead of ref type ports when resolving "
             "wiring problems inside the LowerAnnotations pass"),
    cl::init(false), cl::Hidden, cl::cat(mainCategory));

using InfoLocHandling = firrtl::FIRParserOptions::InfoLocHandling;
static cl::opt<InfoLocHandling> infoLocHandling(
    cl::desc("Location tracking:"),
    cl::values(
        clEnumValN(InfoLocHandling::IgnoreInfo, "ignore-info-locators",
                   "Ignore the @info locations in the .fir file"),
        clEnumValN(InfoLocHandling::FusedInfo, "fuse-info-locators",
                   "@info locations are fused with .fir locations"),
        clEnumValN(
            InfoLocHandling::PreferInfo, "prefer-info-locators",
            "Use @info locations when present, fallback to .fir locations")),
    cl::init(InfoLocHandling::PreferInfo), cl::cat(mainCategory));

static cl::opt<bool> exportModuleHierarchy(
    "export-module-hierarchy",
    cl::desc("Export module and instance hierarchy as JSON"), cl::init(false),
    cl::cat(mainCategory));

static cl::opt<bool>
    scalarizeTopModule("scalarize-top-module",
                       cl::desc("Scalarize the ports of the top module"),
                       cl::init(true), cl::cat(mainCategory));

static cl::opt<bool>
    scalarizeExtModules("scalarize-ext-modules",
                        cl::desc("Scalarize the ports of any external modules"),
                        cl::init(true), cl::cat(mainCategory));

static firtool::FirtoolOptions firtoolOptions(mainCategory);

enum OutputFormatKind {
  OutputParseOnly,
  OutputIRFir,
  OutputIRHW,
  OutputIRSV,
  OutputIRVerilog,
  OutputVerilog,
  OutputSplitVerilog,
  OutputDisabled
};

static cl::opt<OutputFormatKind> outputFormat(
    cl::desc("Specify output format:"),
    cl::values(
        clEnumValN(OutputParseOnly, "parse-only",
                   "Emit FIR dialect after parsing, verification, and "
                   "annotation lowering"),
        clEnumValN(OutputIRFir, "ir-fir", "Emit FIR dialect after pipeline"),
        clEnumValN(OutputIRHW, "ir-hw", "Emit HW dialect"),
        clEnumValN(OutputIRSV, "ir-sv", "Emit SV dialect"),
        clEnumValN(OutputIRVerilog, "ir-verilog",
                   "Emit IR after Verilog lowering"),
        clEnumValN(OutputVerilog, "verilog", "Emit Verilog"),
        clEnumValN(OutputSplitVerilog, "split-verilog",
                   "Emit Verilog (one file per module; specify "
                   "directory with -o=<dir>)"),
        clEnumValN(OutputDisabled, "disable-output", "Do not output anything")),
    cl::init(OutputVerilog), cl::cat(mainCategory));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

static cl::list<std::string> inputAnnotationFilenames(
    "annotation-file", cl::desc("Optional input annotation file"),
    cl::CommaSeparated, cl::value_desc("filename"), cl::cat(mainCategory));

static cl::list<std::string> inputOMIRFilenames(
    "omir-file", cl::desc("Optional input object model 2.0 file"),
    cl::CommaSeparated, cl::value_desc("filename"), cl::cat(mainCategory));

static cl::opt<std::string>
    mlirOutFile("output-final-mlir",
                cl::desc("Optional file name to output the final MLIR into, in "
                         "addition to the output requested by -o"),
                cl::init(""), cl::value_desc("filename"),
                cl::cat(mainCategory));

static cl::opt<bool>
    emitBytecode("emit-bytecode",
                 cl::desc("Emit bytecode when generating MLIR output"),
                 cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> force("f", cl::desc("Enable binary output on terminals"),
                           cl::init(false), cl::cat(mainCategory));

static cl::opt<bool>
    verbosePassExecutions("verbose-pass-executions",
                          cl::desc("Log executions of toplevel module passes"),
                          cl::init(false), cl::cat(mainCategory));

static cl::opt<bool> stripFirDebugInfo(
    "strip-fir-debug-info",
    cl::desc("Disable source fir locator information in output Verilog"),
    cl::init(true), cl::cat(mainCategory));

static cl::opt<bool> stripDebugInfo(
    "strip-debug-info",
    cl::desc("Disable source locator information in output Verilog"),
    cl::init(false), cl::cat(mainCategory));

static LoweringOptionsOption loweringOptions(mainCategory);

/// Check output stream before writing bytecode to it.
/// Warn and return true if output is known to be displayed.
static bool checkBytecodeOutputToConsole(raw_ostream &os) {
  if (os.is_displayed()) {
    errs() << "WARNING: You're attempting to print out a bytecode file.\n"
              "This is inadvisable as it may cause display problems. If\n"
              "you REALLY want to taste MLIR bytecode first-hand, you\n"
              "can force output with the `-f' option.\n\n";
    return true;
  }
  return false;
}

/// Print the operation to the specified stream, emitting bytecode when
/// requested and politely avoiding dumping to terminal unless forced.
static LogicalResult printOp(Operation *op, raw_ostream &os) {
  if (emitBytecode && (force || !checkBytecodeOutputToConsole(os)))
    return writeBytecodeToFile(op, os,
                               mlir::BytecodeWriterConfig(getCirctVersion()));
  op->print(os);
  return success();
}

/// Process a single buffer of the input.
static LogicalResult processBuffer(
    MLIRContext &context, TimingScope &ts, llvm::SourceMgr &sourceMgr,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  // Add the annotation file if one was explicitly specified.
  unsigned numAnnotationFiles = 0;
  for (const auto &inputAnnotationFilename : inputAnnotationFilenames) {
    std::string annotationFilenameDetermined;
    if (!sourceMgr.AddIncludeFile(inputAnnotationFilename, llvm::SMLoc(),
                                  annotationFilenameDetermined)) {
      llvm::errs() << "cannot open input annotation file '"
                   << inputAnnotationFilename
                   << "': No such file or directory\n";
      return failure();
    }
    ++numAnnotationFiles;
  }

  for (const auto &file : inputOMIRFilenames) {
    std::string filename;
    if (!sourceMgr.AddIncludeFile(file, llvm::SMLoc(), filename)) {
      llvm::errs() << "cannot open input annotation file '" << file
                   << "': No such file or directory\n";
      return failure();
    }
  }

  // Parse the input.
  mlir::OwningOpRef<mlir::ModuleOp> module;

  llvm::sys::TimePoint<> parseStartTime;
  if (verbosePassExecutions) {
    llvm::errs() << "[firtool] Running "
                 << (inputFormat == InputFIRFile ? "fir" : "mlir")
                 << " parser\n";
    parseStartTime = llvm::sys::TimePoint<>::clock::now();
  }

  if (inputFormat == InputFIRFile) {
    auto parserTimer = ts.nest("FIR Parser");
    firrtl::FIRParserOptions options;
    options.infoLocatorHandling = infoLocHandling;
    options.numAnnotationFiles = numAnnotationFiles;
    options.scalarizeTopModule = scalarizeTopModule;
    options.scalarizeExtModules = scalarizeExtModules;
    module = importFIRFile(sourceMgr, &context, parserTimer, options);
  } else {
    auto parserTimer = ts.nest("MLIR Parser");
    assert(inputFormat == InputMLIRFile);
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  }
  if (!module)
    return failure();

  if (verbosePassExecutions) {
    auto elapsed = std::chrono::duration<double>(
                       llvm::sys::TimePoint<>::clock::now() - parseStartTime) /
                   std::chrono::seconds(1);
    llvm::errs() << "[firtool] -- Done in " << llvm::format("%.3f", elapsed)
                 << " sec\n";
  }

  // Apply any pass manager command line options.
  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (verbosePassExecutions)
    pm.addInstrumentation(
        std::make_unique<
            VerbosePassInstrumentation<firrtl::CircuitOp, mlir::ModuleOp>>(
            "firtool"));
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  // Legalize away "open" aggregates to hw-only versions.
  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerOpenAggsPass());

  pm.nest<firrtl::CircuitOp>().addPass(firrtl::createLowerFIRRTLAnnotationsPass(
      disableAnnotationsUnknown, disableAnnotationsClassless,
      lowerAnnotationsNoRefTypePorts));

  // If the user asked for --parse-only, stop after running LowerAnnotations.
  if (outputFormat == OutputParseOnly) {
    if (failed(pm.run(module.get())))
      return failure();
    auto outputTimer = ts.nest("Print .mlir output");
    return printOp(*module, (*outputFile)->os());
  }

  if (failed(firtool::populateCHIRRTLToLowFIRRTL(pm, firtoolOptions, *module,
                                                 inputFilename)))
    return failure();

  // Lower if we are going to verilog or if lowering was specifically requested.
  if (outputFormat != OutputIRFir) {
    if (failed(firtool::populateLowFIRRTLToHW(pm, firtoolOptions)))
      return failure();
    if (outputFormat != OutputIRHW)
      if (failed(firtool::populateHWToSV(pm, firtoolOptions)))
        return failure();
  }

  // Load the emitter options from the command line. Command line options if
  // specified will override any module options.
  if (loweringOptions.getNumOccurrences())
    loweringOptions.setAsAttribute(module.get());

  if (failed(pm.run(module.get())))
    return failure();

  // Add passes specific to Verilog emission if we're going there.
  if (outputFormat == OutputVerilog || outputFormat == OutputSplitVerilog ||
      outputFormat == OutputIRVerilog) {
    PassManager exportPm(&context);
    exportPm.enableTiming(ts);
    if (failed(applyPassManagerCLOptions(exportPm)))
      return failure();
    if (verbosePassExecutions)
      exportPm.addInstrumentation(
          std::make_unique<
              VerbosePassInstrumentation<firrtl::CircuitOp, mlir::ModuleOp>>(
              "firtool"));
    // Legalize unsupported operations within the modules.
    exportPm.nest<hw::HWModuleOp>().addPass(sv::createHWLegalizeModulesPass());

    // Tidy up the IR to improve verilog emission quality.
    if (!firtoolOptions.disableOptimization)
      exportPm.nest<hw::HWModuleOp>().addPass(sv::createPrettifyVerilogPass());

    if (stripFirDebugInfo)
      exportPm.addPass(
          circt::createStripDebugInfoWithPredPass([](mlir::Location loc) {
            if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
              return fileLoc.getFilename().getValue().endswith(".fir");
            return false;
          }));

    if (stripDebugInfo)
      exportPm.addPass(circt::createStripDebugInfoWithPredPass(
          [](mlir::Location loc) { return true; }));

    // Emit module and testbench hierarchy JSON files.
    if (exportModuleHierarchy)
      exportPm.addPass(sv::createHWExportModuleHierarchyPass(outputFilename));

    // Emit a single file or multiple files depending on the output format.
    switch (outputFormat) {
    default:
      llvm_unreachable("can't reach this");
    case OutputVerilog:
      exportPm.addPass(createExportVerilogPass((*outputFile)->os()));
      break;
    case OutputSplitVerilog:
      exportPm.addPass(createExportSplitVerilogPass(outputFilename));
      break;
    case OutputIRVerilog:
      // Run the ExportVerilog pass to get its lowering, but discard the output.
      exportPm.addPass(createExportVerilogPass(llvm::nulls()));
      break;
    }

    // Run final IR mutations to clean it up after ExportVerilog and before
    // emitting the final MLIR.
    if (!mlirOutFile.empty())
      exportPm.addPass(firrtl::createFinalizeIRPass());

    if (failed(exportPm.run(module.get())))
      return failure();
  }

  if (outputFormat == OutputIRFir || outputFormat == OutputIRHW ||
      outputFormat == OutputIRSV || outputFormat == OutputIRVerilog) {
    auto outputTimer = ts.nest("Print .mlir output");
    if (failed(printOp(*module, (*outputFile)->os())))
      return failure();
  }

  // If requested, print the final MLIR into mlirOutFile.
  if (!mlirOutFile.empty()) {
    std::string mlirOutError;
    auto mlirFile = openOutputFile(mlirOutFile, &mlirOutError);
    if (!mlirFile) {
      llvm::errs() << mlirOutError;
      return failure();
    }

    if (failed(printOp(*module, mlirFile->os())))
      return failure();
    mlirFile->keep();
  }

  // We intentionally "leak" the Module into the MLIRContext instead of
  // deallocating it.  There is no need to deallocate it right before process
  // exit.
  (void)module.release();
  return success();
}

class FileLineColLocsAsNotesDiagnosticHandler : public ScopedDiagnosticHandler {
public:
  FileLineColLocsAsNotesDiagnosticHandler(MLIRContext *ctxt)
      : ScopedDiagnosticHandler(ctxt) {
    setHandler([](Diagnostic &d) {
      SmallPtrSet<Location, 8> locs;
      // Recursively scan for FileLineColLoc locations.
      d.getLocation()->walk([&](Location loc) {
        if (isa<FileLineColLoc>(loc))
          locs.insert(loc);
        return WalkResult::advance();
      });

      // Drop top-level location the diagnostic is reported on.
      locs.erase(d.getLocation());
      // As well as the location the SourceMgrDiagnosticHandler will use.
      if (auto reportLoc = d.getLocation()->findInstanceOf<FileLineColLoc>())
        locs.erase(reportLoc);

      // Attach additional locations as notes on the diagnostic.
      for (auto l : locs)
        d.attachNote(l) << "additional location here";
      return failure();
    });
  }
};

/// Process a single split of the input. This allocates a source manager and
/// creates a regular or verifying diagnostic handler, depending on whether the
/// user set the verifyDiagnostics option.
static LogicalResult processInputSplit(
    MLIRContext &context, TimingScope &ts,
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
  sourceMgr.setIncludeDirs(includeDirs);
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr,
                                                &context /*, shouldShow */);
    FileLineColLocsAsNotesDiagnosticHandler addLocs(&context);
    return processBuffer(context, ts, sourceMgr, outputFile);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);
  context.printOpOnDiagnostic(false);
  (void)processBuffer(context, ts, sourceMgr, outputFile);
  return sourceMgrHandler.verify();
}

/// Process the entire input provided by the user, splitting it up if the
/// corresponding option was specified.
static LogicalResult
processInput(MLIRContext &context, TimingScope &ts,
             std::unique_ptr<llvm::MemoryBuffer> input,
             std::optional<std::unique_ptr<llvm::ToolOutputFile>> &outputFile) {
  if (!splitInputFile)
    return processInputSplit(context, ts, std::move(input), outputFile);

  // Emit an error if the user provides a separate annotation file alongside
  // split input. This is technically not a problem, but the user likely
  // expects the annotation file to be split as well, which is not the case.
  // To prevent any frustration, we detect this constellation and emit an
  // error here. The user can provide annotations for each split using the
  // inline JSON syntax in FIRRTL.
  if (!inputAnnotationFilenames.empty()) {
    llvm::errs() << "annotation file cannot be used with split input: "
                    "use inline JSON syntax on FIRRTL `circuit` to specify "
                    "per-split annotations\n";
    return failure();
  }

  return splitAndProcessBuffer(
      std::move(input),
      [&](std::unique_ptr<MemoryBuffer> buffer, raw_ostream &) {
        return processInputSplit(context, ts, std::move(buffer), outputFile);
      },
      llvm::outs());
}

/// This implements the top-level logic for the firtool command, invoked once
/// command line options are parsed and LLVM/MLIR are all set up and ready to
/// go.
static LogicalResult executeFirtool(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  // Set up the input file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Figure out the input format if unspecified.
  if (inputFormat == InputUnspecified) {
    if (StringRef(inputFilename).endswith(".fir"))
      inputFormat = InputFIRFile;
    else if (StringRef(inputFilename).endswith(".mlir") ||
             StringRef(inputFilename).endswith(".mlirbc") ||
             mlir::isBytecode(*input))
      inputFormat = InputMLIRFile;
    else {
      llvm::errs() << "unknown input format: "
                      "specify with -format=fir or -format=mlir\n";
      return failure();
    }
  }

  // Create the output directory or output file depending on our mode.
  std::optional<std::unique_ptr<llvm::ToolOutputFile>> outputFile;
  if (outputFormat != OutputSplitVerilog) {
    // Create an output file.
    outputFile.emplace(openOutputFile(outputFilename, &errorMessage));
    if (!(*outputFile)) {
      llvm::errs() << errorMessage << "\n";
      return failure();
    }
  } else {
    // Create an output directory.
    if (outputFilename.isDefaultOption() || outputFilename == "-") {
      llvm::errs() << "missing output directory: specify with -o=<dir>\n";
      return failure();
    }
    auto error = llvm::sys::fs::create_directories(outputFilename);
    if (error) {
      llvm::errs() << "cannot create output directory '" << outputFilename
                   << "': " << error.message() << "\n";
      return failure();
    }
  }

  // Register our dialects.
  context.loadDialect<chirrtl::CHIRRTLDialect, firrtl::FIRRTLDialect,
                      hw::HWDialect, comb::CombDialect, seq::SeqDialect,
                      om::OMDialect, sv::SVDialect, verif::VerifDialect,
                      ltl::LTLDialect>();

  // Process the input.
  if (failed(processInput(context, ts, std::move(input), outputFile)))
    return failure();

  // If the result succeeded and we're emitting a file, close it.
  if (outputFile.has_value())
    (*outputFile)->keep();

  return success();
}

/// Main driver for firtool command.  This sets up LLVM and MLIR, and parses
/// command line options before passing off to 'executeFirtool'.  This is set up
/// so we can `exit(0)` at the end of the program to avoid teardown of the
/// MLIRContext and modules inside of it (reducing compile time).
int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  setBugReportMsg(circtBugReportMsg);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register passes before parsing command-line options, so that they are
  // available for use with options like `--mlir-print-ir-before`.
  {
    // MLIR transforms:
    // Don't use registerTransformsPasses, pulls in too much.
    registerCSEPass();
    registerCanonicalizerPass();
    registerStripDebugInfoPass();
    registerSymbolDCEPass();

    // Dialect passes:
    firrtl::registerPasses();
    sv::registerPasses();

    // Export passes:
    registerExportChiselInterfacePass();
    registerExportSplitChiselInterfacePass();
    registerExportSplitVerilogPass();
    registerExportVerilogPass();
  }

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](raw_ostream &os) { os << getCirctVersion() << '\n'; });
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "MLIR-based FIRRTL compiler\n");

  MLIRContext context;

  // Do the guts of the firtool process.
  auto result = executeFirtool(context);

  // Use "exit" instead of return'ing to signal completion.  This avoids
  // invoking the MLIRContext destructor, which spends a bunch of time
  // deallocating memory etc which process exit will do for us.
  exit(failed(result));
}
