//===- Tester.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tester class used in the CIRCT reduce tool.
//
//===----------------------------------------------------------------------===//

#include "circt/Reduce/Tester.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Tester
//===----------------------------------------------------------------------===//

Tester::Tester(StringRef testScript, ArrayRef<std::string> testScriptArgs,
               bool testMustFail)
    : testScript(testScript), testScriptArgs(testScriptArgs),
      testMustFail(testMustFail) {}

std::pair<bool, size_t> Tester::isInteresting(ModuleOp module) const {
  auto test = get(module);
  return std::make_pair(test.isInteresting(), test.getSize());
}

/// Runs the interestingness testing script on a MLIR test case file. Returns
/// true if the interesting behavior is present in the test case or false
/// otherwise.
bool Tester::isInteresting(StringRef testCase) const {
  // Assemble the arguments to the tester. Note that the first one has to be the
  // name of the program.
  SmallVector<StringRef> testerArgs;
  testerArgs.push_back(testScript);
  testerArgs.append(testScriptArgs.begin(), testScriptArgs.end());
  testerArgs.push_back(testCase);

  // Run the tester.
  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(
      testScript, testerArgs, /*Env=*/std::nullopt, /*Redirects=*/std::nullopt,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);
  if (result < 0)
    llvm::report_fatal_error(
        Twine("Error running interestingness test: ") + errMsg, false);

  if (testMustFail)
    return result > 0;

  return result == 0;
}

/// Create a new test case for the given `module`.
TestCase Tester::get(mlir::ModuleOp module) const {
  return TestCase(*this, module);
}

/// Create a new test case for the given file already on disk.
TestCase Tester::get(llvm::Twine filepath) const {
  return TestCase(*this, filepath);
}

//===----------------------------------------------------------------------===//
// Test Case
//===----------------------------------------------------------------------===//

/// Check whether the MLIR module is valid. Actual validation is only
/// performed on the first call; subsequent calls return the cached result.
bool TestCase::isValid() {
  // Assume already-provided test cases on disk are valid.
  if (!module)
    return true;
  if (!valid)
    valid = succeeded(verify(module));
  return *valid;
}

/// Determine the path to the MLIR module on disk. Actual writing to disk is
/// only performed on the first call; subsequent calls return the cached result.
StringRef TestCase::getFilepath() {
  if (!isValid())
    return "";
  ensureFileOnDisk();
  return filepath;
}

/// Determine the size of the MLIR module on disk. Actual writing to disk is
/// only performed on the first call; subsequent calls return the cached result.
size_t TestCase::getSize() {
  if (!isValid())
    return 0;
  ensureFileOnDisk();
  return *size;
}

/// Run the tester on the MLIR module and return whether it is deemed
/// interesting. Actual testing is only performed on the first call; subsequent
/// calls return the cached result.
bool TestCase::isInteresting() {
  if (!isValid())
    return false;
  ensureFileOnDisk();
  if (!interesting)
    interesting = tester.isInteresting(filepath);
  return *interesting;
}

/// Ensure `filepath` and `size` are populated, and that the test case is in a
/// file on disk.
void TestCase::ensureFileOnDisk() {
  // Write the module to a temporary file if no already-prepared file path has
  // been provided to the test.
  if (filepath.empty()) {
    assert(module);

    // Pick a temporary output file path.
    int fd;
    std::error_code ec = llvm::sys::fs::createTemporaryFile(
        "circt-reduce", "mlir", fd, filepath);
    if (ec)
      llvm::report_fatal_error(
          Twine("Error making unique filename: ") + ec.message(), false);

    // Write to the output.
    file = std::make_unique<llvm::ToolOutputFile>(filepath, fd);
    module.print(file->os());
    file->os().close();
    if (file->os().has_error())
      llvm::report_fatal_error(llvm::Twine("Error emitting the IR to file `") +
                                   filepath + "`",
                               false);

    // Update the file size.
    size = file->os().tell();
    return;
  }

  // Otherwise just determine the size of the already-prepared file on disk.
  if (!size) {
    uint64_t fileSize;
    std::error_code ec = llvm::sys::fs::file_size(filepath, fileSize);
    if (ec)
      llvm::report_fatal_error(Twine("Error determining size of file `") +
                                   filepath + "`: " + ec.message(),
                               false);
    size = fileSize;
  }
}
