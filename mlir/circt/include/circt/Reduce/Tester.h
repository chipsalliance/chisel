//===- Tester.h -------------------------------------------------*- C++ -*-===//
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

#ifndef CIRCT_REDUCE_TESTER_H
#define CIRCT_REDUCE_TESTER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace llvm {
class ToolOutputFile;
} // namespace llvm

namespace circt {

class TestCase;

/// A testing environment for reduction attempts.
///
/// This class tracks the program used to check reduction attempts for
/// interestingness and additional arguments to pass to that tool. Use `get()`
/// to obtain a new test case that can be queried for information on an
/// individual MLIR module.
class Tester {
public:
  Tester(llvm::StringRef testScript, llvm::ArrayRef<std::string> testScriptArgs,
         bool testMustFail);

  /// Runs the interestingness testing script on a MLIR test case file. Returns
  /// true if the interesting behavior is present in the test case or false
  /// otherwise.
  std::pair<bool, size_t> isInteresting(mlir::ModuleOp module) const;

  /// Return whether the file in the given path is interesting.
  bool isInteresting(llvm::StringRef testCase) const;

  /// Create a new test case for the given `module`.
  TestCase get(mlir::ModuleOp module) const;

  /// Create a new test case for the given file already on disk.
  TestCase get(llvm::Twine filepath) const;

private:
  /// The binary to execute in order to check a reduction attempt for
  /// interestingness.
  llvm::StringRef testScript;

  /// Additional arguments to pass to `testScript`.
  llvm::ArrayRef<std::string> testScriptArgs;

  /// Consider the testcase to be interesting if it fails rather than on exit
  /// code 0.
  bool testMustFail;
};

/// A single test case to be run by a tester.
///
/// This is a helper object that wraps a `ModuleOp` and can be used to query
/// initial information about the test, such as validity of the module and size
/// on disk, before the test is actually executed.
class TestCase {
public:
  /// Create a test case with an MLIR module that will be written to a temporary
  /// file on disk. The `TestCase` will clean up the temporary file after use.
  TestCase(const Tester &tester, mlir::ModuleOp module)
      : tester(tester), module(module) {}

  /// Create a test case for an already-prepared file on disk. The caller
  /// remains responsible for cleaning up the file on disk.
  TestCase(const Tester &tester, llvm::Twine filepath) : tester(tester) {
    filepath.toVector(this->filepath);
  }

  /// Check whether the MLIR module is valid. Actual validation is only
  /// performed on the first call; subsequent calls return the cached result.
  bool isValid();

  /// Determine the path to the MLIR module on disk. Actual writing to disk is
  /// only performed on the first call; subsequent calls return the cached
  /// result.
  llvm::StringRef getFilepath();

  /// Determine the size of the MLIR module on disk. Actual writing to disk is
  /// only performed on the first call; subsequent calls return the cached
  /// result.
  size_t getSize();

  /// Run the tester on the MLIR module and return whether it is deemed
  /// interesting. Actual testing is only performed on the first call;
  /// subsequent calls return the cached result.
  bool isInteresting();

private:
  friend class Tester;

  /// Ensure `filepath` and `size` are populated, and that the test case is in a
  /// file on disk.
  void ensureFileOnDisk();

  /// The tester that is used to run this test case.
  const Tester &tester;
  /// The module to be tested.
  mlir::ModuleOp module;
  /// The path on disk where the test case is located.
  llvm::SmallString<32> filepath;

  /// In case this test case has created a temporary file on disk, this is the
  /// `ToolOutputFile` that did the writing. Keeping this class around ensures
  /// that the file will be cleaned up properly afterwards. This field remains
  /// null if the user already has provided a filepath in the constructor.
  std::unique_ptr<llvm::ToolOutputFile> file;

  /// Whether the MLIR module validation has run, and its result.
  std::optional<bool> valid;
  /// Whether the size of the test case on disk has already been determined, and
  /// if yes, that size.
  std::optional<size_t> size;
  /// Whether the tester has run on this test case, and its result.
  std::optional<bool> interesting;
};

} // namespace circt

#endif // CIRCT_REDUCE_TESTER_H
