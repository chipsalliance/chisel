//===- Path.cpp - Path Utilities --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for file system path handling, supplementing the ones from
// llvm::sys::path.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Path.h"
#include "llvm/Support/Path.h"

using namespace circt;

/// Append a path to an existing path, replacing it if the other path is
/// absolute. This mimicks the behaviour of `foo/bar` and `/foo/bar` being used
/// in a working directory `/home`, resulting in `/home/foo/bar` and `/foo/bar`,
/// respectively.
void circt::appendPossiblyAbsolutePath(llvm::SmallVectorImpl<char> &base,
                                       const llvm::Twine &suffix) {
  if (llvm::sys::path::is_absolute(suffix)) {
    base.clear();
    suffix.toVector(base);
  } else {
    llvm::sys::path::append(base, suffix);
  }
}
