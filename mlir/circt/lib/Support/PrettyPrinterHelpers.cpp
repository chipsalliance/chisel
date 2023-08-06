//===- PrettyPrinterHelpers.cpp - Pretty printing helpers -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for using PrettyPrinter.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/PrettyPrinterHelpers.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"
#include <array>

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// Convenience builders.
//===----------------------------------------------------------------------===//

void TokenStringSaver::clear() { alloc.Reset(); }

/// Add multiple non-breaking spaces as a single token.
void detail::emitNBSP(unsigned n, llvm::function_ref<void(Token)> add) {
  static const std::array<char, 128> spaces = ([]() constexpr {
    std::array<char, 128> s = {};
    for (auto &c : s)
      c = ' ';
    return s;
  })();

  const auto size = spaces.size();
  if (n <= size) {
    if (n != 0)
      add(StringToken({spaces.data(), n}));
    return;
  }
  while (n) {
    auto chunk = std::min<uint32_t>(n, size);
    add(StringToken({spaces.data(), chunk}));
    n -= chunk;
  }
}

} // end namespace pretty
} // end namespace circt
