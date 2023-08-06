//===- ValueMapper.cpp - Support for mapping SSA values ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides support for mapping SSA values between two domains.
// Provided a BackedgeBuilder, the ValueMapper supports mappings between
// GraphRegions, creating Backedges in cases of 'get'ing mapped values which are
// yet to be 'set'.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/ValueMapper.h"

using namespace mlir;
using namespace circt;
mlir::Value ValueMapper::get(Value from, TypeTransformer typeTransformer) {
  if (mapping.count(from) == 0) {
    assert(bb && "Trying to 'get' a mapped value without any value set. No "
                 "BackedgeBuilder was provided, so cannot provide any mapped "
                 "SSA value!");
    // Create a backedge which will be resolved at a later time once all
    // operands are created.
    mapping[from] = bb->get(typeTransformer(from.getType()));
  }
  auto operandMapping = mapping[from];
  Value mappedOperand;
  if (auto *v = std::get_if<Value>(&operandMapping))
    mappedOperand = *v;
  else
    mappedOperand = std::get<Backedge>(operandMapping);
  return mappedOperand;
}

llvm::SmallVector<Value> ValueMapper::get(ValueRange from,
                                          TypeTransformer typeTransformer) {
  llvm::SmallVector<Value> to;
  for (auto f : from)
    to.push_back(get(f, typeTransformer));
  return to;
}

void ValueMapper::set(Value from, Value to, bool replace) {
  auto it = mapping.find(from);
  if (it != mapping.end()) {
    if (auto *backedge = std::get_if<Backedge>(&it->second))
      backedge->setValue(to);
    else if (!replace)
      assert(false && "'from' was already mapped to a final value!");
  }
  // Register the new mapping
  mapping[from] = to;
}

void ValueMapper::set(ValueRange from, ValueRange to, bool replace) {
  assert(from.size() == to.size() &&
         "Expected # of 'from' values and # of 'to' values to be identical.");
  for (auto [f, t] : llvm::zip(from, to))
    set(f, t, replace);
}
