//===- InstanceGraphTest.cpp - FIRRTL type unit tests ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

TEST(InstanceGraphTest, PostOrderTraversal) {
  MLIRContext context;
  context.loadDialect<HWDialect>();

  // Build the following graph:
  // hw.module @Top() {
  //   hw.instance "alligator" @Alligator() -> ()
  //   hw.instance "cat" @Cat() -> ()
  // }
  // hw.module private @Alligator() {
  //   hw.instance "bear" @Bear() -> ()
  // }
  // hw.module private @Bear() {
  //   hw.instance "cat" @Cat() -> ()
  // }
  // hw.module private @Cat() { }

  LocationAttr loc = UnknownLoc::get(&context);
  auto module = ModuleOp::create(loc);
  auto builder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  auto top = builder.create<HWModuleOp>(StringAttr::get(&context, "Top"),
                                        ArrayRef<PortInfo>{});
  auto alligator = builder.create<HWModuleOp>(
      StringAttr::get(&context, "Alligator"), ArrayRef<PortInfo>{});
  auto bear = builder.create<HWModuleOp>(StringAttr::get(&context, "Bear"),
                                         ArrayRef<PortInfo>{});
  auto cat = builder.create<HWModuleOp>(StringAttr::get(&context, "Cat"),
                                        ArrayRef<PortInfo>{});

  builder.setInsertionPointToStart(top.getBodyBlock());
  builder.create<InstanceOp>(alligator, "alligator", ArrayRef<Value>{});
  builder.create<InstanceOp>(cat, "cat", ArrayRef<Value>{});

  builder.setInsertionPointToStart(alligator.getBodyBlock());
  builder.create<InstanceOp>(bear, "bear", ArrayRef<Value>{});

  builder.setInsertionPointToStart(bear.getBodyBlock());
  builder.create<InstanceOp>(cat, "cat", ArrayRef<Value>{});

  InstanceGraph graph(module);

  auto range = llvm::post_order(&graph);

  auto it = range.begin();
  ASSERT_EQ("Cat", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Bear", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Alligator", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ("Top", it->getModule().getModuleName());
  ++it;
  ASSERT_EQ(graph.getTopLevelNode(), *it);
  ++it;
  ASSERT_EQ(range.end(), it);
}

} // namespace
