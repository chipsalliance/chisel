//===- ESICleanMetadata.cpp - Clean ESI metadata ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// ESI clean metadata pass.
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"

using namespace circt;
using namespace esi;

namespace {
struct ESICleanMetadataPass
    : public ESICleanMetadataBase<ESICleanMetadataPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ESICleanMetadataPass::runOnOperation() {
  auto mod = getOperation();

  // Delete all service declarations.
  mod.walk([&](ServiceHierarchyMetadataOp op) { op.erase(); });
  // Track declarations which are still used so that the service impl reqs are
  // still valid.
  DenseSet<StringAttr> stillUsed;
  mod.walk([&](ServiceImplementReqOp req) {
    auto sym = req.getServiceSymbol();
    if (sym.has_value())
      stillUsed.insert(StringAttr::get(req.getContext(), *sym));
  });
  mod.walk([&](ServiceDeclOpInterface decl) {
    if (!stillUsed.contains(SymbolTable::getSymbolName(decl)))
      decl.getOperation()->erase();
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
circt::esi::createESICleanMetadataPass() {
  return std::make_unique<ESICleanMetadataPass>();
}
