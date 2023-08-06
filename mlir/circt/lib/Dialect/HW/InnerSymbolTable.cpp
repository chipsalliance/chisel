//===- InnerSymbolTable.cpp - InnerSymbolTable and InnerRef verification --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements InnerSymbolTable and verification for InnerRef's.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace hw;

namespace circt {
namespace hw {

//===----------------------------------------------------------------------===//
// InnerSymbolTable
//===----------------------------------------------------------------------===//
InnerSymbolTable::InnerSymbolTable(Operation *op) {
  assert(op->hasTrait<OpTrait::InnerSymbolTable>());
  // Save the operation this table is for.
  this->innerSymTblOp = op;

  walkSymbols(op, [&](StringAttr name, const InnerSymTarget &target) {
    auto it = symbolTable.try_emplace(name, target);
    (void)it;
    assert(it.second && "repeated symbol found");
  });
}

FailureOr<InnerSymbolTable> InnerSymbolTable::get(Operation *op) {
  assert(op);
  if (!op->hasTrait<OpTrait::InnerSymbolTable>())
    return op->emitError("expected operation to have InnerSymbolTable trait");

  TableTy table;
  auto result = walkSymbols(
      op, [&](StringAttr name, const InnerSymTarget &target) -> LogicalResult {
        auto it = table.try_emplace(name, target);
        if (it.second)
          return success();
        auto existing = it.first->second;
        return target.getOp()
            ->emitError()
            .append("redefinition of inner symbol named '", name.strref(), "'")
            .attachNote(existing.getOp()->getLoc())
            .append("see existing inner symbol definition here");
      });
  if (failed(result))
    return failure();
  return InnerSymbolTable(op, std::move(table));
}

LogicalResult InnerSymbolTable::walkSymbols(Operation *op,
                                            InnerSymCallbackFn callback) {
  auto walkSym = [&](StringAttr name, const InnerSymTarget &target) {
    assert(name && !name.getValue().empty());
    return callback(name, target);
  };

  auto walkSyms = [&](hw::InnerSymAttr symAttr,
                      const InnerSymTarget &baseTarget) -> LogicalResult {
    assert(baseTarget.getField() == 0);
    for (auto symProp : symAttr) {
      if (failed(walkSym(symProp.getName(),
                         InnerSymTarget::getTargetForSubfield(
                             baseTarget, symProp.getFieldID()))))
        return failure();
    }
    return success();
  };

  // Walk the operation and add InnerSymbolTarget's to the table.
  return success(
      !op->walk<mlir::WalkOrder::PreOrder>([&](Operation *curOp) -> WalkResult {
           if (auto symOp = dyn_cast<InnerSymbolOpInterface>(curOp))
             if (auto symAttr = symOp.getInnerSymAttr())
               if (failed(walkSyms(symAttr, InnerSymTarget(symOp))))
                 return WalkResult::interrupt();

           // Check for ports
           // TODO: Add fields per port, once they work that way (use addSyms)
           if (auto mod = dyn_cast<HWModuleLike>(curOp)) {
             for (size_t i = 0, e = mod.getNumPorts(); i < e; ++i) {
               if (auto symAttr = mod.getPortSymbolAttr(i))
                 if (failed(walkSyms(symAttr, InnerSymTarget(i, curOp))))
                   return WalkResult::interrupt();
             }
           }
           return WalkResult::advance();
         }).wasInterrupted());
}

/// Look up a symbol with the specified name, returning empty InnerSymTarget if
/// no such name exists. Names never include the @ on them.
InnerSymTarget InnerSymbolTable::lookup(StringRef name) const {
  return lookup(StringAttr::get(innerSymTblOp->getContext(), name));
}
InnerSymTarget InnerSymbolTable::lookup(StringAttr name) const {
  return symbolTable.lookup(name);
}

/// Look up a symbol with the specified name, returning null if no such
/// name exists or doesn't target just an operation.
Operation *InnerSymbolTable::lookupOp(StringRef name) const {
  return lookupOp(StringAttr::get(innerSymTblOp->getContext(), name));
}
Operation *InnerSymbolTable::lookupOp(StringAttr name) const {
  auto result = lookup(name);
  if (result.isOpOnly())
    return result.getOp();
  return nullptr;
}

/// Get InnerSymbol for an operation.
StringAttr InnerSymbolTable::getInnerSymbol(Operation *op) {
  if (auto innerSymOp = dyn_cast<InnerSymbolOpInterface>(op))
    return innerSymOp.getInnerNameAttr();
  return {};
}

/// Get InnerSymbol for a target.  Be robust to queries on unexpected
/// operations to avoid users needing to know the details.
StringAttr InnerSymbolTable::getInnerSymbol(const InnerSymTarget &target) {
  // Assert on misuse, but try to handle queries otherwise.
  assert(target);

  // Obtain the base InnerSymAttr for the specified target.
  auto getBase = [](auto &target) -> hw::InnerSymAttr {
    if (target.isPort()) {
      // TODO: This needs to be made to work with HWModuleLike
      if (auto mod = dyn_cast<HWModuleLike>(target.getOp())) {
        assert(target.getPort() < mod.getNumPorts());
        return mod.getPortSymbolAttr(target.getPort());
      }
    } else {
      // InnerSymbols only supported if op implements the interface.
      if (auto symOp = dyn_cast<InnerSymbolOpInterface>(target.getOp()))
        return symOp.getInnerSymAttr();
    }
    return {};
  };

  if (auto base = getBase(target))
    return base.getSymIfExists(target.getField());
  return {};
}

//===----------------------------------------------------------------------===//
// InnerSymbolTableCollection
//===----------------------------------------------------------------------===//

InnerSymbolTable &
InnerSymbolTableCollection::getInnerSymbolTable(Operation *op) {
  auto it = symbolTables.try_emplace(op, nullptr);
  if (it.second)
    it.first->second = ::std::make_unique<InnerSymbolTable>(op);
  return *it.first->second;
}

LogicalResult
InnerSymbolTableCollection::populateAndVerifyTables(Operation *innerRefNSOp) {
  // Gather top-level operations that have the InnerSymbolTable trait.
  SmallVector<Operation *> innerSymTableOps(llvm::make_filter_range(
      llvm::make_pointer_range(innerRefNSOp->getRegion(0).front()),
      [&](Operation *op) {
        return op->hasTrait<OpTrait::InnerSymbolTable>();
      }));

  // Ensure entries exist for each operation.
  llvm::for_each(innerSymTableOps,
                 [&](auto *op) { symbolTables.try_emplace(op, nullptr); });

  // Construct the tables in parallel (if context allows it).
  return mlir::failableParallelForEach(
      innerRefNSOp->getContext(), innerSymTableOps, [&](auto *op) {
        auto it = symbolTables.find(op);
        assert(it != symbolTables.end());
        if (!it->second) {
          auto result = InnerSymbolTable::get(op);
          if (failed(result))
            return failure();
          it->second = std::make_unique<InnerSymbolTable>(std::move(*result));
          return success();
        }
        return failure();
      });
}

//===----------------------------------------------------------------------===//
// InnerRefNamespace
//===----------------------------------------------------------------------===//

InnerSymTarget InnerRefNamespace::lookup(hw::InnerRefAttr inner) {
  auto *mod = symTable.lookup(inner.getModule());
  if (!mod)
    return {};
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookup(inner.getName());
}

Operation *InnerRefNamespace::lookupOp(hw::InnerRefAttr inner) {
  auto *mod = symTable.lookup(inner.getModule());
  if (!mod)
    return nullptr;
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookupOp(inner.getName());
}

//===----------------------------------------------------------------------===//
// InnerRefNamespace verification
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult verifyInnerRefNamespace(Operation *op) {
  // Construct the symbol tables.
  InnerSymbolTableCollection innerSymTables;
  if (failed(innerSymTables.populateAndVerifyTables(op)))
    return failure();

  SymbolTable symbolTable(op);
  InnerRefNamespace ns{symbolTable, innerSymTables};

  // Conduct parallel walks of the top-level children of this
  // InnerRefNamespace, verifying all InnerRefUserOp's discovered within.
  auto verifySymbolUserFn = [&](Operation *op) -> WalkResult {
    if (auto user = dyn_cast<InnerRefUserOpInterface>(op))
      return WalkResult(user.verifyInnerRefs(ns));
    return WalkResult::advance();
  };
  return mlir::failableParallelForEach(
      op->getContext(), op->getRegion(0).front(), [&](auto &op) {
        return success(!op.walk(verifySymbolUserFn).wasInterrupted());
      });
}

} // namespace detail
} // namespace hw
} // namespace circt
