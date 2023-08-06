//===- InstanceGraphBase.cpp - Instance Graph -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/InstanceGraphBase.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"

using namespace circt;
using namespace hw;

void InstanceRecord::erase() {
  // Update the prev node to point to the next node.
  if (prevUse)
    prevUse->nextUse = nextUse;
  else
    target->firstUse = nextUse;
  // Update the next node to point to the prev node.
  if (nextUse)
    nextUse->prevUse = prevUse;
  getParent()->instances.erase(this);
}

InstanceRecord *InstanceGraphNode::addInstance(HWInstanceLike instance,
                                               InstanceGraphNode *target) {
  auto *instanceRecord = new InstanceRecord(this, instance, target);
  target->recordUse(instanceRecord);
  instances.push_back(instanceRecord);
  return instanceRecord;
}

void InstanceGraphNode::recordUse(InstanceRecord *record) {
  record->nextUse = firstUse;
  if (firstUse)
    firstUse->prevUse = record;
  firstUse = record;
}

InstanceGraphNode *InstanceGraphBase::getOrAddNode(StringAttr name) {
  // Try to insert an InstanceGraphNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto *&node = nodeMap[name];
  if (!node) {
    node = new InstanceGraphNode();
    nodes.push_back(node);
  }
  return node;
}

InstanceGraphBase::InstanceGraphBase(Operation *parent) : parent(parent) {
  assert(parent->hasTrait<mlir::OpTrait::SingleBlock>() &&
         "top-level operation must have a single block");
  SmallVector<std::pair<HWModuleLike, SmallVector<HWInstanceLike>>>
      moduleToInstances;
  // First accumulate modules inside the parent op.
  for (auto module : parent->getRegion(0).front().getOps<hw::HWModuleLike>())
    moduleToInstances.push_back({module, {}});

  // Populate instances in the module parallelly.
  mlir::parallelFor(parent->getContext(), 0, moduleToInstances.size(),
                    [&](size_t idx) {
                      auto module = moduleToInstances[idx].first;
                      auto &instances = moduleToInstances[idx].second;
                      // Find all instance operations in the module body.
                      module.walk([&](HWInstanceLike instanceOp) {
                        instances.push_back(instanceOp);
                      });
                    });

  // Construct an instance graph sequentially.
  for (auto &[module, instances] : moduleToInstances) {
    auto name = module.getModuleNameAttr();
    auto *currentNode = getOrAddNode(name);
    currentNode->module = module;
    for (auto instanceOp : instances) {
      // Add an edge to indicate that this module instantiates the target.
      auto *targetNode = getOrAddNode(instanceOp.getReferencedModuleNameAttr());
      currentNode->addInstance(instanceOp, targetNode);
    }
  }
}

InstanceGraphNode *InstanceGraphBase::addModule(HWModuleLike module) {
  assert(!nodeMap.count(module.getModuleNameAttr()) && "module already added");
  auto *node = new InstanceGraphNode();
  node->module = module;
  nodeMap[module.getModuleNameAttr()] = node;
  nodes.push_back(node);
  return node;
}

void InstanceGraphBase::erase(InstanceGraphNode *node) {
  assert(node->noUses() &&
         "all instances of this module must have been erased.");
  // Erase all instances inside this module.
  for (auto *instance : llvm::make_early_inc_range(*node))
    instance->erase();
  nodeMap.erase(node->getModule().getModuleNameAttr());
  nodes.erase(node);
}

InstanceGraphNode *InstanceGraphBase::lookup(StringAttr name) {
  auto it = nodeMap.find(name);
  assert(it != nodeMap.end() && "Module not in InstanceGraph!");
  return it->second;
}

InstanceGraphNode *InstanceGraphBase::lookup(HWModuleLike op) {
  return lookup(cast<HWModuleLike>(op).getModuleNameAttr());
}

HWModuleLike InstanceGraphBase::getReferencedModule(HWInstanceLike op) {
  return lookup(op.getReferencedModuleNameAttr())->getModule();
}

InstanceGraphBase::~InstanceGraphBase() {}

void InstanceGraphBase::replaceInstance(HWInstanceLike inst,
                                        HWInstanceLike newInst) {
  assert(inst.getReferencedModuleName() == newInst.getReferencedModuleName() &&
         "Both instances must be targeting the same module");

  // Find the instance record of this instance.
  auto *node = lookup(inst.getReferencedModuleNameAttr());
  auto it = llvm::find_if(node->uses(), [&](InstanceRecord *record) {
    return record->getInstance() == inst;
  });
  assert(it != node->usesEnd() && "Instance of module not recorded in graph");

  // We can just replace the instance op in the InstanceRecord without updating
  // any instance lists.
  (*it)->instance = newInst;
}

bool InstanceGraphBase::isAncestor(HWModuleLike child, HWModuleLike parent) {
  DenseSet<InstanceGraphNode *> seen;
  SmallVector<InstanceGraphNode *> worklist;
  auto *cn = lookup(child);
  worklist.push_back(cn);
  seen.insert(cn);
  while (!worklist.empty()) {
    auto *node = worklist.back();
    worklist.pop_back();
    if (node->getModule() == parent)
      return true;
    for (auto *use : node->uses()) {
      auto *mod = use->getParent();
      if (!seen.count(mod)) {
        seen.insert(mod);
        worklist.push_back(mod);
      }
    }
  }
  return false;
}

FailureOr<llvm::ArrayRef<InstanceGraphNode *>>
InstanceGraphBase::getInferredTopLevelNodes() {
  if (!inferredTopLevelNodes.empty())
    return {inferredTopLevelNodes};

  /// Topologically sort the instance graph.
  llvm::SetVector<InstanceGraphNode *> visited, marked;
  llvm::SetVector<InstanceGraphNode *> candidateTopLevels(this->begin(),
                                                          this->end());
  SmallVector<InstanceGraphNode *> cycleTrace;

  // Recursion function; returns true if a cycle was detected.
  std::function<bool(InstanceGraphNode *, SmallVector<InstanceGraphNode *>)>
      cycleUtil =
          [&](InstanceGraphNode *node, SmallVector<InstanceGraphNode *> trace) {
            if (visited.contains(node))
              return false;
            trace.push_back(node);
            if (marked.contains(node)) {
              // Cycle detected.
              cycleTrace = trace;
              return true;
            }
            marked.insert(node);
            for (auto use : *node) {
              InstanceGraphNode *targetModule = use->getTarget();
              candidateTopLevels.remove(targetModule);
              if (cycleUtil(targetModule, trace))
                return true; // Cycle detected.
            }
            marked.remove(node);
            visited.insert(node);
            return false;
          };

  bool cyclic = false;
  for (auto moduleIt : *this) {
    if (visited.contains(moduleIt))
      continue;

    cyclic |= cycleUtil(moduleIt, {});
    if (cyclic)
      break;
  }

  if (cyclic) {
    auto err = getParent()->emitOpError();
    err << "cannot deduce top level module - cycle "
           "detected in instance graph (";
    llvm::interleave(
        cycleTrace, err,
        [&](auto node) { err << node->getModule().getModuleName(); }, "->");
    err << ").";
    return err;
  }
  assert(!candidateTopLevels.empty() &&
         "if non-cyclic, there should be at least 1 candidate top level");

  inferredTopLevelNodes = llvm::SmallVector<InstanceGraphNode *>(
      candidateTopLevels.begin(), candidateTopLevels.end());
  return {inferredTopLevelNodes};
}

ArrayRef<InstancePath> InstancePathCache::getAbsolutePaths(HWModuleLike op) {
  InstanceGraphNode *node = instanceGraph[op];

  // If we have reached the circuit root, we're done.
  if (node == instanceGraph.getTopLevelNode()) {
    static InstancePath empty{};
    return empty; // array with single empty path
  }

  // Fast path: hit the cache.
  auto cached = absolutePathsCache.find(op);
  if (cached != absolutePathsCache.end())
    return cached->second;

  // For each instance, collect the instance paths to its parent and append the
  // instance itself to each.
  SmallVector<InstancePath, 8> extendedPaths;
  for (auto *inst : node->uses()) {
    if (auto module = inst->getParent()->getModule()) {
      auto instPaths = getAbsolutePaths(module);
      extendedPaths.reserve(instPaths.size());
      for (auto path : instPaths) {
        extendedPaths.push_back(
            appendInstance(path, cast<HWInstanceLike>(*inst->getInstance())));
      }
    }
  }

  // Move the list of paths into the bump allocator for later quick retrieval.
  ArrayRef<InstancePath> pathList;
  if (!extendedPaths.empty()) {
    auto *paths = allocator.Allocate<InstancePath>(extendedPaths.size());
    std::copy(extendedPaths.begin(), extendedPaths.end(), paths);
    pathList = ArrayRef<InstancePath>(paths, extendedPaths.size());
  }
  absolutePathsCache.insert({op, pathList});
  return pathList;
}

InstancePath InstancePathCache::appendInstance(InstancePath path,
                                               HWInstanceLike inst) {
  size_t n = path.size() + 1;
  auto *newPath = allocator.Allocate<HWInstanceLike>(n);
  std::copy(path.begin(), path.end(), newPath);
  newPath[path.size()] = inst;
  return InstancePath(newPath, n);
}

void InstancePathCache::replaceInstance(HWInstanceLike oldOp,
                                        HWInstanceLike newOp) {

  instanceGraph.replaceInstance(oldOp, newOp);

  // Iterate over all the paths, and search for the old HWInstanceLike. If
  // found, then replace it with the new HWInstanceLike, and create a new copy
  // of the paths and update the cache.
  auto instanceExists = [&](const ArrayRef<InstancePath> &paths) -> bool {
    return llvm::any_of(
        paths, [&](InstancePath p) { return llvm::is_contained(p, oldOp); });
  };

  for (auto &iter : absolutePathsCache) {
    if (!instanceExists(iter.getSecond()))
      continue;
    SmallVector<InstancePath, 8> updatedPaths;
    for (auto path : iter.getSecond()) {
      const auto *iter = llvm::find(path, oldOp);
      if (iter == path.end()) {
        // path does not contain the oldOp, just copy it as is.
        updatedPaths.push_back(path);
        continue;
      }
      auto *newPath = allocator.Allocate<HWInstanceLike>(path.size());
      llvm::copy(path, newPath);
      newPath[iter - path.begin()] = newOp;
      updatedPaths.push_back(InstancePath(newPath, path.size()));
    }
    // Move the list of paths into the bump allocator for later quick
    // retrieval.
    auto *paths = allocator.Allocate<InstancePath>(updatedPaths.size());
    llvm::copy(updatedPaths, paths);
    iter.getSecond() = ArrayRef<InstancePath>(paths, updatedPaths.size());
  }
}
