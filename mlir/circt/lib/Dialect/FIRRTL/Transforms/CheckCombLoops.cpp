//===- CheckCombLoops.cpp - FIRRTL check combinational cycles ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL combinational cycles detection pass. The
// algorithm handles aggregates and sub-index/field/access ops.
// 1. Traverse each module in the Instance Graph bottom up.
// 2. Preprocess step: Gather all the Value which serve as the root for the
//    DFS traversal. The input arguments and wire ops and Instance results
//    are the roots.
//    Then populate the map for Value to all the FieldRefs it can refer to,
//    and another map of FieldRef to all the Values that refer to it.
//    (A single Value can refer to multiple FieldRefs, if the Value is the
//    result of a SubAccess op. Multiple values can refer to the same
//    FieldRef, since multiple SubIndex/Field ops with the same fieldIndex
//    can exist in the IR). We also maintain an aliasingValuesMap that maps
//    each Value to the set of Values that can refer to the same FieldRef.
// 3. Start from DFS traversal from the root. Push the root to the DFS stack.
// 4. Pop a Value from the DFS stack, add all the Values that alias with it
//    to the Visiting set. Add all the unvisited children of the Values in the
//    alias set to the DFS stack.
// 5. If any child is already present in the Visiting set, then a cycle is
//    found.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallSet.h"
#include <variant>

#define DEBUG_TYPE "check-comb-loops"

using namespace circt;
using namespace firrtl;

using SetOfFieldRefs = DenseSet<FieldRef>;

/// A value is in VisitingSet if its subtree is still being traversed. That is,
/// all its children have not yet been visited. If any Value is visited while
/// its still in the `VisitingSet`, that implies a back edge and a cycle.
struct VisitingSet {
private:
  /// The stack is maintained to keep track of the cycle, if one is found. This
  /// is required for an iterative DFS traversal, its implicitly recorded for a
  /// recursive version of this algorithm. Each entry in the stack is a list of
  /// aliasing Values, which were visited at the same time.
  SmallVector<SmallVector<Value, 2>> visitingStack;
  /// This map of the Visiting values, is for faster query, to check if a Value
  /// is in VisitingSet. It also records the corresponding index into the
  /// visitingStack, for faster pop until the Value.
  DenseMap<Value, unsigned> valToStackMap;

public:
  void appendEmpty() { visitingStack.push_back({}); }
  void appendToEnd(SmallVector<Value> &values) {
    auto stackSize = visitingStack.size() - 1;
    visitingStack.back().append(values.begin(), values.end());
    // Record the stack location where this Value is pushed.
    llvm::for_each(values, [&](Value v) { valToStackMap[v] = stackSize; });
  }
  bool contains(Value v) {
    return valToStackMap.find(v) != valToStackMap.end();
  }
  // Pop all the Values which were visited after v. Then invoke f (if present)
  // on a popped value for each index.
  void popUntilVal(Value v,
                   const llvm::function_ref<void(Value poppedVal)> f = {}) {
    auto valPos = valToStackMap[v];
    while (visitingStack.size() != valPos) {
      auto poppedVals = visitingStack.pop_back_val();
      Value poppedVal;
      llvm::for_each(poppedVals, [&](Value pv) {
        if (!poppedVal)
          poppedVal = pv;
        valToStackMap.erase(pv);
      });
      if (f && poppedVal)
        f(poppedVal);
    }
  }
};

class DiscoverLoops {

public:
  DiscoverLoops(FModuleOp module, InstanceGraph &instanceGraph,
                DenseMap<FieldRef, SetOfFieldRefs> &portPaths)
      : module(module), instanceGraph(instanceGraph), portPaths(portPaths) {}

  LogicalResult processModule() {
    LLVM_DEBUG(llvm::dbgs() << "\n processing module :" << module.getName());
    SmallVector<Value> worklist;
    // Traverse over ports and ops, to populate the worklist and get the
    // FieldRef corresponding to every Value. Also process the InstanceOps and
    // get the paths that exist between the ports of the referenced module.
    preprocess(worklist);

    llvm::DenseSet<Value> visited;
    VisitingSet visiting;
    SmallVector<Value> dfsStack;
    SmallVector<FieldRef> inputArgFields;
    // Record all the children of Value being visited.
    SmallVector<Value, 8> children;
    // If this is an input port field, then record it. This is used to
    // discover paths from input to output ports. Only the last input port
    // that is visited on the DFS traversal is recorded.
    SmallVector<FieldRef, 2> inputArgFieldsTemp;
    SmallVector<Value> aliasingValues;

    // worklist is the list of roots, to begin the traversal from.
    for (auto root : worklist) {
      dfsStack = {root};
      inputArgFields.clear();
      LLVM_DEBUG(llvm::dbgs() << "\n Starting traversal from root :"
                              << getFieldName(FieldRef(root, 0)).first);
      if (auto inArg = dyn_cast<BlockArgument>(root)) {
        if (module.getPortDirection(inArg.getArgNumber()) == Direction::In)
          // This is required, such that paths to output port can be discovered.
          // If there is an overlapping path from two input ports to an output
          // port, then the already visited nodes must be re-visited to discover
          // the comb paths to the output port.
          visited.clear();
      }
      while (!dfsStack.empty()) {
        auto dfsVal = dfsStack.back();
        if (!visiting.contains(dfsVal)) {
          unsigned dfsSize = dfsStack.size();

          LLVM_DEBUG(llvm::dbgs() << "\n Stack pop :"
                                  << getFieldName(FieldRef(dfsVal, 0)).first
                                  << "," << dfsVal;);

          // Visiting set will contain all the values which alias with the
          // dfsVal, this is required to detect back edges to aliasing Values.
          // That is fieldRefs that can refer to the same memory location.
          visiting.appendEmpty();
          children.clear();
          inputArgFieldsTemp.clear();
          // All the Values that refer to the same FieldRef are added to the
          // aliasingValues.
          aliasingValues = {dfsVal};
          auto aToVIter = aliasingValuesMap.find(dfsVal);
          if (aToVIter != aliasingValuesMap.end()) {
            aliasingValues.append(aToVIter->getSecond().begin(),
                                  aToVIter->getSecond().end());
          }
          // If `dfsVal` is a subfield, then get all the FieldRefs that it
          // refers to and then get all the values that alias with it.
          forallRefersTo(dfsVal, [&](FieldRef ref) {
            // If this subfield refers to instance/mem results(input port), then
            // add the output port FieldRefs that exist in the referenced module
            // comb paths to the children.
            handlePorts(ref, children);
            // Get all the values that refer to this FieldRef, and add them to
            // the aliasing values.
            if (auto arg = dyn_cast<BlockArgument>(ref.getValue()))
              if (module.getPortDirection(arg.getArgNumber()) == Direction::In)
                inputArgFieldsTemp.push_back(ref);

            return success();
          });
          if (!inputArgFieldsTemp.empty())
            inputArgFields = std::move(inputArgFieldsTemp);

          visiting.appendToEnd(aliasingValues);
          visited.insert(aliasingValues.begin(), aliasingValues.end());
          // Add the Value to `children`, to which a path exists from `dfsVal`.
          for (auto dfsFromVal : aliasingValues) {

            for (auto &use : dfsFromVal.getUses()) {
              auto childVal =
                  TypeSwitch<Operation *, Value>(use.getOwner())
                      // Registers stop walk for comb loops.
                      .Case<RegOp, RegResetOp>([](auto _) { return Value(); })
                      // For non-register declarations, look at data result.
                      .Case<Forceable>([](auto op) { return op.getDataRaw(); })
                      // Handle connect ops specially.
                      .Case<FConnectLike>([&](FConnectLike connect) -> Value {
                        if (use.getOperandNumber() == 1) {
                          auto dst = connect.getDest();
                          if (handleConnects(dst, inputArgFields).succeeded())
                            return dst;
                        }
                        return {};
                      })
                      // For everything else (e.g., expressions), if has single
                      // result use that.
                      .Default([](auto op) -> Value {
                        if (op->getNumResults() == 1)
                          return op->getResult(0);
                        return {};
                      });
              if (childVal && type_isa<FIRRTLType>(childVal.getType()))
                children.push_back(childVal);
            }
          }
          for (auto childVal : children) {
            // This childVal can be ignored, if
            // It is a Register or a subfield of a register.
            if (!visited.contains(childVal))
              dfsStack.push_back(childVal);
            // If the childVal is a sub, then check if it aliases with any of
            // the predecessors (the visiting set).
            if (visiting.contains(childVal)) {
              // Comb Cycle Detected !!
              reportLoopFound(childVal, visiting);
              return failure();
            }
          }
          // child nodes added, continue the DFS
          if (dfsSize != dfsStack.size())
            continue;
        }
        // FieldRef is an SCC root node, pop the visiting stack to remove the
        // nodes that are no longer active predecessors, that is their sub-tree
        // is already explored. All the Values reachable from `dfsVal` have been
        // explored, remove it and its children from the visiting stack.
        visiting.popUntilVal(dfsVal);

        auto popped = dfsStack.pop_back_val();
        (void)popped;
        LLVM_DEBUG({
          llvm::dbgs() << "\n dfs popped :"
                       << getFieldName(FieldRef(popped, 0)).first;
          dump();
        });
      }
    }

    return success();
  }

  // Preprocess the module ops to get the
  // 1. roots for DFS traversal,
  // 2. FieldRef corresponding to each Value.
  void preprocess(SmallVector<Value> &worklist) {
    // All the input ports are added to the worklist.
    for (BlockArgument arg : module.getArguments()) {
      auto argType = type_cast<FIRRTLType>(arg.getType());
      if (type_isa<RefType>(argType))
        continue;
      if (module.getPortDirection(arg.getArgNumber()) == Direction::In)
        worklist.push_back(arg);
      if (!argType.isGround())
        setValRefsTo(arg, FieldRef(arg, 0));
    }
    DenseSet<Value> memPorts;

    for (auto &op : module.getOps()) {
      TypeSwitch<Operation *>(&op)
          // Wire is added to the worklist
          .Case<WireOp>([&](WireOp wire) {
            worklist.push_back(wire.getResult());
            auto ty = type_dyn_cast<FIRRTLBaseType>(wire.getResult().getType());
            if (ty && !ty.isGround())
              setValRefsTo(wire.getResult(), FieldRef(wire.getResult(), 0));
          })
          // All sub elements are added to the worklist.
          .Case<SubfieldOp>([&](SubfieldOp sub) {
            auto res = sub.getResult();
            bool isValid = false;
            auto fieldIndex = sub.getAccessedField().getFieldID();
            if (memPorts.contains(sub.getInput())) {
              auto memPort = sub.getInput();
              BundleType type = memPort.getType();
              auto enableFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::en);
              auto dataFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::data);
              auto addressFieldId =
                  type.getFieldID((unsigned)ReadPortSubfield::addr);
              if (fieldIndex == enableFieldId || fieldIndex == dataFieldId ||
                  fieldIndex == addressFieldId) {
                setValRefsTo(memPort, FieldRef(memPort, 0));
              } else
                return;
            }
            SmallVector<FieldRef, 4> fields;
            forallRefersTo(
                sub.getInput(),
                [&](FieldRef subBase) {
                  isValid = true;
                  fields.push_back(subBase.getSubField(fieldIndex));
                  return success();
                },
                false);
            if (isValid) {
              for (auto f : fields)
                setValRefsTo(res, f);
            }
          })
          .Case<SubindexOp>([&](SubindexOp sub) {
            auto res = sub.getResult();
            bool isValid = false;
            auto index = sub.getAccessedField().getFieldID();
            SmallVector<FieldRef, 4> fields;
            forallRefersTo(
                sub.getInput(),
                [&](FieldRef subBase) {
                  isValid = true;
                  fields.push_back(subBase.getSubField(index));
                  return success();
                },
                false);
            if (isValid) {
              for (auto f : fields)
                setValRefsTo(res, f);
            }
          })
          .Case<SubaccessOp>([&](SubaccessOp sub) {
            FVectorType vecType = sub.getInput().getType();
            auto res = sub.getResult();
            bool isValid = false;
            SmallVector<FieldRef, 4> fields;
            forallRefersTo(
                sub.getInput(),
                [&](FieldRef subBase) {
                  isValid = true;
                  // The result of a subaccess can refer to multiple storage
                  // locations corresponding to all the possible indices.
                  for (size_t index = 0; index < vecType.getNumElements();
                       ++index)
                    fields.push_back(subBase.getSubField(
                        1 + index * (vecType.getElementType().getMaxFieldID() +
                                     1)));
                  return success();
                },
                false);
            if (isValid) {
              for (auto f : fields)
                setValRefsTo(res, f);
            }
          })
          .Case<InstanceOp>(
              [&](InstanceOp ins) { handleInstanceOp(ins, worklist); })
          .Case<MemOp>([&](MemOp mem) {
            if (!(mem.getReadLatency() == 0)) {
              return;
            }
            for (auto memPort : mem.getResults()) {
              if (!type_isa<FIRRTLBaseType>(memPort.getType()))
                continue;
              memPorts.insert(memPort);
            }
          })
          .Default([&](auto) {});
    }
  }

  void handleInstanceOp(InstanceOp ins, SmallVector<Value> &worklist) {
    for (auto port : ins.getResults()) {
      if (auto type = type_dyn_cast<FIRRTLBaseType>(port.getType())) {
        worklist.push_back(port);
        if (!type.isGround())
          setValRefsTo(port, FieldRef(port, 0));
      } else if (auto type = type_dyn_cast<PropertyType>(port.getType())) {
        worklist.push_back(port);
      }
    }
  }

  void handlePorts(FieldRef ref, SmallVectorImpl<Value> &children) {
    if (auto inst = dyn_cast_or_null<InstanceOp>(ref.getDefiningOp())) {
      auto res = cast<OpResult>(ref.getValue());
      auto portNum = res.getResultNumber();
      auto refMod =
          dyn_cast_or_null<FModuleOp>(*instanceGraph.getReferencedModule(inst));
      if (!refMod)
        return;
      FieldRef modArg(refMod.getArgument(portNum), ref.getFieldID());
      auto pathIter = portPaths.find(modArg);
      if (pathIter == portPaths.end())
        return;
      for (auto modOutPort : pathIter->second) {
        auto outPortNum =
            cast<BlockArgument>(modOutPort.getValue()).getArgNumber();
        if (modOutPort.getFieldID() == 0) {
          children.push_back(inst.getResult(outPortNum));
          continue;
        }
        FieldRef instanceOutPort(inst.getResult(outPortNum),
                                 modOutPort.getFieldID());
        llvm::append_range(children, fieldToVals[instanceOutPort]);
      }
    } else if (auto mem = dyn_cast<MemOp>(ref.getDefiningOp())) {
      if (mem.getReadLatency() > 0)
        return;
      auto memPort = ref.getValue();
      auto type = type_cast<BundleType>(memPort.getType());
      auto enableFieldId = type.getFieldID((unsigned)ReadPortSubfield::en);
      auto dataFieldId = type.getFieldID((unsigned)ReadPortSubfield::data);
      auto addressFieldId = type.getFieldID((unsigned)ReadPortSubfield::addr);
      if (ref.getFieldID() == enableFieldId ||
          ref.getFieldID() == addressFieldId) {
        for (auto dataField : fieldToVals[FieldRef(memPort, dataFieldId)])
          children.push_back(dataField);
      }
    }
  }

  void reportLoopFound(Value childVal, VisitingSet visiting) {
    // TODO: Work harder to provide best information possible to user,
    // especially across instances or when we trace through aliasing values.
    // We're about to exit, and can afford to do some slower work here.
    auto getName = [&](Value v) {
      if (isa_and_nonnull<SubfieldOp, SubindexOp, SubaccessOp>(
              v.getDefiningOp())) {
        assert(!valRefersTo[v].empty());
        // Pick representative of the "alias set", not deterministic.
        return getFieldName(*valRefersTo[v].begin()).first;
      }
      return getFieldName(FieldRef(v, 0)).first;
    };
    auto errorDiag = mlir::emitError(
        module.getLoc(), "detected combinational cycle in a FIRRTL module");

    SmallVector<Value, 16> path;
    path.push_back(childVal);
    visiting.popUntilVal(
        childVal, [&](Value visitingVal) { path.push_back(visitingVal); });
    assert(path.back() == childVal);
    path.pop_back();

    // Find a value we can name
    auto *it =
        llvm::find_if(path, [&](Value v) { return !getName(v).empty(); });
    if (it == path.end()) {
      errorDiag.append(", but unable to find names for any involved values.");
      errorDiag.attachNote(childVal.getLoc()) << "cycle detected here";
      return;
    }
    errorDiag.append(", sample path: ");

    bool lastWasDots = false;
    errorDiag << module.getName() << ".{" << getName(*it);
    for (auto v :
         llvm::concat<Value>(llvm::make_range(std::next(it), path.end()),
                             llvm::make_range(path.begin(), std::next(it)))) {
      auto name = getName(v);
      if (!name.empty()) {
        errorDiag << " <- " << name;
        lastWasDots = false;
      } else {
        if (!lastWasDots)
          errorDiag << " <- ...";
        lastWasDots = true;
      }
    }
    errorDiag << "}";
  }

  LogicalResult handleConnects(Value dst,
                               SmallVector<FieldRef> &inputArgFields) {

    bool onlyFieldZero = true;
    auto pathsToOutPort = [&](FieldRef dstFieldRef) {
      if (dstFieldRef.getFieldID() != 0)
        onlyFieldZero = false;
      if (!isa<BlockArgument>(dstFieldRef.getValue())) {
        return failure();
      }
      onlyFieldZero = false;
      for (auto inArg : inputArgFields) {
        portPaths[inArg].insert(dstFieldRef);
      }
      return success();
    };
    forallRefersTo(dst, pathsToOutPort);

    if (onlyFieldZero) {
      if (isa<RegOp, RegResetOp, SubfieldOp, SubaccessOp, SubindexOp>(
              dst.getDefiningOp()))
        return failure();
    }
    return success();
  }

  void setValRefsTo(Value val, FieldRef ref) {
    assert(val && ref && " Value and Ref cannot be null");
    valRefersTo[val].insert(ref);
    auto fToVIter = fieldToVals.find(ref);
    if (fToVIter != fieldToVals.end()) {
      for (auto aliasingVal : fToVIter->second) {
        aliasingValuesMap[val].insert(aliasingVal);
        aliasingValuesMap[aliasingVal].insert(val);
      }
      fToVIter->getSecond().insert(val);
    } else
      fieldToVals[ref].insert(val);
  }

  void
  forallRefersTo(Value val,
                 const llvm::function_ref<LogicalResult(FieldRef &refNode)> f,
                 bool baseCase = true) {
    auto refersToIter = valRefersTo.find(val);
    if (refersToIter != valRefersTo.end()) {
      for (auto ref : refersToIter->second)
        if (f(ref).failed())
          return;
    } else if (baseCase) {
      FieldRef base(val, 0);
      if (f(base).failed())
        return;
    }
  }

  void dump() {
    for (const auto &valRef : valRefersTo) {
      llvm::dbgs() << "\n val :" << valRef.first;
      for (auto node : valRef.second)
        llvm::dbgs() << "\n Refers to :" << getFieldName(node).first;
    }
    for (const auto &dtv : fieldToVals) {
      llvm::dbgs() << "\n Field :" << getFieldName(dtv.first).first
                   << " ::" << dtv.first.getValue();
      for (auto val : dtv.second)
        llvm::dbgs() << "\n val :" << val;
    }
    for (const auto &p : portPaths) {
      llvm::dbgs() << "\n Output port : " << getFieldName(p.first).first
                   << " has comb path from :";
      for (const auto &src : p.second)
        llvm::dbgs() << "\n Input port : " << getFieldName(src).first;
    }
  }

  FModuleOp module;
  InstanceGraph &instanceGraph;
  /// Map of a Value to all the FieldRefs that it refers to.
  DenseMap<Value, SetOfFieldRefs> valRefersTo;

  DenseMap<Value, DenseSet<Value>> aliasingValuesMap;

  DenseMap<FieldRef, DenseSet<Value>> fieldToVals;
  /// Comb paths that exist between module ports. This is maintained across
  /// modules.
  DenseMap<FieldRef, SetOfFieldRefs> &portPaths;
};

/// This pass constructs a local graph for each module to detect combinational
/// cycles. To capture the cross-module combinational cycles, this pass inlines
/// the combinational paths between IOs of its subinstances into a subgraph and
/// encodes them in a `combPathsMap`.
class CheckCombLoopsPass : public CheckCombLoopsBase<CheckCombLoopsPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    DenseMap<FieldRef, SetOfFieldRefs> portPaths;
    // Traverse modules in a post order to make sure the combinational paths
    // between IOs of a module have been detected and recorded in `portPaths`
    // before we handle its parent modules.
    for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
      if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
        DiscoverLoops rdf(module, instanceGraph, portPaths);
        if (rdf.processModule().failed()) {
          return signalPassFailure();
        }
      }
    }
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<mlir::Pass> circt::firrtl::createCheckCombLoopsPass() {
  return std::make_unique<CheckCombLoopsPass>();
}
