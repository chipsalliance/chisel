//===- Analysis.cpp - Analysis Pass -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the Analysis pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace circt;
using namespace handshake;
using namespace mlir;

static bool isControlOp(Operation *op) {
  auto controlInterface = dyn_cast<handshake::ControlInterface>(op);
  return controlInterface && controlInterface.isControl();
}

namespace {
struct HandshakeDotPrintPass
    : public HandshakeDotPrintBase<HandshakeDotPrintPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Resolve the instance graph to get a top-level module.
    std::string topLevel;
    handshake::InstanceGraph uses;
    SmallVector<std::string> sortedFuncs;
    if (resolveInstanceGraph(m, uses, topLevel, sortedFuncs).failed()) {
      signalPassFailure();
      return;
    }

    handshake::FuncOp topLevelOp =
        cast<handshake::FuncOp>(m.lookupSymbol(topLevel));

    // Create top-level graph.
    std::error_code ec;
    llvm::raw_fd_ostream outfile(topLevel + ".dot", ec);
    mlir::raw_indented_ostream os(outfile);

    os << "Digraph G {\n";
    os.indent();
    os << "splines=spline;\n";
    os << "compound=true; // Allow edges between clusters\n";
    dotPrint(os, "TOP", topLevelOp, /*isTop=*/true);
    os.unindent();
    os << "}\n";
    outfile.close();
  };

private:
  /// Prints an instance of a handshake.func to the graph. Returns the unique
  /// name that was assigned to the instance.
  std::string dotPrint(mlir::raw_indented_ostream &os, StringRef parentName,
                       handshake::FuncOp f, bool isTop);

  /// Maintain a mapping of module names and the number of times one of those
  /// modules have been instantiated in the design. This is used to generate
  /// unique names in the output graph.
  std::map<std::string, unsigned> instanceIdMap;

  /// A mapping between operations and their unique name in the .dot file.
  DenseMap<Operation *, std::string> opNameMap;

  /// A mapping between block arguments and their unique name in the .dot file.
  DenseMap<Value, std::string> argNameMap;

  void setUsedByMapping(Value v, Operation *op, StringRef node);
  void setProducedByMapping(Value v, Operation *op, StringRef node);

  /// Returns the name of the vertex using 'v' through 'consumer'.
  std::string getUsedByNode(Value v, Operation *consumer);
  /// Returns the name of the vertex producing 'v' through 'producer'.
  std::string getProducedByNode(Value v, Operation *producer);

  /// Maintain mappings between a value, the operation which (uses/produces) it,
  /// and the node name which the (tail/head) of an edge should refer to. This
  /// is used to resolve edges across handshake.instance's.
  // "'value' used by 'operation*' is used by the 'string' vertex"
  DenseMap<Value, std::map<Operation *, std::string>> usedByMapping;
  // "'value' produced by 'operation*' is produced from the 'string' vertex"
  DenseMap<Value, std::map<Operation *, std::string>> producedByMapping;
};

struct HandshakeOpCountPass
    : public HandshakeOpCountBase<HandshakeOpCountPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    for (auto func : m.getOps<handshake::FuncOp>()) {
      std::map<std::string, int> cnts;
      for (Operation &op : func.getOps()) {
        llvm::TypeSwitch<Operation *, void>(&op)
            .Case<handshake::ConstantOp>([&](auto) { cnts["Constant"]++; })
            .Case<handshake::MuxOp>([&](auto) { cnts["Mux"]++; })
            .Case<handshake::LoadOp>([&](auto) { cnts["Load"]++; })
            .Case<handshake::StoreOp>([&](auto) { cnts["Store"]++; })
            .Case<handshake::MergeOp>([&](auto) { cnts["Merge"]++; })
            .Case<handshake::ForkOp>([&](auto) { cnts["Fork"]++; })
            .Case<handshake::BranchOp>([&](auto) { cnts["Branch"]++; })
            .Case<handshake::MemoryOp, handshake::ExternalMemoryOp>(
                [&](auto) { cnts["Memory"]++; })
            .Case<handshake::ControlMergeOp>(
                [&](auto) { cnts["CntrlMerge"]++; })
            .Case<handshake::SinkOp>([&](auto) { cnts["Sink"]++; })
            .Case<handshake::SourceOp>([&](auto) { cnts["Source"]++; })
            .Case<handshake::JoinOp>([&](auto) { cnts["Join"]++; })
            .Case<handshake::BufferOp>([&](auto) { cnts["Buffer"]++; })
            .Case<handshake::ConditionalBranchOp>(
                [&](auto) { cnts["Branch"]++; })
            .Case<arith::AddIOp>([&](auto) { cnts["Add"]++; })
            .Case<arith::SubIOp>([&](auto) { cnts["Sub"]++; })
            .Case<arith::AddIOp>([&](auto) { cnts["Add"]++; })
            .Case<arith::MulIOp>([&](auto) { cnts["Mul"]++; })
            .Case<arith::CmpIOp>([&](auto) { cnts["Cmp"]++; })
            .Case<arith::IndexCastOp, arith::ShLIOp, arith::ShRSIOp,
                  arith::ShRUIOp>([&](auto) { cnts["Ext/Sh"]++; })
            .Case<handshake::ReturnOp>([&](auto) {})
            .Default([&](auto op) {
              llvm::outs() << "Unhandled operation: " << *op << "\n";
              assert(false);
            });
      }

      llvm::outs() << "// RESOURCES"
                   << "\n";
      for (auto it : cnts)
        llvm::outs() << it.first << "\t" << it.second << "\n";
      llvm::outs() << "// END"
                   << "\n";
    }
  }
};

} // namespace

/// Prints an operation to the dot file and returns the unique name for the
/// operation within the graph.
static std::string dotPrintNode(mlir::raw_indented_ostream &outfile,
                                StringRef instanceName, Operation *op,
                                DenseMap<Operation *, unsigned> &opIDs) {

  // We use "." to distinguish hierarchy in the dot file, but an op by default
  // prints using "." between the dialect name and the op name. Replace uses of
  // "." with "_".
  std::string opDialectName = op->getName().getStringRef().str();
  std::replace(opDialectName.begin(), opDialectName.end(), '.', '_');
  std::string opName = (instanceName + "." + opDialectName).str();

  // Follow the naming convention used in FIRRTL lowering.
  auto idAttr = op->getAttrOfType<IntegerAttr>("handshake_id");
  if (idAttr)
    opName += "_id" + std::to_string(idAttr.getValue().getZExtValue());
  else
    opName += std::to_string(opIDs[op]);

  outfile << "\"" << opName << "\""
          << " [";

  /// Fill color
  outfile << "fillcolor = ";
  outfile
      << llvm::TypeSwitch<Operation *, std::string>(op)
             .Case<handshake::ForkOp, handshake::LazyForkOp, handshake::MuxOp,
                   handshake::JoinOp>([&](auto) { return "lavender"; })
             .Case<handshake::BufferOp>([&](auto) { return "lightgreen"; })
             .Case<handshake::ReturnOp>([&](auto) { return "gold"; })
             .Case<handshake::SinkOp, handshake::ConstantOp>(
                 [&](auto) { return "gainsboro"; })
             .Case<handshake::MemoryOp, handshake::LoadOp, handshake::StoreOp>(
                 [&](auto) { return "coral"; })
             .Case<handshake::MergeOp, handshake::ControlMergeOp,
                   handshake::BranchOp, handshake::ConditionalBranchOp>(
                 [&](auto) { return "lightblue"; })
             .Default([&](auto) { return "moccasin"; });

  /// Shape
  outfile << ", shape=";
  if (op->getDialect()->getNamespace() == "handshake")
    outfile << "box";
  else
    outfile << "oval";

  /// Label
  outfile << ", label=\"";
  outfile << llvm::TypeSwitch<Operation *, std::string>(op)
                 .Case<handshake::ConstantOp>([&](auto op) {
                   return std::to_string(
                       op->template getAttrOfType<mlir::IntegerAttr>("value")
                           .getValue()
                           .getSExtValue());
                 })
                 .Case<handshake::ControlMergeOp>(
                     [&](auto) { return "cmerge"; })
                 .Case<handshake::ConditionalBranchOp>(
                     [&](auto) { return "cbranch"; })
                 .Case<handshake::BufferOp>([&](auto op) {
                   std::string n = "buffer ";
                   n += stringifyEnum(op.getBufferType());
                   return n;
                 })
                 .Case<arith::AddIOp>([&](auto) { return "+"; })
                 .Case<arith::SubIOp>([&](auto) { return "-"; })
                 .Case<arith::AndIOp>([&](auto) { return "&"; })
                 .Case<arith::OrIOp>([&](auto) { return "|"; })
                 .Case<arith::XOrIOp>([&](auto) { return "^"; })
                 .Case<arith::MulIOp>([&](auto) { return "*"; })
                 .Case<arith::ShRSIOp, arith::ShRUIOp>(
                     [&](auto) { return ">>"; })
                 .Case<arith::ShLIOp>([&](auto) { return "<<"; })
                 .Case<arith::CmpIOp>([&](arith::CmpIOp op) {
                   switch (op.getPredicate()) {
                   case arith::CmpIPredicate::eq:
                     return "==";
                   case arith::CmpIPredicate::ne:
                     return "!=";
                   case arith::CmpIPredicate::uge:
                   case arith::CmpIPredicate::sge:
                     return ">=";
                   case arith::CmpIPredicate::ugt:
                   case arith::CmpIPredicate::sgt:
                     return ">";
                   case arith::CmpIPredicate::ule:
                   case arith::CmpIPredicate::sle:
                     return "<=";
                   case arith::CmpIPredicate::ult:
                   case arith::CmpIPredicate::slt:
                     return "<";
                   }
                   llvm_unreachable("unhandled cmpi predicate");
                 })
                 .Default([&](auto op) {
                   auto opDialect = op->getDialect()->getNamespace();
                   std::string label = op->getName().getStringRef().str();
                   if (opDialect == "handshake")
                     label.erase(0, StringLiteral("handshake.").size());

                   return label;
                 });
  /// If we have an ID attribute, we'll add the ID of the operation as well.
  /// This helps crossprobing the diagram with the Handshake IR and waveform
  /// diagrams.
  if (idAttr)
    outfile << " [" << std::to_string(idAttr.getValue().getZExtValue()) << "]";

  outfile << "\"";

  /// Style; add dashed border for control nodes
  outfile << ", style=\"filled";
  if (isControlOp(op))
    outfile << ", dashed";
  outfile << "\"";
  outfile << "]\n";

  return opName;
}

/// Returns true if v is used as a control operand in op
static bool isControlOperand(Operation *op, Value v) {
  if (isControlOp(op))
    return true;

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<handshake::MuxOp, handshake::ConditionalBranchOp>(
          [&](auto op) { return v == op.getOperand(0); })
      .Case<handshake::ControlMergeOp>([&](auto) { return true; })
      .Default([](auto) { return false; });
}

static std::string getLocalName(StringRef instanceName, StringRef suffix) {
  return (instanceName + "." + suffix).str();
}

static std::string getArgName(handshake::FuncOp op, unsigned index) {
  return op.getArgName(index).getValue().str();
}

static std::string getUniqueArgName(StringRef instanceName,
                                    handshake::FuncOp op, unsigned index) {
  return getLocalName(instanceName, getArgName(op, index));
}

static std::string getResName(handshake::FuncOp op, unsigned index) {
  return op.getResName(index).getValue().str();
}

static std::string getUniqueResName(StringRef instanceName,
                                    handshake::FuncOp op, unsigned index) {
  return getLocalName(instanceName, getResName(op, index));
}

void HandshakeDotPrintPass::setUsedByMapping(Value v, Operation *op,
                                             StringRef node) {
  usedByMapping[v][op] = node;
}
void HandshakeDotPrintPass::setProducedByMapping(Value v, Operation *op,
                                                 StringRef node) {
  producedByMapping[v][op] = node;
}

std::string HandshakeDotPrintPass::getUsedByNode(Value v, Operation *consumer) {
  // Check if there is any mapping registerred for the value-use relation.
  auto it = usedByMapping.find(v);
  if (it != usedByMapping.end()) {
    auto it2 = it->second.find(consumer);
    if (it2 != it->second.end())
      return it2->second;
  }

  // fallback to the registerred name for the operation
  auto opNameIt = opNameMap.find(consumer);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

std::string HandshakeDotPrintPass::getProducedByNode(Value v,
                                                     Operation *producer) {
  // Check if there is any mapping registerred for the value-produce relation.
  auto it = producedByMapping.find(v);
  if (it != producedByMapping.end()) {
    auto it2 = it->second.find(producer);
    if (it2 != it->second.end())
      return it2->second;
  }

  // fallback to the registerred name for the operation
  auto opNameIt = opNameMap.find(producer);
  assert(opNameIt != opNameMap.end() &&
         "No name registered for the operation!");
  return opNameIt->second;
}

/// Emits additional, non-graphviz information about the connection between
/// from- and to. This does not have any effect on the graph itself, but may be
/// used by other tools to reason about the connectivity between nodes.
static void tryAddExtraEdgeInfo(mlir::raw_indented_ostream &os, Operation *from,
                                Value result, Operation *to) {
  os << " // ";

  if (from) {
    // Output port information
    auto results = from->getResults();
    unsigned resIdx =
        std::distance(results.begin(), llvm::find(results, result));
    auto fromNamedOpInterface = dyn_cast<handshake::NamedIOInterface>(from);
    if (fromNamedOpInterface) {
      auto resName = fromNamedOpInterface.getResultName(resIdx);
      os << " output=\"" << resName << "\"";
    } else
      os << " output=\"out" << resIdx << "\"";
  }

  if (to) {
    // Input port information
    auto ops = to->getOperands();
    unsigned opIdx = std::distance(ops.begin(), llvm::find(ops, result));
    auto toNamedOpInterface = dyn_cast<handshake::NamedIOInterface>(to);
    if (toNamedOpInterface) {
      auto opName = toNamedOpInterface.getOperandName(opIdx);
      os << " input=\"" << opName << "\"";
    } else
      os << " input=\"in" << opIdx << "\"";
  }
}

std::string HandshakeDotPrintPass::dotPrint(mlir::raw_indented_ostream &os,
                                            StringRef parentName,
                                            handshake::FuncOp f, bool isTop) {
  // Prints DOT representation of the dataflow graph, used for debugging.
  DenseMap<Block *, unsigned> blockIDs;
  std::map<std::string, unsigned> opTypeCntrs;
  DenseMap<Operation *, unsigned> opIDs;
  auto name = f.getName();
  unsigned thisId = instanceIdMap[name.str()]++;
  std::string instanceName = parentName.str() + "." + name.str();
  // Follow submodule naming convention from FIRRTL lowering:
  if (!isTop)
    instanceName += std::to_string(thisId);

  /// Maintain a reference to any node in the args, body and result. These are
  /// used to generate cluster edges at the end of this function, to facilitate
  /// a  nice layout.
  std::optional<std::string> anyArg, anyBody, anyRes;

  unsigned i = 0;

  // Sequentially scan across the operations in the function and assign instance
  // IDs to each operation.
  for (Block &block : f) {
    blockIDs[&block] = i++;
    for (Operation &op : block)
      opIDs[&op] = opTypeCntrs[op.getName().getStringRef().str()]++;
  }

  if (!isTop) {
    os << "// Subgraph for instance of " << name << "\n";
    os << "subgraph \"cluster_" << instanceName << "\" {\n";
    os.indent();
    os << "label = \"" << name << "\"\n";
    os << "labeljust=\"l\"\n";
    os << "color = \"darkgreen\"\n";
  }
  os << "node [shape=box style=filled fillcolor=\"white\"]\n";

  Block *bodyBlock = &f.getBody().front();

  /// Print function arg and res nodes.
  os << "// Function argument nodes\n";
  std::string argsCluster = "cluster_" + instanceName + "_args";
  os << "subgraph \"" << argsCluster << "\" {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  for (const auto &barg : enumerate(bodyBlock->getArguments())) {
    auto argName = getArgName(f, barg.index());
    auto localArgName = getLocalName(instanceName, argName);
    os << "\"" << localArgName << "\" [shape=diamond";
    if (barg.index() == bodyBlock->getNumArguments() - 1) // ctrl
      os << ", style=dashed";
    os << " label=\"" << argName << "\"";
    os << "]\n";
    if (!anyArg.has_value())
      anyArg = localArgName;
  }
  os.unindent();
  os << "}\n";

  os << "// Function return nodes\n";
  std::string resCluster = "cluster_" + instanceName + "_res";
  os << "subgraph \"" << resCluster << "\" {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  // Get the return op; a handshake.func always has a terminator, making this
  // safe.
  auto returnOp = *f.getBody().getOps<handshake::ReturnOp>().begin();
  for (const auto &res : llvm::enumerate(returnOp.getOperands())) {
    auto resName = getResName(f, res.index());
    auto uniqueResName = getUniqueResName(instanceName, f, res.index());
    os << "\"" << uniqueResName << "\" [shape=diamond";
    if (res.index() == bodyBlock->getNumArguments() - 1) // ctrl
      os << ", style=dashed";
    os << " label=\"" << resName << "\"";
    os << "]\n";

    // Create a mapping between the return op argument uses and the return
    // nodes.
    setUsedByMapping(res.value(), returnOp, uniqueResName);

    if (!anyRes.has_value())
      anyRes = uniqueResName;
  }
  os.unindent();
  os << "}\n";

  /// Print operation nodes.
  std::string opsCluster = "cluster_" + instanceName + "_ops";
  os << "subgraph \"" << opsCluster << "\" {\n";
  os.indent();
  // No label or border; the subgraph just forces args to stay together in the
  // diagram.
  os << "label=\"\"\n";
  os << "peripheries=0\n";
  for (Operation &op : *bodyBlock) {
    if (!isa<handshake::InstanceOp, handshake::ReturnOp>(op)) {
      // Regular node in the diagram.
      opNameMap[&op] = dotPrintNode(os, instanceName, &op, opIDs);
      continue;
    }
    auto instOp = dyn_cast<handshake::InstanceOp>(op);
    if (instOp) {
      // Recurse into instantiated submodule.
      auto calledFuncOp =
          instOp->getParentOfType<ModuleOp>().lookupSymbol<handshake::FuncOp>(
              instOp.getModule());
      assert(calledFuncOp);
      auto subInstanceName = dotPrint(os, instanceName, calledFuncOp, false);

      // Create a mapping between the instance arguments and the arguments to
      // the module which it instantiated.
      for (const auto &arg : llvm::enumerate(instOp.getOperands())) {
        setUsedByMapping(
            arg.value(), instOp,
            getUniqueArgName(subInstanceName, calledFuncOp, arg.index()));
      }
      // Create a  mapping between the instance results and the results from the
      // module which it instantiated.
      for (const auto &res : llvm::enumerate(instOp.getResults())) {
        setProducedByMapping(
            res.value(), instOp,
            getUniqueResName(subInstanceName, calledFuncOp, res.index()));
      }
    }
  }
  if (!opNameMap.empty())
    anyBody = opNameMap.begin()->second;

  os.unindent();
  os << "}\n";

  /// Print operation result edges.
  os << "// Operation result edges\n";
  for (Operation &op : *bodyBlock) {
    for (auto result : op.getResults()) {
      for (auto &u : result.getUses()) {
        Operation *useOp = u.getOwner();
        if (useOp->getBlock() == bodyBlock) {
          os << "\"" << getProducedByNode(result, &op);
          os << "\" -> \"";
          os << getUsedByNode(result, useOp) << "\"";
          if (isControlOp(&op) || isControlOperand(useOp, result))
            os << " [style=\"dashed\"]";

          // Add extra, non-graphviz info to the edge.
          tryAddExtraEdgeInfo(os, &op, result, useOp);

          os << "\n";
        }
      }
    }
  }

  if (!isTop)
    os << "}\n";

  /// Print edges for function argument uses.
  os << "// Function argument edges\n";
  for (const auto &barg : enumerate(bodyBlock->getArguments())) {
    auto argName = getArgName(f, barg.index());
    os << "\"" << getLocalName(instanceName, argName) << "\" [shape=diamond";
    if (barg.index() == bodyBlock->getNumArguments() - 1)
      os << ", style=dashed";
    os << "]\n";
    for (auto *useOp : barg.value().getUsers()) {
      os << "\"" << getLocalName(instanceName, argName) << "\" -> \""
         << getUsedByNode(barg.value(), useOp) << "\"";
      if (isControlOperand(useOp, barg.value()))
        os << " [style=\"dashed\"]";

      tryAddExtraEdgeInfo(os, nullptr, barg.value(), useOp);
      os << "\n";
    }
  }

  /// Print edges from arguments cluster to ops cluster and ops cluster to
  /// results cluser, to coerce a nice layout.
  if (anyArg.has_value() && anyBody.has_value())
    os << "\"" << anyArg.value() << "\" -> \"" << anyBody.value()
       << "\" [lhead=\"" << opsCluster << "\" ltail=\"" << argsCluster
       << "\" style=invis]\n";
  if (anyBody.has_value() && anyRes.has_value())
    os << "\"" << anyBody.value() << "\" -> \"" << anyRes.value()
       << "\" [lhead=\"" << resCluster << "\" ltail=\"" << opsCluster
       << "\" style=invis]\n";

  os.unindent();
  return instanceName;
}

namespace {
struct HandshakeAddIDsPass : public HandshakeAddIDsBase<HandshakeAddIDsPass> {
  void runOnOperation() override {
    handshake::FuncOp funcOp = getOperation();
    auto *ctx = &getContext();
    OpBuilder builder(funcOp);
    funcOp.walk([&](Operation *op) {
      if (op->hasAttr("handshake_id"))
        return;
      llvm::SmallVector<NamedAttribute> attrs;
      llvm::copy(op->getAttrs(), std::back_inserter(attrs));
      attrs.push_back(builder.getNamedAttr(
          "handshake_id",
          IntegerAttr::get(IndexType::get(ctx),
                           opCounters[op->getName().getStringRef().str()]++)));

      op->setAttrs(DictionaryAttr::get(ctx, attrs));
    });
  };

private:
  /// Maintain a counter for each operation type in the function.
  std::map<std::string, unsigned> opCounters;
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeDotPrintPass() {
  return std::make_unique<HandshakeDotPrintPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::handshake::createHandshakeOpCountPass() {
  return std::make_unique<HandshakeOpCountPass>();
}

std::unique_ptr<mlir::Pass> circt::handshake::createHandshakeAddIDsPass() {
  return std::make_unique<HandshakeAddIDsPass>();
}
