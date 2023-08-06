//===- CalyxOps.cpp - Calyx op code defs ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where op definitions live.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

using namespace circt;
using namespace circt::calyx;
using namespace mlir;

namespace {

// A struct to enforce that the LHS template is one of the RHS templates.
// For example:
//   std::is_any<uint32_t, uint16_t, float, int32_t>::value is false.
template <class T, class... Ts>
struct IsAny : std::disjunction<std::is_same<T, Ts>...> {};

} // namespace

//===----------------------------------------------------------------------===//
// Utilities related to Direction
//===----------------------------------------------------------------------===//

Direction direction::get(bool isOutput) {
  return static_cast<Direction>(isOutput);
}

IntegerAttr direction::packAttribute(MLIRContext *ctx, size_t nIns,
                                     size_t nOuts) {
  // Pack the array of directions into an APInt.  Input direction is zero,
  // output direction is one.
  size_t numDirections = nIns + nOuts;
  APInt portDirections(/*width=*/numDirections, /*value=*/0);
  for (size_t i = nIns, e = numDirections; i != e; ++i)
    portDirections.setBit(i);

  return IntegerAttr::get(IntegerType::get(ctx, numDirections), portDirections);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// This pattern collapses a calyx.seq or calyx.par operation when it
/// contains exactly one calyx.enable operation.
template <typename CtrlOp>
struct CollapseUnaryControl : mlir::OpRewritePattern<CtrlOp> {
  using mlir::OpRewritePattern<CtrlOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter &rewriter) const override {
    auto &ops = ctrlOp.getBodyBlock()->getOperations();
    bool isUnaryControl =
        (ops.size() == 1) && isa<EnableOp>(ops.front()) &&
        isa<SeqOp, ParOp, StaticSeqOp, StaticParOp>(ctrlOp->getParentOp());
    if (!isUnaryControl)
      return failure();

    ops.front().moveBefore(ctrlOp);
    rewriter.eraseOp(ctrlOp);
    return success();
  }
};

/// Verify that the value is not a "complex" value. For example, the source
/// of an AssignOp should be a constant or port, e.g.
/// %and = comb.and %a, %b : i1
/// calyx.assign %port = %c1_i1 ? %and   : i1   // Incorrect
/// calyx.assign %port = %and   ? %c1_i1 : i1   // Correct
/// TODO(Calyx): This is useful to verify current MLIR can be lowered to the
/// native compiler. Remove this when Calyx supports wire declarations.
/// See: https://github.com/llvm/circt/pull/1774 for context.
template <typename Op>
static LogicalResult verifyNotComplexSource(Op op) {
  Operation *definingOp = op.getSrc().getDefiningOp();
  if (definingOp == nullptr)
    // This is a port of the parent component.
    return success();

  // Currently, we use the Combinational dialect to perform logical operations
  // on wires, i.e. comb::AndOp, comb::OrOp, comb::XorOp.
  if (auto dialect = definingOp->getDialect(); isa<comb::CombDialect>(dialect))
    return op->emitOpError("has source that is not a port or constant. "
                           "Complex logic should be conducted in the guard.");

  return success();
}

/// Convenience function for getting the SSA name of `v` under the scope of
/// operation `scopeOp`.
static std::string valueName(Operation *scopeOp, Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  // CAVEAT: Since commit 27df7158fe MLIR prefers verifiers to print errors for
  // operations in generic form, and the printer by default runs a verification.
  // `valueName` is used in some of these verifiers where preferably the generic
  // operand form should be used instead.
  AsmState asmState(scopeOp, OpPrintingFlags().assumeVerified());
  v.printAsOperand(os, asmState);
  return s;
}

/// Returns whether this value is either (1) a port on a ComponentOp or (2) a
/// port on a cell interface.
static bool isPort(Value value) {
  Operation *definingOp = value.getDefiningOp();
  return value.isa<BlockArgument>() ||
         (definingOp && isa<CellInterface>(definingOp));
}

/// Gets the port for a given BlockArgument.
PortInfo calyx::getPortInfo(BlockArgument arg) {
  Operation *op = arg.getOwner()->getParentOp();
  assert(isa<ComponentInterface>(op) &&
         "Only ComponentInterface should support lookup by BlockArgument.");
  return cast<ComponentInterface>(op).getPortInfo()[arg.getArgNumber()];
}

/// Returns whether the given operation has a control region.
static bool hasControlRegion(Operation *op) {
  return isa<ControlOp, SeqOp, IfOp, RepeatOp, WhileOp, ParOp, StaticRepeatOp,
             StaticParOp, StaticSeqOp, StaticIfOp>(op);
}

/// Returns whether the given operation is a static control operator
static bool isStaticControl(Operation *op) {
  if (isa<EnableOp>(op)) {
    // for enables, we need to check whether its corresponding group is static
    auto component = op->getParentOfType<ComponentOp>();
    auto enableOp = llvm::cast<EnableOp>(op);
    StringRef groupName = enableOp.getGroupName();
    auto group = component.getWiresOp().lookupSymbol<GroupInterface>(groupName);
    return isa<StaticGroupOp>(group);
  }
  return isa<StaticIfOp, StaticSeqOp, StaticRepeatOp, StaticParOp>(op);
}

/// Verifies the body of a ControlLikeOp.
static LogicalResult verifyControlBody(Operation *op) {
  if (isa<SeqOp, ParOp, StaticSeqOp, StaticParOp>(op))
    // This does not apply to sequential and parallel regions.
    return success();

  // Some ControlLike operations have (possibly) multiple regions, e.g. IfOp.
  for (auto &region : op->getRegions()) {
    auto opsIt = region.getOps();
    size_t numOperations = std::distance(opsIt.begin(), opsIt.end());
    // A body of a ControlLike operation may have a single EnableOp within it.
    // However, that must be the only operation.
    //  E.g. Allowed:  calyx.control { calyx.enable @A }
    //   Not Allowed:  calyx.control { calyx.enable @A calyx.seq { ... } }
    bool usesEnableAsCompositionOperator =
        numOperations > 1 && llvm::any_of(region.front(), [](auto &&bodyOp) {
          return isa<EnableOp>(bodyOp);
        });
    if (usesEnableAsCompositionOperator)
      return op->emitOpError(
          "EnableOp is not a composition operator. It should be nested "
          "in a control flow operation, such as \"calyx.seq\"");

    // Verify that multiple control flow operations are nested inside a single
    // control operator. See: https://github.com/llvm/circt/issues/1723
    size_t numControlFlowRegions =
        llvm::count_if(opsIt, [](auto &&op) { return hasControlRegion(&op); });
    if (numControlFlowRegions > 1)
      return op->emitOpError(
          "has an invalid control sequence. Multiple control flow operations "
          "must all be nested in a single calyx.seq or calyx.par");
  }
  return success();
}

LogicalResult calyx::verifyComponent(Operation *op) {
  auto *opParent = op->getParentOp();
  if (!isa<ModuleOp>(opParent))
    return op->emitOpError()
           << "has parent: " << opParent << ", expected ModuleOp.";
  return success();
}

LogicalResult calyx::verifyCell(Operation *op) {
  auto opParent = op->getParentOp();
  if (!isa<ComponentInterface>(opParent))
    return op->emitOpError()
           << "has parent: " << opParent << ", expected ComponentInterface.";
  return success();
}

LogicalResult calyx::verifyControlLikeOp(Operation *op) {
  auto parent = op->getParentOp();

  if (isa<calyx::EnableOp>(op) &&
      !isa<calyx::CalyxDialect>(parent->getDialect())) {
    // Allow embedding calyx.enable ops within other dialects. This is motivated
    // by allowing experimentation with new styles of Calyx lowering. For more
    // info and the historical discussion, see:
    // https://github.com/llvm/circt/pull/3211
    return success();
  }

  if (!hasControlRegion(parent))
    return op->emitOpError()
           << "has parent: " << parent
           << ", which is not allowed for a control-like operation.";

  if (op->getNumRegions() == 0)
    return success();

  auto &region = op->getRegion(0);
  // Operations that are allowed in the body of a ControlLike op.
  auto isValidBodyOp = [](Operation *operation) {
    return isa<EnableOp, InvokeOp, SeqOp, IfOp, RepeatOp, WhileOp, ParOp,
               StaticParOp, StaticRepeatOp, StaticSeqOp, StaticIfOp>(operation);
  };
  for (auto &&bodyOp : region.front()) {
    if (isValidBodyOp(&bodyOp))
      continue;

    return op->emitOpError()
           << "has operation: " << bodyOp.getName()
           << ", which is not allowed in this control-like operation";
  }
  return verifyControlBody(op);
}

LogicalResult calyx::verifyIf(Operation *op) {
  auto ifOp = dyn_cast<IfInterface>(op);

  if (ifOp.elseBodyExists() && ifOp.getElseBody()->empty())
    return ifOp->emitOpError() << "empty 'else' region.";

  return success();
}

// Helper function for parsing a group port operation, i.e. GroupDoneOp and
// GroupPortOp. These may take one of two different forms:
// (1) %<guard> ? %<src> : i1
// (2) %<src> : i1
static ParseResult parseGroupPort(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfos;
  OpAsmParser::UnresolvedOperand guardOrSource;
  if (parser.parseOperand(guardOrSource))
    return failure();

  if (succeeded(parser.parseOptionalQuestion())) {
    OpAsmParser::UnresolvedOperand source;
    // The guard exists.
    if (parser.parseOperand(source))
      return failure();
    operandInfos.push_back(source);
  }
  // No matter if this is the source or guard, it should be last.
  operandInfos.push_back(guardOrSource);

  Type type;
  // Resolving the operands with the same type works here since the source and
  // guard of a group port is always i1.
  if (parser.parseColonType(type) ||
      parser.resolveOperands(operandInfos, type, result.operands))
    return failure();

  return success();
}

// A helper function for printing group ports, i.e. GroupGoOp and GroupDoneOp.
template <typename GroupPortType>
static void printGroupPort(OpAsmPrinter &p, GroupPortType op) {
  static_assert(IsAny<GroupPortType, GroupGoOp, GroupDoneOp>(),
                "Should be a Calyx Group port.");

  p << " ";
  // The guard is optional.
  Value guard = op.getGuard(), source = op.getSrc();
  if (guard)
    p << guard << " ? ";
  p << source << " : " << source.getType();
}

// Collapse nested control of the same type for SeqOp and ParOp, e.g.
// calyx.seq { calyx.seq { ... } } -> calyx.seq { ... }
template <typename OpTy>
static LogicalResult collapseControl(OpTy controlOp,
                                     PatternRewriter &rewriter) {
  static_assert(IsAny<OpTy, SeqOp, ParOp, StaticSeqOp, StaticParOp>(),
                "Should be a SeqOp, ParOp, StaticSeqOp, or StaticParOp");

  if (isa<OpTy>(controlOp->getParentOp())) {
    Block *controlBody = controlOp.getBodyBlock();
    for (auto &op : make_early_inc_range(*controlBody))
      op.moveBefore(controlOp);

    rewriter.eraseOp(controlOp);
    return success();
  }

  return failure();
}

template <typename OpTy>
static LogicalResult emptyControl(OpTy controlOp, PatternRewriter &rewriter) {
  if (controlOp.getBodyBlock()->empty()) {
    rewriter.eraseOp(controlOp);
    return success();
  }
  return failure();
}

/// A helper function to check whether the conditional and group (if it exists)
/// needs to be erased to maintain a valid state of a Calyx program. If these
/// have no more uses, they will be erased.
template <typename OpTy>
static void eraseControlWithGroupAndConditional(OpTy op,
                                                PatternRewriter &rewriter) {
  static_assert(IsAny<OpTy, IfOp, WhileOp>(),
                "This is only applicable to WhileOp and IfOp.");

  // Save information about the operation, and erase it.
  Value cond = op.getCond();
  std::optional<StringRef> groupName = op.getGroupName();
  auto component = op->template getParentOfType<ComponentOp>();
  rewriter.eraseOp(op);

  // Clean up the attached conditional and combinational group (if it exists).
  if (groupName) {
    auto group = component.getWiresOp().template lookupSymbol<GroupInterface>(
        *groupName);
    if (SymbolTable::symbolKnownUseEmpty(group, component.getRegion()))
      rewriter.eraseOp(group);
  }
  // Check the conditional after the Group, since it will be driven within.
  if (!cond.isa<BlockArgument>() && cond.getDefiningOp()->use_empty())
    rewriter.eraseOp(cond.getDefiningOp());
}

/// A helper function to check whether the conditional needs to be erased
/// to maintain a valid state of a Calyx program. If these
/// have no more uses, they will be erased.
template <typename OpTy>
static void eraseControlWithConditional(OpTy op, PatternRewriter &rewriter) {
  static_assert(std::is_same<OpTy, StaticIfOp>(),
                "This is only applicable to StatifIfOp.");

  // Save information about the operation, and erase it.
  Value cond = op.getCond();
  rewriter.eraseOp(op);

  // Check if conditional is still needed, and remove if it isn't
  if (!cond.isa<BlockArgument>() && cond.getDefiningOp()->use_empty())
    rewriter.eraseOp(cond.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// ComponentInterface
//===----------------------------------------------------------------------===//

template <typename ComponentTy>
static void printComponentInterface(OpAsmPrinter &p, ComponentInterface comp) {
  auto componentName = comp->template getAttrOfType<StringAttr>(
                               ::mlir::SymbolTable::getSymbolAttrName())
                           .getValue();
  p << " ";
  p.printSymbolName(componentName);

  // Print the port definition list for input and output ports.
  auto printPortDefList = [&](auto ports) {
    p << "(";
    llvm::interleaveComma(ports, p, [&](const PortInfo &port) {
      p << "%" << port.name.getValue() << ": " << port.type;
      if (!port.attributes.empty()) {
        p << " ";
        p.printAttributeWithoutType(port.attributes);
      }
    });
    p << ")";
  };
  printPortDefList(comp.getInputPortInfo());
  p << " -> ";
  printPortDefList(comp.getOutputPortInfo());

  p << " ";
  p.printRegion(*comp.getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false,
                /*printEmptyBlock=*/false);

  SmallVector<StringRef> elidedAttrs = {
      "portAttributes",
      "portNames",
      "portDirections",
      "sym_name",
      ComponentTy::getFunctionTypeAttrName(comp->getName()),
      ComponentTy::getArgAttrsAttrName(comp->getName()),
      ComponentTy::getResAttrsAttrName(comp->getName())};
  p.printOptionalAttrDict(comp->getAttrs(), elidedAttrs);
}

/// Parses the ports of a Calyx component signature, and adds the corresponding
/// port names to `attrName`.
static ParseResult
parsePortDefList(OpAsmParser &parser, OperationState &result,
                 SmallVectorImpl<OpAsmParser::Argument> &ports,
                 SmallVectorImpl<Type> &portTypes,
                 SmallVectorImpl<NamedAttrList> &portAttrs) {
  auto parsePort = [&]() -> ParseResult {
    OpAsmParser::Argument port;
    Type portType;
    // Expect each port to have the form `%<ssa-name> : <type>`.
    if (parser.parseArgument(port) || parser.parseColon() ||
        parser.parseType(portType))
      return failure();
    port.type = portType;
    ports.push_back(port);
    portTypes.push_back(portType);

    NamedAttrList portAttr;
    portAttrs.push_back(succeeded(parser.parseOptionalAttrDict(portAttr))
                            ? portAttr
                            : NamedAttrList());
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parsePort);
}

/// Parses the signature of a Calyx component.
static ParseResult
parseComponentSignature(OpAsmParser &parser, OperationState &result,
                        SmallVectorImpl<OpAsmParser::Argument> &ports,
                        SmallVectorImpl<Type> &portTypes) {
  SmallVector<OpAsmParser::Argument> inPorts, outPorts;
  SmallVector<Type> inPortTypes, outPortTypes;
  SmallVector<NamedAttrList> portAttributes;

  if (parsePortDefList(parser, result, inPorts, inPortTypes, portAttributes))
    return failure();

  if (parser.parseArrow() ||
      parsePortDefList(parser, result, outPorts, outPortTypes, portAttributes))
    return failure();

  auto *context = parser.getBuilder().getContext();
  // Add attribute for port names; these are currently
  // just inferred from the SSA names of the component.
  SmallVector<Attribute> portNames;
  auto getPortName = [context](const auto &port) -> StringAttr {
    StringRef name = port.ssaName.name;
    if (name.startswith("%"))
      name = name.drop_front();
    return StringAttr::get(context, name);
  };
  llvm::transform(inPorts, std::back_inserter(portNames), getPortName);
  llvm::transform(outPorts, std::back_inserter(portNames), getPortName);

  result.addAttribute("portNames", ArrayAttr::get(context, portNames));
  result.addAttribute(
      "portDirections",
      direction::packAttribute(context, inPorts.size(), outPorts.size()));

  ports.append(inPorts);
  ports.append(outPorts);
  portTypes.append(inPortTypes);
  portTypes.append(outPortTypes);

  SmallVector<Attribute> portAttrs;
  llvm::transform(portAttributes, std::back_inserter(portAttrs),
                  [&](auto attr) { return attr.getDictionary(context); });
  result.addAttribute("portAttributes", ArrayAttr::get(context, portAttrs));

  return success();
}

template <typename ComponentTy>
static ParseResult parseComponentInterface(OpAsmParser &parser,
                                           OperationState &result) {
  using namespace mlir::function_interface_impl;

  StringAttr componentName;
  if (parser.parseSymbolName(componentName,
                             ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<mlir::OpAsmParser::Argument> ports;

  SmallVector<Type> portTypes;
  if (parseComponentSignature(parser, result, ports, portTypes))
    return failure();

  // Build the component's type for FunctionLike trait. All ports are listed
  // as arguments so they may be accessed within the component.
  auto type = parser.getBuilder().getFunctionType(portTypes, /*results=*/{});
  result.addAttribute(ComponentTy::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, ports))
    return failure();

  if (body->empty())
    body->push_back(new Block());

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

/// Returns a new vector containing the concatenation of vectors `a` and `b`.
template <typename T>
static SmallVector<T> concat(const SmallVectorImpl<T> &a,
                             const SmallVectorImpl<T> &b) {
  SmallVector<T> out;
  out.append(a);
  out.append(b);
  return out;
}

static void buildComponentLike(OpBuilder &builder, OperationState &result,
                               StringAttr name, ArrayRef<PortInfo> ports,
                               bool combinational) {
  using namespace mlir::function_interface_impl;

  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  std::pair<SmallVector<Type, 8>, SmallVector<Type, 8>> portIOTypes;
  std::pair<SmallVector<Attribute, 8>, SmallVector<Attribute, 8>> portIONames;
  std::pair<SmallVector<Attribute, 8>, SmallVector<Attribute, 8>>
      portIOAttributes;
  SmallVector<Direction, 8> portDirections;
  // Avoid using llvm::partition or llvm::sort to preserve relative ordering
  // between individual inputs and outputs.
  for (auto &&port : ports) {
    bool isInput = port.direction == Direction::Input;
    (isInput ? portIOTypes.first : portIOTypes.second).push_back(port.type);
    (isInput ? portIONames.first : portIONames.second).push_back(port.name);
    (isInput ? portIOAttributes.first : portIOAttributes.second)
        .push_back(port.attributes);
  }
  auto portTypes = concat(portIOTypes.first, portIOTypes.second);
  auto portNames = concat(portIONames.first, portIONames.second);
  auto portAttributes = concat(portIOAttributes.first, portIOAttributes.second);

  // Build the function type of the component.
  auto functionType = builder.getFunctionType(portTypes, {});
  if (combinational) {
    result.addAttribute(CombComponentOp::getFunctionTypeAttrName(result.name),
                        TypeAttr::get(functionType));
  } else {
    result.addAttribute(ComponentOp::getFunctionTypeAttrName(result.name),
                        TypeAttr::get(functionType));
  }

  // Record the port names and number of input ports of the component.
  result.addAttribute("portNames", builder.getArrayAttr(portNames));
  result.addAttribute("portDirections",
                      direction::packAttribute(builder.getContext(),
                                               portIOTypes.first.size(),
                                               portIOTypes.second.size()));
  // Record the attributes of the ports.
  result.addAttribute("portAttributes", builder.getArrayAttr(portAttributes));

  // Create a single-blocked region.
  Region *region = result.addRegion();
  Block *body = new Block();
  region->push_back(body);

  // Add all ports to the body.
  body->addArguments(portTypes, SmallVector<Location, 4>(
                                    portTypes.size(), builder.getUnknownLoc()));

  // Insert the WiresOp and ControlOp.
  IRRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(body);
  builder.create<WiresOp>(result.location);
  if (!combinational)
    builder.create<ControlOp>(result.location);
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

/// This is a helper function that should only be used to get the WiresOp or
/// ControlOp of a ComponentOp, which are guaranteed to exist and generally at
/// the end of a component's body. In the worst case, this will run in linear
/// time with respect to the number of instances within the component.
template <typename Op>
static Op getControlOrWiresFrom(ComponentOp op) {
  auto *body = op.getBodyBlock();
  // We verify there is a single WiresOp and ControlOp,
  // so this is safe.
  auto opIt = body->getOps<Op>().begin();
  return *opIt;
}

/// Returns the Block argument with the given name from a ComponentOp.
/// If the name doesn't exist, returns an empty Value.
static Value getBlockArgumentWithName(StringRef name, ComponentOp op) {
  ArrayAttr portNames = op.getPortNames();

  for (size_t i = 0, e = portNames.size(); i != e; ++i) {
    auto portName = portNames[i].cast<StringAttr>();
    if (portName.getValue() == name)
      return op.getBodyBlock()->getArgument(i);
  }
  return Value{};
}

WiresOp calyx::ComponentOp::getWiresOp() {
  return getControlOrWiresFrom<WiresOp>(*this);
}

ControlOp calyx::ComponentOp::getControlOp() {
  return getControlOrWiresFrom<ControlOp>(*this);
}

Value calyx::ComponentOp::getGoPort() {
  return getBlockArgumentWithName("go", *this);
}

Value calyx::ComponentOp::getDonePort() {
  return getBlockArgumentWithName("done", *this);
}

Value calyx::ComponentOp::getClkPort() {
  return getBlockArgumentWithName("clk", *this);
}

Value calyx::ComponentOp::getResetPort() {
  return getBlockArgumentWithName("reset", *this);
}

SmallVector<PortInfo> ComponentOp::getPortInfo() {
  auto portTypes = getArgumentTypes();
  ArrayAttr portNamesAttr = getPortNames(), portAttrs = getPortAttributes();
  APInt portDirectionsAttr = getPortDirections();

  SmallVector<PortInfo> results;
  for (size_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
    results.push_back(PortInfo{portNamesAttr[i].cast<StringAttr>(),
                               portTypes[i],
                               direction::get(portDirectionsAttr[i]),
                               portAttrs[i].cast<DictionaryAttr>()});
  }
  return results;
}

/// A helper function to return a filtered subset of a component's ports.
template <typename Pred>
static SmallVector<PortInfo> getFilteredPorts(ComponentOp op, Pred p) {
  SmallVector<PortInfo> ports = op.getPortInfo();
  llvm::erase_if(ports, p);
  return ports;
}

SmallVector<PortInfo> ComponentOp::getInputPortInfo() {
  return getFilteredPorts(
      *this, [](const PortInfo &port) { return port.direction == Output; });
}

SmallVector<PortInfo> ComponentOp::getOutputPortInfo() {
  return getFilteredPorts(
      *this, [](const PortInfo &port) { return port.direction == Input; });
}

void ComponentOp::print(OpAsmPrinter &p) {
  printComponentInterface<ComponentOp>(p, *this);
}

ParseResult ComponentOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseComponentInterface<ComponentOp>(parser, result);
}

/// Determines whether the given ComponentOp has all the required ports.
static LogicalResult hasRequiredPorts(ComponentOp op) {
  // Get all identifiers from the component ports.
  llvm::SmallVector<StringRef, 4> identifiers;
  for (PortInfo &port : op.getPortInfo()) {
    auto portIds = port.getAllIdentifiers();
    identifiers.append(portIds.begin(), portIds.end());
  }
  // Sort the identifiers: a pre-condition for std::set_intersection.
  std::sort(identifiers.begin(), identifiers.end());

  llvm::SmallVector<StringRef, 4> intersection,
      interfacePorts{"clk", "done", "go", "reset"};
  // Find the intersection between all identifiers and required ports.
  std::set_intersection(interfacePorts.begin(), interfacePorts.end(),
                        identifiers.begin(), identifiers.end(),
                        std::back_inserter(intersection));

  if (intersection.size() == interfacePorts.size())
    return success();

  SmallVector<StringRef, 4> difference;
  std::set_difference(interfacePorts.begin(), interfacePorts.end(),
                      intersection.begin(), intersection.end(),
                      std::back_inserter(difference));
  return op->emitOpError()
         << "is missing the following required port attribute identifiers: "
         << difference;
}

LogicalResult ComponentOp::verify() {
  // Verify there is exactly one of each the wires and control operations.
  auto wIt = getBodyBlock()->getOps<WiresOp>();
  auto cIt = getBodyBlock()->getOps<ControlOp>();
  if (std::distance(wIt.begin(), wIt.end()) +
          std::distance(cIt.begin(), cIt.end()) !=
      2)
    return emitOpError() << "requires exactly one of each: '"
                         << WiresOp::getOperationName() << "', '"
                         << ControlOp::getOperationName() << "'.";

  if (failed(hasRequiredPorts(*this)))
    return failure();

  // Verify the component actually does something: has a non-empty Control
  // region, or continuous assignments.
  bool hasNoControlConstructs =
      getControlOp().getBodyBlock()->getOperations().empty();
  bool hasNoAssignments =
      getWiresOp().getBodyBlock()->getOps<AssignOp>().empty();
  if (hasNoControlConstructs && hasNoAssignments)
    return emitOpError(
        "The component currently does nothing. It needs to either have "
        "continuous assignments in the Wires region or control constructs in "
        "the Control region.");

  return success();
}

void ComponentOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<PortInfo> ports) {
  buildComponentLike(builder, result, name, ports, /*combinational=*/false);
}

void ComponentOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto ports = getPortNames();
  auto *block = &getRegion()->front();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i)
    setNameFn(block->getArgument(i), ports[i].cast<StringAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// CombComponentOp
//===----------------------------------------------------------------------===//

SmallVector<PortInfo> CombComponentOp::getPortInfo() {
  auto portTypes = getArgumentTypes();
  ArrayAttr portNamesAttr = getPortNames(), portAttrs = getPortAttributes();
  APInt portDirectionsAttr = getPortDirections();

  SmallVector<PortInfo> results;
  for (size_t i = 0, e = portNamesAttr.size(); i != e; ++i) {
    results.push_back(PortInfo{portNamesAttr[i].cast<StringAttr>(),
                               portTypes[i],
                               direction::get(portDirectionsAttr[i]),
                               portAttrs[i].cast<DictionaryAttr>()});
  }
  return results;
}

WiresOp calyx::CombComponentOp::getWiresOp() {
  auto *body = getBodyBlock();
  auto opIt = body->getOps<WiresOp>().begin();
  return *opIt;
}

/// A helper function to return a filtered subset of a comb component's ports.
template <typename Pred>
static SmallVector<PortInfo> getFilteredPorts(CombComponentOp op, Pred p) {
  SmallVector<PortInfo> ports = op.getPortInfo();
  llvm::erase_if(ports, p);
  return ports;
}

SmallVector<PortInfo> CombComponentOp::getInputPortInfo() {
  return getFilteredPorts(
      *this, [](const PortInfo &port) { return port.direction == Output; });
}

SmallVector<PortInfo> CombComponentOp::getOutputPortInfo() {
  return getFilteredPorts(
      *this, [](const PortInfo &port) { return port.direction == Input; });
}

void CombComponentOp::print(OpAsmPrinter &p) {
  printComponentInterface<CombComponentOp>(p, *this);
}

ParseResult CombComponentOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseComponentInterface<CombComponentOp>(parser, result);
}

LogicalResult CombComponentOp::verify() {
  // Verify there is exactly one wires operation.
  auto wIt = getBodyBlock()->getOps<WiresOp>();
  if (std::distance(wIt.begin(), wIt.end()) != 1)
    return emitOpError() << "requires exactly one "
                         << WiresOp::getOperationName() << " op.";

  // Verify there is not a control operation.
  auto cIt = getBodyBlock()->getOps<ControlOp>();
  if (std::distance(cIt.begin(), cIt.end()) != 0)
    return emitOpError() << "must not have a `" << ControlOp::getOperationName()
                         << "` op.";

  // Verify the component actually does something: has continuous assignments.
  bool hasNoAssignments =
      getWiresOp().getBodyBlock()->getOps<AssignOp>().empty();
  if (hasNoAssignments)
    return emitOpError(
        "The component currently does nothing. It needs to either have "
        "continuous assignments in the Wires region or control constructs in "
        "the Control region.");

  // Check that all cells are combinational
  auto cells = getOps<CellInterface>();
  for (auto cell : cells) {
    if (!cell.isCombinational())
      return emitOpError() << "contains non-combinational cell "
                           << cell.instanceName();
  }

  // Check that the component has no groups
  auto groups = getWiresOp().getOps<GroupOp>();
  if (!groups.empty())
    return emitOpError() << "contains group " << (*groups.begin()).getSymName();

  // Combinational groups aren't allowed in combinational components either.
  // For more information see here:
  // https://docs.calyxir.org/lang/ref.html#comb-group-definitions
  auto combGroups = getWiresOp().getOps<CombGroupOp>();
  if (!combGroups.empty())
    return emitOpError() << "contains comb group "
                         << (*combGroups.begin()).getSymName();

  return success();
}

void CombComponentOp::build(OpBuilder &builder, OperationState &result,
                            StringAttr name, ArrayRef<PortInfo> ports) {
  buildComponentLike(builder, result, name, ports, /*combinational=*/true);
}

void CombComponentOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  auto ports = getPortNames();
  auto *block = &getRegion()->front();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i)
    setNameFn(block->getArgument(i), ports[i].cast<StringAttr>().getValue());
}

//===----------------------------------------------------------------------===//
// ControlOp
//===----------------------------------------------------------------------===//
LogicalResult ControlOp::verify() { return verifyControlBody(*this); }

// Get the InvokeOps of this ControlOp.
SmallVector<InvokeOp, 4> ControlOp::getInvokeOps() {
  SmallVector<InvokeOp, 4> ret;
  this->walk([&](InvokeOp invokeOp) { ret.push_back(invokeOp); });
  return ret;
}

//===----------------------------------------------------------------------===//
// SeqOp
//===----------------------------------------------------------------------===//

void SeqOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add(collapseControl<SeqOp>);
  patterns.add(emptyControl<SeqOp>);
  patterns.insert<CollapseUnaryControl<SeqOp>>(context);
}

//===----------------------------------------------------------------------===//
// StaticSeqOp
//===----------------------------------------------------------------------===//

LogicalResult StaticSeqOp::verify() {
  // StaticSeqOp should only have static control in it
  auto &ops = (*this).getBodyBlock()->getOperations();
  if (!llvm::all_of(ops, [&](Operation &op) { return isStaticControl(&op); })) {
    return emitOpError("StaticSeqOp has non static control within it");
  }

  return success();
}

void StaticSeqOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add(collapseControl<StaticSeqOp>);
  patterns.add(emptyControl<StaticSeqOp>);
  patterns.insert<CollapseUnaryControl<StaticSeqOp>>(context);
}

//===----------------------------------------------------------------------===//
// ParOp
//===----------------------------------------------------------------------===//

LogicalResult ParOp::verify() {
  llvm::SmallSet<StringRef, 8> groupNames;

  // Add loose requirement that the body of a ParOp may not enable the same
  // Group more than once, e.g. calyx.par { calyx.enable @G calyx.enable @G }
  for (EnableOp op : getBodyBlock()->getOps<EnableOp>()) {
    StringRef groupName = op.getGroupName();
    if (groupNames.count(groupName))
      return emitOpError() << "cannot enable the same group: \"" << groupName
                           << "\" more than once.";
    groupNames.insert(groupName);
  }

  return success();
}

void ParOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
  patterns.add(collapseControl<ParOp>);
  patterns.add(emptyControl<ParOp>);
  patterns.insert<CollapseUnaryControl<ParOp>>(context);
}

//===----------------------------------------------------------------------===//
// StaticParOp
//===----------------------------------------------------------------------===//

LogicalResult StaticParOp::verify() {
  llvm::SmallSet<StringRef, 8> groupNames;

  // Add loose requirement that the body of a ParOp may not enable the same
  // Group more than once, e.g. calyx.par { calyx.enable @G calyx.enable @G }
  for (EnableOp op : getBodyBlock()->getOps<EnableOp>()) {
    StringRef groupName = op.getGroupName();
    if (groupNames.count(groupName))
      return emitOpError() << "cannot enable the same group: \"" << groupName
                           << "\" more than once.";
    groupNames.insert(groupName);
  }

  // static par must only have static control in it
  auto &ops = (*this).getBodyBlock()->getOperations();
  for (Operation &op : ops) {
    if (!isStaticControl(&op)) {
      return op.emitOpError("StaticParOp has non static control within it");
    }
  }

  return success();
}

void StaticParOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add(collapseControl<StaticParOp>);
  patterns.add(emptyControl<StaticParOp>);
  patterns.insert<CollapseUnaryControl<StaticParOp>>(context);
}

//===----------------------------------------------------------------------===//
// WiresOp
//===----------------------------------------------------------------------===//
LogicalResult WiresOp::verify() {
  auto componentInterface = (*this)->getParentOfType<ComponentInterface>();
  if (llvm::isa<ComponentOp>(componentInterface)) {
    auto component = llvm::cast<ComponentOp>(componentInterface);
    auto control = component.getControlOp();

    // Verify each group is referenced in the control section.
    for (auto &&op : *getBodyBlock()) {
      if (!isa<GroupInterface>(op))
        continue;
      auto group = cast<GroupInterface>(op);
      auto groupName = group.symName();
      if (mlir::SymbolTable::symbolKnownUseEmpty(groupName, control))
        return op.emitOpError()
               << "with name: " << groupName
               << " is unused in the control execution schedule";
    }
  }

  // Verify that:
  // - At most one continuous assignment exists for any given value
  // - A continuously assigned wire has no assignments inside groups.
  for (auto thisAssignment : getBodyBlock()->getOps<AssignOp>()) {
    // Always assume guarded assignments will not be driven simultaneously. We
    // liberally assume that guards are mutually exclusive (more elaborate
    // static and dynamic checking can be performed to validate such cases).
    if (thisAssignment.getGuard())
      continue;

    Value dest = thisAssignment.getDest();
    for (Operation *user : dest.getUsers()) {
      auto assignUser = dyn_cast<AssignOp>(user);
      if (!assignUser || assignUser.getDest() != dest ||
          assignUser == thisAssignment)
        continue;

      return user->emitOpError() << "destination is already continuously "
                                    "driven. Other assignment is "
                                 << thisAssignment;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CombGroupOp
//===----------------------------------------------------------------------===//

/// Verifies the defining operation of a value is combinational.
static LogicalResult isCombinational(Value value, GroupInterface group) {
  Operation *definingOp = value.getDefiningOp();
  if (definingOp == nullptr || definingOp->hasTrait<Combinational>())
    // This is a port of the parent component or combinational.
    return success();

  // For now, assumes all component instances are combinational. Once
  // combinational components are supported, this can be strictly enforced.
  if (isa<InstanceOp>(definingOp))
    return success();

  // Constants and logical operations are OK.
  if (isa<comb::CombDialect, hw::HWDialect>(definingOp->getDialect()))
    return success();

  // Reads to MemoryOp and RegisterOp are combinational. Writes are not.
  if (auto r = dyn_cast<RegisterOp>(definingOp)) {
    return value == r.getOut()
               ? success()
               : group->emitOpError()
                     << "with register: \"" << r.instanceName()
                     << "\" is conducting a memory store. This is not "
                        "combinational.";
  } else if (auto m = dyn_cast<MemoryOp>(definingOp)) {
    auto writePorts = {m.writeData(), m.writeEn()};
    return (llvm::none_of(writePorts, [&](Value p) { return p == value; }))
               ? success()
               : group->emitOpError()
                     << "with memory: \"" << m.instanceName()
                     << "\" is conducting a memory store. This "
                        "is not combinational.";
  }

  std::string portName =
      valueName(group->getParentOfType<ComponentOp>(), value);
  return group->emitOpError() << "with port: " << portName
                              << ". This operation is not combinational.";
}

/// Verifies a combinational group may contain only combinational primitives or
/// perform combinational logic.
LogicalResult CombGroupOp::verify() {
  for (auto &&op : *getBodyBlock()) {
    auto assign = dyn_cast<AssignOp>(op);
    if (assign == nullptr)
      continue;
    Value dst = assign.getDest(), src = assign.getSrc();
    if (failed(isCombinational(dst, *this)) ||
        failed(isCombinational(src, *this)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GroupGoOp
//===----------------------------------------------------------------------===//
GroupGoOp GroupOp::getGoOp() {
  auto goOps = getBodyBlock()->getOps<GroupGoOp>();
  size_t nOps = std::distance(goOps.begin(), goOps.end());
  return nOps ? *goOps.begin() : GroupGoOp();
}

GroupDoneOp GroupOp::getDoneOp() {
  auto body = this->getBodyBlock();
  return cast<GroupDoneOp>(body->getTerminator());
}

//===----------------------------------------------------------------------===//
// CycleOp
//===----------------------------------------------------------------------===//
void CycleOp::print(OpAsmPrinter &p) {
  p << " ";
  // The guard is optional.
  auto start = this->getStart();
  auto end = this->getEnd();
  if (end.has_value()) {
    p << "[" << start << ":" << end.value() << "]";
  } else {
    p << start;
  }
}

ParseResult CycleOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operandInfos;

  uint32_t startLiteral;
  uint32_t endLiteral;

  auto hasEnd = succeeded(parser.parseOptionalLSquare());

  if (parser.parseInteger(startLiteral)) {
    parser.emitError(parser.getNameLoc(), "Could not parse start cycle");
    return failure();
  }

  auto start = parser.getBuilder().getI32IntegerAttr(startLiteral);
  result.addAttribute(getStartAttrName(result.name), start);

  if (hasEnd) {
    if (parser.parseColon())
      return failure();

    if (auto res = parser.parseOptionalInteger(endLiteral); res.has_value()) {
      auto end = parser.getBuilder().getI32IntegerAttr(endLiteral);
      result.addAttribute(getEndAttrName(result.name), end);
    }

    if (parser.parseRSquare())
      return failure();
  }

  result.addTypes(parser.getBuilder().getI1Type());

  return success();
}

LogicalResult CycleOp::verify() {
  uint32_t latency = this->getGroupLatency();

  if (this->getStart() >= latency) {
    emitOpError("start cycle must be less than the group latency");
    return failure();
  }

  if (this->getEnd().has_value()) {
    if (this->getStart() >= this->getEnd().value()) {
      emitOpError("start cycle must be less than end cycle");
      return failure();
    }

    if (this->getEnd() >= latency) {
      emitOpError("end cycle must be less than the group latency");
      return failure();
    }
  }

  return success();
}

uint32_t CycleOp::getGroupLatency() {
  auto group = (*this)->getParentOfType<StaticGroupOp>();
  return group.getLatency();
}

//===----------------------------------------------------------------------===//
// GroupInterface
//===----------------------------------------------------------------------===//

/// Determines whether the given port is used in the group. Its use depends on
/// the `isDriven` value; if true, then the port should be a destination in an
/// AssignOp. Otherwise, it should be the source, i.e. a read.
static bool portIsUsedInGroup(GroupInterface group, Value port, bool isDriven) {
  return llvm::any_of(port.getUses(), [&](auto &&use) {
    auto assignOp = dyn_cast<AssignOp>(use.getOwner());
    if (assignOp == nullptr)
      return false;

    Operation *parent = assignOp->getParentOp();
    if (isa<WiresOp>(parent))
      // This is a continuous assignment.
      return false;

    // A port is used if it meet the criteria:
    // (1) it is a {source, destination} of an assignment.
    // (2) that assignment is found in the provided group.

    // If not driven, then read.
    Value expected = isDriven ? assignOp.getDest() : assignOp.getSrc();
    return expected == port && group == parent;
  });
}

/// Checks whether `port` is driven from within `groupOp`.
static LogicalResult portDrivenByGroup(GroupInterface groupOp, Value port) {
  // Check if the port is driven by an assignOp from within `groupOp`.
  if (portIsUsedInGroup(groupOp, port, /*isDriven=*/true))
    return success();

  // If `port` is an output of a cell then we conservatively enforce that at
  // least one input port of the cell must be driven by the group.
  if (auto cell = dyn_cast<CellInterface>(port.getDefiningOp());
      cell && cell.direction(port) == calyx::Direction::Output)
    return groupOp.drivesAnyPort(cell.getInputPorts());

  return failure();
}

LogicalResult GroupOp::drivesPort(Value port) {
  return portDrivenByGroup(*this, port);
}

LogicalResult CombGroupOp::drivesPort(Value port) {
  return portDrivenByGroup(*this, port);
}

LogicalResult StaticGroupOp::drivesPort(Value port) {
  return portDrivenByGroup(*this, port);
}

/// Checks whether all ports are driven within the group.
static LogicalResult allPortsDrivenByGroup(GroupInterface group,
                                           ValueRange ports) {
  return success(llvm::all_of(ports, [&](Value port) {
    return portIsUsedInGroup(group, port, /*isDriven=*/true);
  }));
}

LogicalResult GroupOp::drivesAllPorts(ValueRange ports) {
  return allPortsDrivenByGroup(*this, ports);
}

LogicalResult CombGroupOp::drivesAllPorts(ValueRange ports) {
  return allPortsDrivenByGroup(*this, ports);
}

LogicalResult StaticGroupOp::drivesAllPorts(ValueRange ports) {
  return allPortsDrivenByGroup(*this, ports);
}

/// Checks whether any ports are driven within the group.
static LogicalResult anyPortsDrivenByGroup(GroupInterface group,
                                           ValueRange ports) {
  return success(llvm::any_of(ports, [&](Value port) {
    return portIsUsedInGroup(group, port, /*isDriven=*/true);
  }));
}

LogicalResult GroupOp::drivesAnyPort(ValueRange ports) {
  return anyPortsDrivenByGroup(*this, ports);
}

LogicalResult CombGroupOp::drivesAnyPort(ValueRange ports) {
  return anyPortsDrivenByGroup(*this, ports);
}

LogicalResult StaticGroupOp::drivesAnyPort(ValueRange ports) {
  return anyPortsDrivenByGroup(*this, ports);
}

/// Checks whether any ports are read within the group.
static LogicalResult anyPortsReadByGroup(GroupInterface group,
                                         ValueRange ports) {
  return success(llvm::any_of(ports, [&](Value port) {
    return portIsUsedInGroup(group, port, /*isDriven=*/false);
  }));
}

LogicalResult GroupOp::readsAnyPort(ValueRange ports) {
  return anyPortsReadByGroup(*this, ports);
}

LogicalResult CombGroupOp::readsAnyPort(ValueRange ports) {
  return anyPortsReadByGroup(*this, ports);
}

LogicalResult StaticGroupOp::readsAnyPort(ValueRange ports) {
  return anyPortsReadByGroup(*this, ports);
}

/// Verifies that certain ports of primitives are either driven or read
/// together.
static LogicalResult verifyPrimitivePortDriving(AssignOp assign,
                                                GroupInterface group) {
  Operation *destDefiningOp = assign.getDest().getDefiningOp();
  if (destDefiningOp == nullptr)
    return success();
  auto destCell = dyn_cast<CellInterface>(destDefiningOp);
  if (destCell == nullptr)
    return success();

  LogicalResult verifyWrites =
      TypeSwitch<Operation *, LogicalResult>(destCell)
          .Case<RegisterOp>([&](auto op) {
            // We only want to verify this is written to if the {write enable,
            // in} port is driven.
            return succeeded(group.drivesAnyPort({op.getWriteEn(), op.getIn()}))
                       ? group.drivesAllPorts({op.getWriteEn(), op.getIn()})
                       : success();
          })
          .Case<MemoryOp>([&](auto op) {
            SmallVector<Value> requiredWritePorts;
            // If writing to memory, write_en, write_data, and all address ports
            // should be driven.
            requiredWritePorts.push_back(op.writeEn());
            requiredWritePorts.push_back(op.writeData());
            for (Value address : op.addrPorts())
              requiredWritePorts.push_back(address);

            // We only want to verify the write ports if either write_data or
            // write_en is driven.
            return succeeded(
                       group.drivesAnyPort({op.writeData(), op.writeEn()}))
                       ? group.drivesAllPorts(requiredWritePorts)
                       : success();
          })
          .Case<AndLibOp, OrLibOp, XorLibOp, AddLibOp, SubLibOp, GtLibOp,
                LtLibOp, EqLibOp, NeqLibOp, GeLibOp, LeLibOp, LshLibOp,
                RshLibOp, SgtLibOp, SltLibOp, SeqLibOp, SneqLibOp, SgeLibOp,
                SleLibOp, SrshLibOp>([&](auto op) {
            Value lhs = op.getLeft(), rhs = op.getRight();
            return succeeded(group.drivesAnyPort({lhs, rhs}))
                       ? group.drivesAllPorts({lhs, rhs})
                       : success();
          })
          .Default([&](auto op) { return success(); });

  if (failed(verifyWrites))
    return group->emitOpError()
           << "with cell: " << destCell->getName() << " \""
           << destCell.instanceName()
           << "\" is performing a write and failed to drive all necessary "
              "ports.";

  Operation *srcDefiningOp = assign.getSrc().getDefiningOp();
  if (srcDefiningOp == nullptr)
    return success();
  auto srcCell = dyn_cast<CellInterface>(srcDefiningOp);
  if (srcCell == nullptr)
    return success();

  LogicalResult verifyReads =
      TypeSwitch<Operation *, LogicalResult>(srcCell)
          .Case<MemoryOp>([&](auto op) {
            // If reading memory, all address ports should be driven. Note that
            // we only want to verify the read ports if read_data is used in the
            // group.
            return succeeded(group.readsAnyPort({op.readData()}))
                       ? group.drivesAllPorts(op.addrPorts())
                       : success();
          })
          .Default([&](auto op) { return success(); });

  if (failed(verifyReads))
    return group->emitOpError() << "with cell: " << srcCell->getName() << " \""
                                << srcCell.instanceName()
                                << "\" is having a read performed upon it, and "
                                   "failed to drive all necessary ports.";

  return success();
}

LogicalResult calyx::verifyGroupInterface(Operation *op) {
  auto group = dyn_cast<GroupInterface>(op);
  if (group == nullptr)
    return success();

  for (auto &&groupOp : *group.getBody()) {
    auto assign = dyn_cast<AssignOp>(groupOp);
    if (assign == nullptr)
      continue;
    if (failed(verifyPrimitivePortDriving(assign, group)))
      return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Utilities for operations with the Cell trait.
//===----------------------------------------------------------------------===//

/// Gives each result of the cell a meaningful name in the form:
/// <instance-name>.<port-name>
static void getCellAsmResultNames(OpAsmSetValueNameFn setNameFn, Operation *op,
                                  ArrayRef<StringRef> portNames) {
  auto cellInterface = dyn_cast<CellInterface>(op);
  assert(cellInterface && "must implement the Cell interface");

  std::string prefix = cellInterface.instanceName().str() + ".";
  for (size_t i = 0, e = portNames.size(); i != e; ++i)
    setNameFn(op->getResult(i), prefix + portNames[i].str());
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

/// Determines whether the given direction is valid with the given inputs. The
/// `isDestination` boolean is used to distinguish whether the value is a source
/// or a destination.
static LogicalResult verifyPortDirection(Operation *op, Value value,
                                         bool isDestination) {
  Operation *definingOp = value.getDefiningOp();
  bool isComponentPort = value.isa<BlockArgument>(),
       isCellInterfacePort = definingOp && isa<CellInterface>(definingOp);
  assert((isComponentPort || isCellInterfacePort) && "Not a port.");

  PortInfo port = isComponentPort
                      ? getPortInfo(value.cast<BlockArgument>())
                      : cast<CellInterface>(definingOp).portInfo(value);

  bool isSource = !isDestination;
  // Component output ports and cell interface input ports should be driven.
  Direction validDirection =
      (isDestination && isComponentPort) || (isSource && isCellInterfacePort)
          ? Direction::Output
          : Direction::Input;

  return port.direction == validDirection
             ? success()
             : op->emitOpError()
                   << "has a " << (isComponentPort ? "component" : "cell")
                   << " port as the "
                   << (isDestination ? "destination" : "source")
                   << " with the incorrect direction.";
}

/// Verifies the value of a given assignment operation. The boolean
/// `isDestination` is used to distinguish whether the destination
/// or source of the AssignOp is to be verified.
static LogicalResult verifyAssignOpValue(AssignOp op, bool isDestination) {
  bool isSource = !isDestination;
  Value value = isDestination ? op.getDest() : op.getSrc();
  if (isPort(value))
    return verifyPortDirection(op, value, isDestination);

  // A destination may also be the Go or Done hole of a GroupOp.
  if (isDestination && !isa<GroupGoOp, GroupDoneOp>(value.getDefiningOp()))
    return op->emitOpError(
        "has an invalid destination port. It must be drive-able.");
  else if (isSource)
    return verifyNotComplexSource(op);

  return success();
}

LogicalResult AssignOp::verify() {
  bool isDestination = true, isSource = false;
  if (failed(verifyAssignOpValue(*this, isDestination)))
    return failure();
  if (failed(verifyAssignOpValue(*this, isSource)))
    return failure();

  return success();
}

ParseResult AssignOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand destination;
  if (parser.parseOperand(destination) || parser.parseEqual())
    return failure();

  // An AssignOp takes one of the two following forms:
  // (1) %<dest> = %<src> : <type>
  // (2) %<dest> = %<guard> ? %<src> : <type>
  OpAsmParser::UnresolvedOperand guardOrSource;
  if (parser.parseOperand(guardOrSource))
    return failure();

  // Since the guard is optional, we need to check if there is an accompanying
  // `?` symbol.
  OpAsmParser::UnresolvedOperand source;
  bool hasGuard = succeeded(parser.parseOptionalQuestion());
  if (hasGuard) {
    // The guard exists. Parse the source.
    if (parser.parseOperand(source))
      return failure();
  }

  Type type;
  if (parser.parseColonType(type) ||
      parser.resolveOperand(destination, type, result.operands))
    return failure();

  if (hasGuard) {
    Type i1Type = parser.getBuilder().getI1Type();
    // Since the guard is optional, it is listed last in the arguments of the
    // AssignOp. Therefore, we must parse the source first.
    if (parser.resolveOperand(source, type, result.operands) ||
        parser.resolveOperand(guardOrSource, i1Type, result.operands))
      return failure();
  } else {
    // This is actually a source.
    if (parser.resolveOperand(guardOrSource, type, result.operands))
      return failure();
  }

  return success();
}

void AssignOp::print(OpAsmPrinter &p) {
  p << " " << getDest() << " = ";

  Value bguard = getGuard(), source = getSrc();
  // The guard is optional.
  if (bguard)
    p << bguard << " ? ";

  // We only need to print a single type; the destination and source are
  // guaranteed to be the same type.
  p << source << " : " << source.getType();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the component for the symbol. This returns null on
/// invalid IR.
ComponentInterface InstanceOp::getReferencedComponent() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;

  return module.lookupSymbol<ComponentInterface>(getComponentName());
}

/// Verifies the port information in comparison with the referenced component
/// of an instance. This helper function avoids conducting a lookup for the
/// referenced component twice.
static LogicalResult
verifyInstanceOpType(InstanceOp instance,
                     ComponentInterface referencedComponent) {
  auto module = instance->getParentOfType<ModuleOp>();
  StringRef entryPointName =
      module->getAttrOfType<StringAttr>("calyx.entrypoint");
  if (instance.getComponentName() == entryPointName)
    return instance.emitOpError()
           << "cannot reference the entry-point component: '" << entryPointName
           << "'.";

  // Verify the instance result ports with those of its referenced component.
  SmallVector<PortInfo> componentPorts = referencedComponent.getPortInfo();
  size_t numPorts = componentPorts.size();

  size_t numResults = instance.getNumResults();
  if (numResults != numPorts)
    return instance.emitOpError()
           << "has a wrong number of results; expected: " << numPorts
           << " but got " << numResults;

  for (size_t i = 0; i != numResults; ++i) {
    auto resultType = instance.getResult(i).getType();
    auto expectedType = componentPorts[i].type;
    if (resultType == expectedType)
      continue;
    return instance.emitOpError()
           << "result type for " << componentPorts[i].name << " must be "
           << expectedType << ", but got " << resultType;
  }
  return success();
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = *this;
  auto module = op->getParentOfType<ModuleOp>();
  Operation *referencedComponent =
      symbolTable.lookupNearestSymbolFrom(module, getComponentNameAttr());
  if (referencedComponent == nullptr)
    return emitError() << "referencing component: '" << getComponentName()
                       << "', which does not exist.";

  Operation *shadowedComponentName =
      symbolTable.lookupNearestSymbolFrom(module, getSymNameAttr());
  if (shadowedComponentName != nullptr)
    return emitError() << "instance symbol: '" << instanceName()
                       << "' is already a symbol for another component.";

  // Verify the referenced component is not instantiating itself.
  auto parentComponent = op->getParentOfType<ComponentOp>();
  if (parentComponent == referencedComponent)
    return emitError() << "recursive instantiation of its parent component: '"
                       << getComponentName() << "'";

  assert(isa<ComponentInterface>(referencedComponent) &&
         "Should be a ComponentInterface.");
  return verifyInstanceOpType(*this,
                              cast<ComponentInterface>(referencedComponent));
}

/// Provide meaningful names to the result values of an InstanceOp.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> InstanceOp::portNames() {
  SmallVector<StringRef> portNames;
  for (Attribute name : getReferencedComponent().getPortNames())
    portNames.push_back(name.cast<StringAttr>().getValue());
  return portNames;
}

SmallVector<Direction> InstanceOp::portDirections() {
  SmallVector<Direction> portDirections;
  for (const PortInfo &port : getReferencedComponent().getPortInfo())
    portDirections.push_back(port.direction);
  return portDirections;
}

SmallVector<DictionaryAttr> InstanceOp::portAttributes() {
  SmallVector<DictionaryAttr> portAttributes;
  for (const PortInfo &port : getReferencedComponent().getPortInfo())
    portAttributes.push_back(port.attributes);
  return portAttributes;
}

bool InstanceOp::isCombinational() {
  return isa<CombComponentOp>(getReferencedComponent());
}

//===----------------------------------------------------------------------===//
// PrimitiveOp
//===----------------------------------------------------------------------===//

/// Lookup the component for the symbol. This returns null on
/// invalid IR.
hw::HWModuleExternOp PrimitiveOp::getReferencedPrimitive() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return nullptr;

  return module.lookupSymbol<hw::HWModuleExternOp>(getPrimitiveName());
}

/// Verifies the port information in comparison with the referenced component
/// of an instance. This helper function avoids conducting a lookup for the
/// referenced component twice.
static LogicalResult
verifyPrimitiveOpType(PrimitiveOp instance,
                      hw::HWModuleExternOp referencedPrimitive) {
  auto module = instance->getParentOfType<ModuleOp>();
  StringRef entryPointName =
      module->getAttrOfType<StringAttr>("calyx.entrypoint");
  if (instance.getPrimitiveName() == entryPointName)
    return instance.emitOpError()
           << "cannot reference the entry-point component: '" << entryPointName
           << "'.";

  // Verify the instance result ports with those of its referenced component.
  hw::ModulePortInfo primitivePorts = referencedPrimitive.getPortList();
  size_t numPorts = primitivePorts.size();

  size_t numResults = instance.getNumResults();
  if (numResults != numPorts)
    return instance.emitOpError()
           << "has a wrong number of results; expected: " << numPorts
           << " but got " << numResults;

  // Verify parameters match up
  ArrayAttr modParameters = referencedPrimitive.getParameters();
  ArrayAttr parameters = instance.getParameters().value_or(ArrayAttr());
  size_t numExpected = modParameters.size();
  size_t numParams = parameters.size();
  if (numParams != numExpected)
    return instance.emitOpError()
           << "has the wrong number of parameters; expected: " << numExpected
           << " but got " << numParams;

  for (size_t i = 0; i != numExpected; ++i) {
    auto param = parameters[i].cast<circt::hw::ParamDeclAttr>();
    auto modParam = modParameters[i].cast<circt::hw::ParamDeclAttr>();

    auto paramName = param.getName();
    if (paramName != modParam.getName())
      return instance.emitOpError()
             << "parameter #" << i << " should have name " << modParam.getName()
             << " but has name " << paramName;

    if (param.getType() != modParam.getType())
      return instance.emitOpError()
             << "parameter " << paramName << " should have type "
             << modParam.getType() << " but has type " << param.getType();

    // All instance parameters must have a value.  Specify the same value as
    // a module's default value if you want the default.
    if (!param.getValue())
      return instance.emitOpError("parameter ")
             << paramName << " must have a value";
  }

  for (size_t i = 0; i != numResults; ++i) {
    auto resultType = instance.getResult(i).getType();
    auto expectedType = primitivePorts.at(i).type;
    auto replacedType = hw::evaluateParametricType(
        instance.getLoc(), instance.getParametersAttr(), expectedType);
    if (failed(replacedType))
      return failure();
    if (resultType == replacedType)
      continue;
    return instance.emitOpError()
           << "result type for " << primitivePorts.at(i).name << " must be "
           << expectedType << ", but got " << resultType;
  }
  return success();
}

LogicalResult
PrimitiveOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *op = *this;
  auto module = op->getParentOfType<ModuleOp>();
  Operation *referencedPrimitive =
      symbolTable.lookupNearestSymbolFrom(module, getPrimitiveNameAttr());
  if (referencedPrimitive == nullptr)
    return emitError() << "referencing primitive: '" << getPrimitiveName()
                       << "', which does not exist.";

  Operation *shadowedPrimitiveName =
      symbolTable.lookupNearestSymbolFrom(module, getSymNameAttr());
  if (shadowedPrimitiveName != nullptr)
    return emitError() << "instance symbol: '" << instanceName()
                       << "' is already a symbol for another primitive.";

  // Verify the referenced primitive is not instantiating itself.
  auto parentPrimitive = op->getParentOfType<hw::HWModuleExternOp>();
  if (parentPrimitive == referencedPrimitive)
    return emitError() << "recursive instantiation of its parent primitive: '"
                       << getPrimitiveName() << "'";

  assert(isa<hw::HWModuleExternOp>(referencedPrimitive) &&
         "Should be a HardwareModuleExternOp.");

  return verifyPrimitiveOpType(*this,
                               cast<hw::HWModuleExternOp>(referencedPrimitive));
}

/// Provide meaningful names to the result values of an PrimitiveOp.
void PrimitiveOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> PrimitiveOp::portNames() {
  SmallVector<StringRef> portNames;
  auto ports = getReferencedPrimitive().getPortList();
  for (auto port : ports)
    portNames.push_back(port.name.getValue());

  return portNames;
}

Direction convertHWDirectionToCalyx(hw::ModulePort::Direction direction) {
  switch (direction) {
  case hw::ModulePort::Direction::Input:
    return Direction::Input;
  case hw::ModulePort::Direction::Output:
    return Direction::Output;
  case hw::ModulePort::Direction::InOut:
    llvm_unreachable("InOut ports not supported by Calyx");
  }
  llvm_unreachable("Impossible port type");
}

SmallVector<Direction> PrimitiveOp::portDirections() {
  SmallVector<Direction> portDirections;
  auto ports = getReferencedPrimitive().getPortList();
  for (hw::PortInfo port : ports)
    portDirections.push_back(convertHWDirectionToCalyx(port.dir));
  return portDirections;
}

bool PrimitiveOp::isCombinational() { return false; }

/// Returns a new DictionaryAttr containing only the calyx dialect attrs
/// in the input DictionaryAttr. Also strips the 'calyx.' prefix from these
/// attrs.
static DictionaryAttr cleanCalyxPortAttrs(OpBuilder builder,
                                          DictionaryAttr dict) {
  if (!dict) {
    return dict;
  }
  llvm::SmallVector<NamedAttribute> attrs;
  for (NamedAttribute attr : dict) {
    Dialect *dialect = attr.getNameDialect();
    if (dialect == nullptr || !isa<CalyxDialect>(*dialect))
      continue;
    StringRef name = attr.getName().strref();
    StringAttr newName = builder.getStringAttr(std::get<1>(name.split(".")));
    attr.setName(newName);
    attrs.push_back(attr);
  }
  return builder.getDictionaryAttr(attrs);
}

// Grabs calyx port attributes from the HWModuleExternOp arg/result attributes.
SmallVector<DictionaryAttr> PrimitiveOp::portAttributes() {
  SmallVector<DictionaryAttr> portAttributes;
  OpBuilder builder(getContext());
  hw::HWModuleExternOp prim = getReferencedPrimitive();
  for (size_t i = 0, e = prim.getNumArguments(); i != e; ++i) {
    DictionaryAttr dict = cleanCalyxPortAttrs(builder, prim.getArgAttrDict(i));
    portAttributes.push_back(dict);
  }
  for (size_t i = 0, e = prim.getNumResults(); i != e; ++i) {
    DictionaryAttr dict =
        cleanCalyxPortAttrs(builder, prim.getResultAttrDict(i));
    portAttributes.push_back(dict);
  }
  return portAttributes;
}

/// Parse an parameter list if present. Same format as HW dialect.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
///
static ParseResult parseParameterList(OpAsmParser &parser,
                                      SmallVector<Attribute> &parameters) {

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() {
        std::string name;
        Type type;
        Attribute value;

        if (parser.parseKeywordOrString(&name) || parser.parseColonType(type))
          return failure();

        // Parse the default value if present.
        if (succeeded(parser.parseOptionalEqual())) {
          if (parser.parseAttribute(value, type))
            return failure();
        }

        auto &builder = parser.getBuilder();
        parameters.push_back(hw::ParamDeclAttr::get(
            builder.getContext(), builder.getStringAttr(name), type, value));
        return success();
      });
}

/// Shim to also use this for the InstanceOp custom parser.
static ParseResult parseParameterList(OpAsmParser &parser,
                                      ArrayAttr &parameters) {
  SmallVector<Attribute> parseParameters;
  if (failed(parseParameterList(parser, parseParameters)))
    return failure();

  parameters = ArrayAttr::get(parser.getContext(), parseParameters);

  return success();
}

/// Print a parameter list for a module or instance. Same format as HW dialect.
static void printParameterList(OpAsmPrinter &p, Operation *op,
                               ArrayAttr parameters) {
  if (parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<hw::ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}

//===----------------------------------------------------------------------===//
// GroupGoOp
//===----------------------------------------------------------------------===//

LogicalResult GroupGoOp::verify() { return verifyNotComplexSource(*this); }

/// Provide meaningful names to the result value of a GroupGoOp.
void GroupGoOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto parent = (*this)->getParentOfType<GroupOp>();
  StringRef name = parent.getSymName();
  std::string resultName = name.str() + ".go";
  setNameFn(getResult(), resultName);
}

void GroupGoOp::print(OpAsmPrinter &p) { printGroupPort(p, *this); }

ParseResult GroupGoOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseGroupPort(parser, result))
    return failure();

  result.addTypes(parser.getBuilder().getI1Type());
  return success();
}

//===----------------------------------------------------------------------===//
// GroupDoneOp
//===----------------------------------------------------------------------===//

LogicalResult GroupDoneOp::verify() {
  Operation *srcOp = getSrc().getDefiningOp();
  Value optionalGuard = getGuard();
  Operation *guardOp = optionalGuard ? optionalGuard.getDefiningOp() : nullptr;
  bool noGuard = (guardOp == nullptr);

  if (srcOp == nullptr)
    // This is a port of the parent component.
    return success();

  if (isa<hw::ConstantOp>(srcOp) && (noGuard || isa<hw::ConstantOp>(guardOp)))
    return emitOpError() << "with constant source"
                         << (noGuard ? "" : " and constant guard")
                         << ". This should be a combinational group.";

  return verifyNotComplexSource(*this);
}

void GroupDoneOp::print(OpAsmPrinter &p) { printGroupPort(p, *this); }

ParseResult GroupDoneOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseGroupPort(parser, result);
}

//===----------------------------------------------------------------------===//
// RegisterOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a RegisterOp.
void RegisterOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> RegisterOp::portNames() {
  return {"in", "write_en", "clk", "reset", "out", "done"};
}

SmallVector<Direction> RegisterOp::portDirections() {
  return {Input, Input, Input, Input, Output, Output};
}

SmallVector<DictionaryAttr> RegisterOp::portAttributes() {
  MLIRContext *context = getContext();
  IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);
  NamedAttrList writeEn, clk, reset, done;
  writeEn.append("go", isSet);
  clk.append("clk", isSet);
  reset.append("reset", isSet);
  done.append("done", isSet);
  return {
      DictionaryAttr::get(context),   // In
      writeEn.getDictionary(context), // Write enable
      clk.getDictionary(context),     // Clk
      reset.getDictionary(context),   // Reset
      DictionaryAttr::get(context),   // Out
      done.getDictionary(context)     // Done
  };
}

bool RegisterOp::isCombinational() { return false; }

//===----------------------------------------------------------------------===//
// MemoryOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a MemoryOp.
void MemoryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> MemoryOp::portNames() {
  SmallVector<StringRef> portNames;
  for (size_t i = 0, e = getAddrSizes().size(); i != e; ++i) {
    auto nameAttr =
        StringAttr::get(this->getContext(), "addr" + std::to_string(i));
    portNames.push_back(nameAttr.getValue());
  }
  portNames.append({"write_data", "write_en", "clk", "read_data", "done"});
  return portNames;
}

SmallVector<Direction> MemoryOp::portDirections() {
  SmallVector<Direction> portDirections;
  for (size_t i = 0, e = getAddrSizes().size(); i != e; ++i)
    portDirections.push_back(Input);
  portDirections.append({Input, Input, Input, Output, Output});
  return portDirections;
}

SmallVector<DictionaryAttr> MemoryOp::portAttributes() {
  SmallVector<DictionaryAttr> portAttributes;
  MLIRContext *context = getContext();
  for (size_t i = 0, e = getAddrSizes().size(); i != e; ++i)
    portAttributes.push_back(DictionaryAttr::get(context)); // Addresses

  // Use a boolean to indicate this attribute is used.
  IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);
  NamedAttrList writeEn, clk, reset, done;
  writeEn.append("go", isSet);
  clk.append("clk", isSet);
  done.append("done", isSet);
  portAttributes.append({DictionaryAttr::get(context),   // In
                         writeEn.getDictionary(context), // Write enable
                         clk.getDictionary(context),     // Clk
                         DictionaryAttr::get(context),   // Out
                         done.getDictionary(context)}    // Done
  );
  return portAttributes;
}

void MemoryOp::build(OpBuilder &builder, OperationState &state,
                     StringRef instanceName, int64_t width,
                     ArrayRef<int64_t> sizes, ArrayRef<int64_t> addrSizes) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(instanceName));
  state.addAttribute("width", builder.getI64IntegerAttr(width));
  state.addAttribute("sizes", builder.getI64ArrayAttr(sizes));
  state.addAttribute("addrSizes", builder.getI64ArrayAttr(addrSizes));
  SmallVector<Type> types;
  for (int64_t size : addrSizes)
    types.push_back(builder.getIntegerType(size)); // Addresses
  types.push_back(builder.getIntegerType(width));  // Write data
  types.push_back(builder.getI1Type());            // Write enable
  types.push_back(builder.getI1Type());            // Clk
  types.push_back(builder.getIntegerType(width));  // Read data
  types.push_back(builder.getI1Type());            // Done
  state.addTypes(types);
}

LogicalResult MemoryOp::verify() {
  ArrayRef<Attribute> opSizes = getSizes().getValue();
  ArrayRef<Attribute> opAddrSizes = getAddrSizes().getValue();
  size_t numDims = getSizes().size();
  size_t numAddrs = getAddrSizes().size();
  if (numDims != numAddrs)
    return emitOpError("mismatched number of dimensions (")
           << numDims << ") and address sizes (" << numAddrs << ")";

  size_t numExtraPorts = 5; // write data/enable, clk, and read data/done.
  if (getNumResults() != numAddrs + numExtraPorts)
    return emitOpError("incorrect number of address ports, expected ")
           << numAddrs;

  for (size_t i = 0; i < numDims; ++i) {
    int64_t size = opSizes[i].cast<IntegerAttr>().getInt();
    int64_t addrSize = opAddrSizes[i].cast<IntegerAttr>().getInt();
    if (llvm::Log2_64_Ceil(size) > addrSize)
      return emitOpError("address size (")
             << addrSize << ") for dimension " << i
             << " can't address the entire range (" << size << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SeqMemoryOp
//===----------------------------------------------------------------------===//

/// Provide meaningful names to the result values of a SeqMemoryOp.
void SeqMemoryOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  getCellAsmResultNames(setNameFn, *this, this->portNames());
}

SmallVector<StringRef> SeqMemoryOp::portNames() {
  SmallVector<StringRef> portNames;
  for (size_t i = 0, e = getAddrSizes().size(); i != e; ++i) {
    auto nameAttr =
        StringAttr::get(this->getContext(), "addr" + std::to_string(i));
    portNames.push_back(nameAttr.getValue());
  }
  portNames.append({"write_data", "write_en", "write_done", "clk", "read_data",
                    "read_en", "read_done"});
  return portNames;
}

SmallVector<Direction> SeqMemoryOp::portDirections() {
  SmallVector<Direction> portDirections;
  for (size_t i = 0, e = getAddrSizes().size(); i != e; ++i)
    portDirections.push_back(Input);
  portDirections.append({Input, Input, Output, Input, Output, Input, Output});
  return portDirections;
}

SmallVector<DictionaryAttr> SeqMemoryOp::portAttributes() {
  SmallVector<DictionaryAttr> portAttributes;
  MLIRContext *context = getContext();
  for (size_t i = 0, e = getAddrSizes().size(); i != e; ++i)
    portAttributes.push_back(DictionaryAttr::get(context)); // Addresses

  OpBuilder builder(context);
  // Use a boolean to indicate this attribute is used.
  IntegerAttr isSet = IntegerAttr::get(builder.getIndexType(), 1);
  IntegerAttr isTwo = IntegerAttr::get(builder.getIndexType(), 2);
  NamedAttrList writeEn, writeDone, clk, reset, readEn, readDone;
  writeEn.append("go", isSet);
  writeDone.append("done", isSet);
  clk.append("clk", isSet);
  readEn.append("go", isTwo);
  readDone.append("done", isTwo);
  portAttributes.append({DictionaryAttr::get(context),     // Write Data
                         writeEn.getDictionary(context),   // Write enable
                         writeDone.getDictionary(context), // Write done
                         clk.getDictionary(context),       // Clk
                         DictionaryAttr::get(context),     // Out
                         readEn.getDictionary(context),    // Read enable
                         readDone.getDictionary(context)}  // Read done
  );
  return portAttributes;
}

void SeqMemoryOp::build(OpBuilder &builder, OperationState &state,
                        StringRef instanceName, int64_t width,
                        ArrayRef<int64_t> sizes, ArrayRef<int64_t> addrSizes) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(instanceName));
  state.addAttribute("width", builder.getI64IntegerAttr(width));
  state.addAttribute("sizes", builder.getI64ArrayAttr(sizes));
  state.addAttribute("addrSizes", builder.getI64ArrayAttr(addrSizes));
  SmallVector<Type> types;
  for (int64_t size : addrSizes)
    types.push_back(builder.getIntegerType(size)); // Addresses
  types.push_back(builder.getIntegerType(width));  // Write data
  types.push_back(builder.getI1Type());            // Write enable
  types.push_back(builder.getI1Type());            // Write done
  types.push_back(builder.getI1Type());            // Clk
  types.push_back(builder.getIntegerType(width));  // Read data
  types.push_back(builder.getI1Type());            // Read enable
  types.push_back(builder.getI1Type());            // Read done
  state.addTypes(types);
}

LogicalResult SeqMemoryOp::verify() {
  ArrayRef<Attribute> opSizes = getSizes().getValue();
  ArrayRef<Attribute> opAddrSizes = getAddrSizes().getValue();
  size_t numDims = getSizes().size();
  size_t numAddrs = getAddrSizes().size();
  if (numDims != numAddrs)
    return emitOpError("mismatched number of dimensions (")
           << numDims << ") and address sizes (" << numAddrs << ")";

  size_t numExtraPorts =
      7; // write data/enable/done, clk, and read data/enable/done.
  if (getNumResults() != numAddrs + numExtraPorts)
    return emitOpError("incorrect number of address ports, expected ")
           << numAddrs;

  for (size_t i = 0; i < numDims; ++i) {
    int64_t size = opSizes[i].cast<IntegerAttr>().getInt();
    int64_t addrSize = opAddrSizes[i].cast<IntegerAttr>().getInt();
    if (llvm::Log2_64_Ceil(size) > addrSize)
      return emitOpError("address size (")
             << addrSize << ") for dimension " << i
             << " can't address the entire range (" << size << ")";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EnableOp
//===----------------------------------------------------------------------===//
LogicalResult EnableOp::verify() {
  auto component = (*this)->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();
  StringRef name = getGroupName();

  auto groupOp = wiresOp.lookupSymbol<GroupInterface>(name);
  if (!groupOp)
    return emitOpError() << "with group '" << name
                         << "', which does not exist.";

  if (isa<CombGroupOp>(groupOp))
    return emitOpError() << "with group '" << name
                         << "', which is a combinational group.";

  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

LogicalResult IfOp::verify() {
  std::optional<StringRef> optGroupName = getGroupName();
  if (!optGroupName) {
    // No combinational group was provided.
    return success();
  }
  auto component = (*this)->getParentOfType<ComponentOp>();
  WiresOp wiresOp = component.getWiresOp();
  StringRef groupName = *optGroupName;
  auto groupOp = wiresOp.lookupSymbol<GroupInterface>(groupName);
  if (!groupOp)
    return emitOpError() << "with group '" << groupName
                         << "', which does not exist.";

  if (isa<GroupOp>(groupOp))
    return emitOpError() << "with group '" << groupName
                         << "', which is not a combinational group.";

  if (failed(groupOp.drivesPort(getCond())))
    return emitError() << "with conditional op: '"
                       << valueName(component, getCond())
                       << "' expected to be driven from group: '" << groupName
                       << "' but no driver was found.";

  return success();
}

/// Returns the last EnableOp within the child tree of 'parentSeqOp' or
/// `parentStaticSeqOp.` If no EnableOp was found (e.g. a "calyx.par" operation
/// is present), returns None.
template <typename OpTy>
static std::optional<EnableOp> getLastEnableOp(OpTy parent) {
  static_assert(IsAny<OpTy, SeqOp, StaticSeqOp>(),
                "Should be a StaticSeqOp or SeqOp.");
  auto &lastOp = parent.getBodyBlock()->back();
  if (auto enableOp = dyn_cast<EnableOp>(lastOp))
    return enableOp;
  if (auto seqOp = dyn_cast<SeqOp>(lastOp))
    return getLastEnableOp(seqOp);
  if (auto staticSeqOp = dyn_cast<StaticSeqOp>(lastOp))
    return getLastEnableOp(staticSeqOp);

  return std::nullopt;
}

/// Returns a mapping of {enabled Group name, EnableOp} for all EnableOps within
/// the immediate ParOp's body.
template <typename OpTy>
static llvm::StringMap<EnableOp> getAllEnableOpsInImmediateBody(OpTy parent) {
  static_assert(IsAny<OpTy, ParOp, StaticParOp>(),
                "Should be a StaticParOp or ParOp.");

  llvm::StringMap<EnableOp> enables;
  Block *body = parent.getBodyBlock();
  for (EnableOp op : body->getOps<EnableOp>())
    enables.insert(std::pair(op.getGroupName(), op));

  return enables;
}

/// Checks preconditions for the common tail pattern. This canonicalization is
/// stringent about not entering nested control operations, as this may cause
/// unintentional changes in behavior.
/// We only look for two cases: (1) both regions are ParOps, and
/// (2) both regions are SeqOps. The case when these are different, e.g. ParOp
/// and SeqOp, will only produce less optimal code, or even worse, change the
/// behavior.
template <typename IfOpTy, typename TailOpTy>
static bool hasCommonTailPatternPreConditions(IfOpTy op) {
  static_assert(IsAny<TailOpTy, SeqOp, ParOp, StaticSeqOp, StaticParOp>(),
                "Should be a SeqOp, ParOp, StaticSeqOp, or StaticParOp.");
  static_assert(IsAny<IfOpTy, IfOp, StaticIfOp>(),
                "Should be a IfOp or StaticIfOp.");

  if (!op.thenBodyExists() || !op.elseBodyExists())
    return false;
  if (op.getThenBody()->empty() || op.getElseBody()->empty())
    return false;

  Block *thenBody = op.getThenBody(), *elseBody = op.getElseBody();
  return isa<TailOpTy>(thenBody->front()) && isa<TailOpTy>(elseBody->front());
}

///                                         seq {
///   if %a with @G {                         if %a with @G {
///     seq { ... calyx.enable @A }             seq { ... }
///   else {                          ->      } else {
///     seq { ... calyx.enable @A }             seq { ... }
///   }                                       }
///                                           calyx.enable @A
///                                         }
template <typename IfOpTy, typename SeqOpTy>
static LogicalResult commonTailPatternWithSeq(IfOpTy ifOp,
                                              PatternRewriter &rewriter) {
  static_assert(IsAny<IfOpTy, IfOp, StaticIfOp>(),
                "Should be an IfOp or StaticIfOp.");
  static_assert(IsAny<SeqOpTy, SeqOp, StaticSeqOp>(),
                "Branches should be checking for an SeqOp or StaticSeqOp");
  if (!hasCommonTailPatternPreConditions<IfOpTy, SeqOpTy>(ifOp))
    return failure();
  auto thenControl = cast<SeqOpTy>(ifOp.getThenBody()->front()),
       elseControl = cast<SeqOpTy>(ifOp.getElseBody()->front());

  std::optional<EnableOp> lastThenEnableOp = getLastEnableOp(thenControl),
                          lastElseEnableOp = getLastEnableOp(elseControl);

  if (!lastThenEnableOp || !lastElseEnableOp)
    return failure();
  if (lastThenEnableOp->getGroupName() != lastElseEnableOp->getGroupName())
    return failure();

  // Place the IfOp and pulled EnableOp inside a sequential region, in case
  // this IfOp is nested in a ParOp. This avoids unintentionally
  // parallelizing the pulled out EnableOps.
  rewriter.setInsertionPointAfter(ifOp);
  SeqOpTy seqOp = rewriter.create<SeqOpTy>(ifOp.getLoc());
  Block *body = seqOp.getBodyBlock();
  ifOp->remove();
  body->push_back(ifOp);
  rewriter.setInsertionPointToEnd(body);
  rewriter.create<EnableOp>(seqOp.getLoc(), lastThenEnableOp->getGroupName());

  // Erase the common EnableOp from the Then and Else regions.
  rewriter.eraseOp(*lastThenEnableOp);
  rewriter.eraseOp(*lastElseEnableOp);
  return success();
}

///    if %a with @G {              par {
///      par {                        if %a with @G {
///        ...                          par { ... }
///        calyx.enable @A            } else {
///        calyx.enable @B    ->        par { ... }
///      }                            }
///    } else {                       calyx.enable @A
///      par {                        calyx.enable @B
///        ...                      }
///        calyx.enable @A
///        calyx.enable @B
///      }
///    }
template <typename OpTy, typename ParOpTy>
static LogicalResult commonTailPatternWithPar(OpTy controlOp,
                                              PatternRewriter &rewriter) {
  static_assert(IsAny<OpTy, IfOp, StaticIfOp>(),
                "Should be an IfOp or StaticIfOp.");
  static_assert(IsAny<ParOpTy, ParOp, StaticParOp>(),
                "Branches should be checking for an ParOp or StaticParOp");
  if (!hasCommonTailPatternPreConditions<OpTy, ParOpTy>(controlOp))
    return failure();
  auto thenControl = cast<ParOpTy>(controlOp.getThenBody()->front()),
       elseControl = cast<ParOpTy>(controlOp.getElseBody()->front());

  llvm::StringMap<EnableOp> a = getAllEnableOpsInImmediateBody(thenControl),
                            b = getAllEnableOpsInImmediateBody(elseControl);
  // Compute the intersection between `A` and `B`.
  SmallVector<StringRef> groupNames;
  for (auto aIndex = a.begin(); aIndex != a.end(); ++aIndex) {
    StringRef groupName = aIndex->getKey();
    auto bIndex = b.find(groupName);
    if (bIndex == b.end())
      continue;
    // This is also an element in B.
    groupNames.push_back(groupName);
    // Since these are being pulled out, erase them.
    rewriter.eraseOp(aIndex->getValue());
    rewriter.eraseOp(bIndex->getValue());
  }

  // Place the IfOp and EnableOp(s) inside a parallel region, in case this
  // IfOp is nested in a SeqOp. This avoids unintentionally sequentializing
  // the pulled out EnableOps.
  rewriter.setInsertionPointAfter(controlOp);

  ParOpTy parOp = rewriter.create<ParOpTy>(controlOp.getLoc());
  Block *body = parOp.getBodyBlock();
  controlOp->remove();
  body->push_back(controlOp);
  // Pull out the intersection between these two sets, and erase their
  // counterparts in the Then and Else regions.
  rewriter.setInsertionPointToEnd(body);
  for (StringRef groupName : groupNames)
    rewriter.create<EnableOp>(parOp.getLoc(), groupName);

  return success();
}

/// This pattern checks for one of two cases that will lead to IfOp deletion:
/// (1) Then and Else bodies are both empty.
/// (2) Then body is empty and Else body does not exist.
struct EmptyIfBody : mlir::OpRewritePattern<IfOp> {
  using mlir::OpRewritePattern<IfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp.getThenBody()->empty())
      return failure();
    if (ifOp.elseBodyExists() && !ifOp.getElseBody()->empty())
      return failure();

    eraseControlWithGroupAndConditional(ifOp, rewriter);

    return success();
  }
};

void IfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<EmptyIfBody>(context);
  patterns.add(commonTailPatternWithPar<IfOp, ParOp>);
  patterns.add(commonTailPatternWithSeq<IfOp, SeqOp>);
}

//===----------------------------------------------------------------------===//
// StaticIfOp
//===----------------------------------------------------------------------===//
LogicalResult StaticIfOp::verify() {
  if (elseBodyExists()) {
    auto *elseBod = getElseBody();
    auto &elseOps = elseBod->getOperations();
    // should only have one Operation, static, in the else branch
    for (Operation &op : elseOps) {
      if (!isStaticControl(&op)) {
        return op.emitOpError(
            "static if's else branch has non static control within it");
      }
    }
  }

  auto *thenBod = getThenBody();
  auto &thenOps = thenBod->getOperations();
  for (Operation &op : thenOps) {
    // should only have one, static, Operation in the then branch
    if (!isStaticControl(&op)) {
      return op.emitOpError(
          "static if's then branch has non static control within it");
    }
  }

  return success();
}

/// This pattern checks for one of two cases that will lead to StaticIfOp
/// deletion: (1) Then and Else bodies are both empty. (2) Then body is empty
/// and Else body does not exist.
struct EmptyStaticIfBody : mlir::OpRewritePattern<StaticIfOp> {
  using mlir::OpRewritePattern<StaticIfOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(StaticIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (!ifOp.getThenBody()->empty())
      return failure();
    if (ifOp.elseBodyExists() && !ifOp.getElseBody()->empty())
      return failure();

    eraseControlWithConditional(ifOp, rewriter);

    return success();
  }
};

void StaticIfOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EmptyStaticIfBody>(context);
  patterns.add(commonTailPatternWithPar<StaticIfOp, StaticParOp>);
  patterns.add(commonTailPatternWithSeq<StaticIfOp, StaticSeqOp>);
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//
LogicalResult WhileOp::verify() {
  auto component = (*this)->getParentOfType<ComponentOp>();
  auto wiresOp = component.getWiresOp();

  std::optional<StringRef> optGroupName = getGroupName();
  if (!optGroupName) {
    /// No combinational group was provided
    return success();
  }
  StringRef groupName = *optGroupName;
  auto groupOp = wiresOp.lookupSymbol<GroupInterface>(groupName);
  if (!groupOp)
    return emitOpError() << "with group '" << groupName
                         << "', which does not exist.";

  if (isa<GroupOp>(groupOp))
    return emitOpError() << "with group '" << groupName
                         << "', which is not a combinational group.";

  if (failed(groupOp.drivesPort(getCond())))
    return emitError() << "conditional op: '" << valueName(component, getCond())
                       << "' expected to be driven from group: '" << groupName
                       << "' but no driver was found.";

  return success();
}

LogicalResult WhileOp::canonicalize(WhileOp whileOp,
                                    PatternRewriter &rewriter) {
  if (whileOp.getBodyBlock()->empty()) {
    eraseControlWithGroupAndConditional(whileOp, rewriter);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// StaticRepeatOp
//===----------------------------------------------------------------------===//
LogicalResult StaticRepeatOp::verify() {
  for (auto &&bodyOp : (*this).getRegion().front()) {
    // there should only be one bodyOp for each StaticRepeatOp
    if (!isStaticControl(&bodyOp)) {
      return bodyOp.emitOpError(
          "static repeat has non static control within it");
    }
  }

  return success();
}

template <typename OpTy>
static LogicalResult zeroRepeat(OpTy op, PatternRewriter &rewriter) {
  static_assert(IsAny<OpTy, RepeatOp, StaticRepeatOp>(),
                "Should be a RepeatOp or StaticPRepeatOp");
  if (op.getCount() == 0) {
    Block *controlBody = op.getBodyBlock();
    for (auto &op : make_early_inc_range(*controlBody))
      op.erase();

    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

void StaticRepeatOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add(emptyControl<StaticRepeatOp>);
  patterns.add(zeroRepeat<StaticRepeatOp>);
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//
void RepeatOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add(emptyControl<RepeatOp>);
  patterns.add(zeroRepeat<RepeatOp>);
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

// Parse the parameter list of invoke.
static ParseResult
parseParameterList(OpAsmParser &parser, OperationState &result,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ports,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
                   SmallVectorImpl<Attribute> &portNames,
                   SmallVectorImpl<Attribute> &inputNames,
                   SmallVectorImpl<Type> &types) {
  OpAsmParser::UnresolvedOperand port;
  OpAsmParser::UnresolvedOperand input;
  Type type;
  auto parseParameter = [&]() -> ParseResult {
    if (parser.parseOperand(port) || parser.parseEqual() ||
        parser.parseOperand(input))
      return failure();
    ports.push_back(port);
    portNames.push_back(StringAttr::get(parser.getContext(), port.name));
    inputs.push_back(input);
    inputNames.push_back(StringAttr::get(parser.getContext(), input.name));
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseParameter))
    return failure();
  if (parser.parseArrow())
    return failure();
  auto parseType = [&]() -> ParseResult {
    if (parser.parseType(type))
      return failure();
    types.push_back(type);
    return success();
  };
  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseType);
}

ParseResult InvokeOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr componentName;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> ports;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  SmallVector<Attribute> portNames;
  SmallVector<Attribute> inputNames;
  SmallVector<Type, 4> types;
  if (parser.parseSymbolName(componentName))
    return failure();
  FlatSymbolRefAttr callee = FlatSymbolRefAttr::get(componentName);
  SMLoc loc = parser.getCurrentLocation();
  result.addAttribute("callee", callee);
  if (parseParameterList(parser, result, ports, inputs, portNames, inputNames,
                         types))
    return failure();
  if (parser.resolveOperands(ports, types, loc, result.operands))
    return failure();
  if (parser.resolveOperands(inputs, types, loc, result.operands))
    return failure();
  result.addAttribute("portNames",
                      ArrayAttr::get(parser.getContext(), portNames));
  result.addAttribute("inputNames",
                      ArrayAttr::get(parser.getContext(), inputNames));
  return success();
}

void InvokeOp::print(OpAsmPrinter &p) {
  p << " @" << getCallee() << "(";
  auto ports = getPorts();
  auto inputs = getInputs();
  llvm::interleaveComma(llvm::zip(ports, inputs), p, [&](auto arg) {
    p << std::get<0>(arg) << " = " << std::get<1>(arg);
  });
  p << ") -> (";
  llvm::interleaveComma(ports, p, [&](auto port) { p << port.getType(); });
  p << ")";
}

// Check the direction of one of the ports in one of the connections of an
// InvokeOp.
static LogicalResult verifyInvokeOpValue(InvokeOp &op, Value &value,
                                         bool isDestination) {
  if (isPort(value))
    return verifyPortDirection(op, value, isDestination);
  return success();
}

// Checks if the value comes from complex logic.
static LogicalResult verifyComplexLogic(InvokeOp &op, Value &value) {
  // Refer to the above function verifyNotComplexSource for its role.
  Operation *operation = value.getDefiningOp();
  if (operation == nullptr)
    return success();
  if (auto *dialect = operation->getDialect(); isa<comb::CombDialect>(dialect))
    return failure();
  return success();
}

// Get the go port of the invoked component.
Value InvokeOp::getInstGoValue() {
  ComponentOp componentOp = (*this)->getParentOfType<ComponentOp>();
  Operation *operation = componentOp.lookupSymbol(getCallee());
  Value ret = nullptr;
  llvm::TypeSwitch<Operation *>(operation)
      .Case<RegisterOp>([&](auto op) { ret = operation->getResult(1); })
      .Case<MemoryOp, DivSPipeLibOp, DivUPipeLibOp, MultPipeLibOp,
            RemSPipeLibOp, RemUPipeLibOp>(
          [&](auto op) { ret = operation->getResult(2); })
      .Case<InstanceOp>([&](auto op) {
        auto portInfo = op.getReferencedComponent().getPortInfo();
        for (auto [portInfo, res] :
             llvm::zip(portInfo, operation->getResults())) {
          if (portInfo.hasAttribute("go"))
            ret = res;
        }
      })
      .Case<PrimitiveOp>([&](auto op) {
        auto moduleExternOp = op.getReferencedPrimitive();
        auto argAttrs = moduleExternOp.getArgAttrsAttr();
        for (auto [attr, res] : llvm::zip(argAttrs, op.getResults())) {
          if (DictionaryAttr dictAttr = dyn_cast<DictionaryAttr>(attr)) {
            if (!dictAttr.empty()) {
              if (dictAttr.begin()->getName().getValue() == "calyx.go")
                ret = res;
            }
          }
        }
      });
  return ret;
}

// Get the done port of the invoked component.
Value InvokeOp::getInstDoneValue() {
  ComponentOp componentOp = (*this)->getParentOfType<ComponentOp>();
  Operation *operation = componentOp.lookupSymbol(getCallee());
  Value ret = nullptr;
  llvm::TypeSwitch<Operation *>(operation)
      .Case<RegisterOp, MemoryOp, DivSPipeLibOp, DivUPipeLibOp, MultPipeLibOp,
            RemSPipeLibOp, RemUPipeLibOp>([&](auto op) {
        size_t doneIdx = operation->getResults().size() - 1;
        ret = operation->getResult(doneIdx);
      })
      .Case<InstanceOp>([&](auto op) {
        InstanceOp instanceOp = cast<InstanceOp>(operation);
        auto portInfo = instanceOp.getReferencedComponent().getPortInfo();
        for (auto [portInfo, res] :
             llvm::zip(portInfo, operation->getResults())) {
          if (portInfo.hasAttribute("done"))
            ret = res;
        }
      })
      .Case<PrimitiveOp>([&](auto op) {
        PrimitiveOp primOp = cast<PrimitiveOp>(operation);
        auto moduleExternOp = primOp.getReferencedPrimitive();
        auto resAttrs = moduleExternOp.getResAttrsAttr();
        for (auto [attr, res] : llvm::zip(resAttrs, primOp.getResults())) {
          if (DictionaryAttr dictAttr = dyn_cast<DictionaryAttr>(attr)) {
            if (!dictAttr.empty()) {
              if (dictAttr.begin()->getName().getValue() == "calyx.done")
                ret = res;
            }
          }
        }
      });
  return ret;
}

// A helper function that gets the number of go or done ports in
// hw.module.extern.
static size_t
getHwModuleExtGoOrDonePortNumber(hw::HWModuleExternOp &moduleExternOp,
                                 bool isGo) {
  size_t ret = 0;
  std::string str = isGo ? "calyx.go" : "calyx.done";
  for (Attribute attr : moduleExternOp.getArgAttrsAttr()) {
    if (DictionaryAttr dictAttr = dyn_cast<DictionaryAttr>(attr)) {
      ret = llvm::count_if(dictAttr, [&](NamedAttribute iter) {
        return iter.getName().getValue() == str;
      });
    }
  }
  return ret;
}

LogicalResult InvokeOp::verify() {
  ComponentOp componentOp = (*this)->getParentOfType<ComponentOp>();
  StringRef callee = getCallee();
  Operation *operation = componentOp.lookupSymbol(callee);
  // The referenced symbol does not exist.
  if (!operation)
    return emitOpError() << "with instance '@" << callee
                         << "', which does not exist.";
  // The argument list of invoke is empty.
  if (getInputs().empty())
    return emitOpError() << "'@" << callee
                         << "' has zero input and output port connections; "
                            "expected at least one.";
  size_t goPortNum = 0, donePortNum = 0;
  // They both have a go port and a done port, but the "go" port for
  // registers and memrey should be "write_en" port.
  llvm::TypeSwitch<Operation *>(operation)
      .Case<RegisterOp, DivSPipeLibOp, DivUPipeLibOp, MemoryOp, MultPipeLibOp,
            RemSPipeLibOp, RemUPipeLibOp>(
          [&](auto op) { goPortNum = 1, donePortNum = 1; })
      .Case<InstanceOp>([&](auto op) {
        auto portInfo = op.getReferencedComponent().getPortInfo();
        for (PortInfo info : portInfo) {
          if (info.hasAttribute("go"))
            ++goPortNum;
          if (info.hasAttribute("done"))
            ++donePortNum;
        }
      })
      .Case<PrimitiveOp>([&](auto op) {
        auto moduleExternOp = op.getReferencedPrimitive();
        // Get the number of go ports and done ports by their attrubutes.
        goPortNum = getHwModuleExtGoOrDonePortNumber(moduleExternOp, true);
        donePortNum = getHwModuleExtGoOrDonePortNumber(moduleExternOp, false);
      });
  // If the number of go ports and done ports is wrong.
  if (goPortNum != 1 && donePortNum != 1)
    return emitOpError()
           << "'@" << callee << "'"
           << " is a combinational component and cannot be invoked, which must "
              "have single go port and single done port.";

  auto ports = getPorts();
  auto inputs = getInputs();
  // We have verified earlier that the instance has a go and a done port.
  Value goValue = getInstGoValue();
  Value doneValue = getInstDoneValue();
  for (auto [port, input, portName, inputName] :
       llvm::zip(ports, inputs, getPortNames(), getInputNames())) {
    // Check the direction of these destination ports.
    // 'calyx.invoke' op '@r0' has input '%r.out', which is a source port. The
    // inputs are required to be destination ports.
    if (failed(verifyInvokeOpValue(*this, port, true)))
      return emitOpError() << "'@" << callee << "' has input '"
                           << portName.cast<StringAttr>().getValue()
                           << "', which is a source port. The inputs are "
                              "required to be destination ports.";
    // The go port should not appear in the parameter list.
    if (port == goValue)
      return emitOpError() << "the go or write_en port of '@" << callee
                           << "' cannot appear here.";
    // Check the direction of these source ports.
    if (failed(verifyInvokeOpValue(*this, input, false)))
      return emitOpError() << "'@" << callee << "' has output '"
                           << inputName.cast<StringAttr>().getValue()
                           << "', which is a destination port. The inputs are "
                              "required to be source ports.";
    if (failed(verifyComplexLogic(*this, input)))
      return emitOpError() << "'@" << callee << "' has '"
                           << inputName.cast<StringAttr>().getValue()
                           << "', which is not a port or constant. Complex "
                              "logic should be conducted in the guard.";
    if (input == doneValue)
      return emitOpError() << "the done port of '@" << callee
                           << "' cannot appear here.";
    // Check if the connection uses the callee's port.
    if (port.getDefiningOp() != operation && input.getDefiningOp() != operation)
      return emitOpError() << "the connection "
                           << portName.cast<StringAttr>().getValue() << " = "
                           << inputName.cast<StringAttr>().getValue()
                           << " is not defined as an input port of '@" << callee
                           << "'.";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Calyx library ops
//===----------------------------------------------------------------------===//

LogicalResult PadLibOp::verify() {
  unsigned inBits = getResult(0).getType().getIntOrFloatBitWidth();
  unsigned outBits = getResult(1).getType().getIntOrFloatBitWidth();
  if (inBits >= outBits)
    return emitOpError("expected input bits (")
           << inBits << ')' << " to be less than output bits (" << outBits
           << ')';
  return success();
}

LogicalResult SliceLibOp::verify() {
  unsigned inBits = getResult(0).getType().getIntOrFloatBitWidth();
  unsigned outBits = getResult(1).getType().getIntOrFloatBitWidth();
  if (inBits <= outBits)
    return emitOpError("expected input bits (")
           << inBits << ')' << " to be greater than output bits (" << outBits
           << ')';
  return success();
}

#define ImplBinPipeOpCellInterface(OpType, outName)                            \
  SmallVector<StringRef> OpType::portNames() {                                 \
    return {"clk", "reset", "go", "left", "right", outName, "done"};           \
  }                                                                            \
                                                                               \
  SmallVector<Direction> OpType::portDirections() {                            \
    return {Input, Input, Input, Input, Input, Output, Output};                \
  }                                                                            \
                                                                               \
  void OpType::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {              \
    getCellAsmResultNames(setNameFn, *this, this->portNames());                \
  }                                                                            \
                                                                               \
  SmallVector<DictionaryAttr> OpType::portAttributes() {                       \
    MLIRContext *context = getContext();                                       \
    IntegerAttr isSet = IntegerAttr::get(IntegerType::get(context, 1), 1);     \
    NamedAttrList go, clk, reset, done;                                        \
    go.append("go", isSet);                                                    \
    clk.append("clk", isSet);                                                  \
    reset.append("reset", isSet);                                              \
    done.append("done", isSet);                                                \
    return {                                                                   \
        clk.getDictionary(context),   /* Clk    */                             \
        reset.getDictionary(context), /* Reset  */                             \
        go.getDictionary(context),    /* Go     */                             \
        DictionaryAttr::get(context), /* Lhs    */                             \
        DictionaryAttr::get(context), /* Rhs    */                             \
        DictionaryAttr::get(context), /* Out    */                             \
        done.getDictionary(context)   /* Done   */                             \
    };                                                                         \
  }                                                                            \
                                                                               \
  bool OpType::isCombinational() { return false; }

#define ImplUnaryOpCellInterface(OpType)                                       \
  SmallVector<StringRef> OpType::portNames() { return {"in", "out"}; }         \
  SmallVector<Direction> OpType::portDirections() { return {Input, Output}; }  \
  SmallVector<DictionaryAttr> OpType::portAttributes() {                       \
    return {DictionaryAttr::get(getContext()),                                 \
            DictionaryAttr::get(getContext())};                                \
  }                                                                            \
  bool OpType::isCombinational() { return true; }                              \
  void OpType::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {              \
    getCellAsmResultNames(setNameFn, *this, this->portNames());                \
  }

#define ImplBinOpCellInterface(OpType)                                         \
  SmallVector<StringRef> OpType::portNames() {                                 \
    return {"left", "right", "out"};                                           \
  }                                                                            \
  SmallVector<Direction> OpType::portDirections() {                            \
    return {Input, Input, Output};                                             \
  }                                                                            \
  void OpType::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {              \
    getCellAsmResultNames(setNameFn, *this, this->portNames());                \
  }                                                                            \
  bool OpType::isCombinational() { return true; }                              \
  SmallVector<DictionaryAttr> OpType::portAttributes() {                       \
    return {DictionaryAttr::get(getContext()),                                 \
            DictionaryAttr::get(getContext()),                                 \
            DictionaryAttr::get(getContext())};                                \
  }

// clang-format off
ImplBinPipeOpCellInterface(MultPipeLibOp, "out")
ImplBinPipeOpCellInterface(DivUPipeLibOp, "out_quotient")
ImplBinPipeOpCellInterface(DivSPipeLibOp, "out_quotient")
ImplBinPipeOpCellInterface(RemUPipeLibOp, "out_remainder")
ImplBinPipeOpCellInterface(RemSPipeLibOp, "out_remainder")

ImplUnaryOpCellInterface(PadLibOp)
ImplUnaryOpCellInterface(SliceLibOp)
ImplUnaryOpCellInterface(NotLibOp)
ImplUnaryOpCellInterface(WireLibOp)
ImplUnaryOpCellInterface(ExtSILibOp)

ImplBinOpCellInterface(LtLibOp)
ImplBinOpCellInterface(GtLibOp)
ImplBinOpCellInterface(EqLibOp)
ImplBinOpCellInterface(NeqLibOp)
ImplBinOpCellInterface(GeLibOp)
ImplBinOpCellInterface(LeLibOp)
ImplBinOpCellInterface(SltLibOp)
ImplBinOpCellInterface(SgtLibOp)
ImplBinOpCellInterface(SeqLibOp)
ImplBinOpCellInterface(SneqLibOp)
ImplBinOpCellInterface(SgeLibOp)
ImplBinOpCellInterface(SleLibOp)

ImplBinOpCellInterface(AddLibOp)
ImplBinOpCellInterface(SubLibOp)
ImplBinOpCellInterface(ShruLibOp)
ImplBinOpCellInterface(RshLibOp)
ImplBinOpCellInterface(SrshLibOp)
ImplBinOpCellInterface(LshLibOp)
ImplBinOpCellInterface(AndLibOp)
ImplBinOpCellInterface(OrLibOp)
ImplBinOpCellInterface(XorLibOp)
// clang-format on

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxInterfaces.cpp.inc"

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
