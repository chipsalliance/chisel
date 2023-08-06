//===- SystemCOps.cpp - Implement the SystemC operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/HW/CustomDirectiveImpl.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyUniqueNamesInRegion(
    Operation *operation, ArrayAttr argNames,
    std::function<void(mlir::InFlightDiagnostic &)> attachNote) {
  DenseMap<StringRef, BlockArgument> portNames;
  DenseMap<StringRef, Operation *> memberNames;
  DenseMap<StringRef, Operation *> localNames;

  if (operation->getNumRegions() != 1)
    return operation->emitError("required to have exactly one region");

  bool portsVerified = true;

  for (auto arg : llvm::zip(argNames, operation->getRegion(0).getArguments())) {
    StringRef argName = std::get<0>(arg).cast<StringAttr>().getValue();
    BlockArgument argValue = std::get<1>(arg);

    if (portNames.count(argName)) {
      auto diag = mlir::emitError(argValue.getLoc(), "redefines name '")
                  << argName << "'";
      diag.attachNote(portNames[argName].getLoc())
          << "'" << argName << "' first defined here";
      attachNote(diag);
      portsVerified = false;
      continue;
    }

    portNames.insert({argName, argValue});
  }

  WalkResult result =
      operation->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<SCModuleOp>(op->getParentOp()))
          localNames.clear();

        if (auto nameDeclOp = dyn_cast<SystemCNameDeclOpInterface>(op)) {
          StringRef name = nameDeclOp.getName();

          auto reportNameRedefinition = [&](Location firstLoc) -> WalkResult {
            auto diag = mlir::emitError(op->getLoc(), "redefines name '")
                        << name << "'";
            diag.attachNote(firstLoc) << "'" << name << "' first defined here";
            attachNote(diag);
            return WalkResult::interrupt();
          };

          if (portNames.count(name))
            return reportNameRedefinition(portNames[name].getLoc());
          if (memberNames.count(name))
            return reportNameRedefinition(memberNames[name]->getLoc());
          if (localNames.count(name))
            return reportNameRedefinition(localNames[name]->getLoc());

          if (isa<SCModuleOp>(op->getParentOp()))
            memberNames.insert({name, op});
          else
            localNames.insert({name, op});
        }

        return WalkResult::advance();
      });

  if (result.wasInterrupted() || !portsVerified)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SCModuleOp
//===----------------------------------------------------------------------===//

static hw::ModulePort::Direction getDirection(Type type) {
  return TypeSwitch<Type, hw::ModulePort::Direction>(type)
      .Case<InOutType>([](auto ty) { return hw::ModulePort::Direction::InOut; })
      .Case<InputType>([](auto ty) { return hw::ModulePort::Direction::Input; })
      .Case<OutputType>(
          [](auto ty) { return hw::ModulePort::Direction::Output; });
}

SCModuleOp::PortDirectionRange
SCModuleOp::getPortsOfDirection(hw::ModulePort::Direction direction) {
  std::function<bool(const BlockArgument &)> predicateFn =
      [&](const BlockArgument &arg) -> bool {
    return getDirection(arg.getType()) == direction;
  };
  return llvm::make_filter_range(getArguments(), predicateFn);
}

hw::ModulePortInfo SCModuleOp::getPortList() {
  SmallVector<hw::PortInfo> ports;
  for (int i = 0, e = getNumArguments(); i < e; ++i) {
    hw::PortInfo info;
    info.name = getPortNames()[i].cast<StringAttr>();
    info.type = getSignalBaseType(getArgument(i).getType());
    info.dir = getDirection(info.type);
    ports.push_back(info);
  }
  return hw::ModulePortInfo{ports};
}

mlir::Region *SCModuleOp::getCallableRegion() { return &getBody(); }

ArrayRef<mlir::Type> SCModuleOp::getCallableResults() {
  return getResultTypes();
}

ArrayAttr SCModuleOp::getCallableArgAttrs() {
  return getArgAttrs().value_or(nullptr);
}

ArrayAttr SCModuleOp::getCallableResAttrs() {
  return getResAttrs().value_or(nullptr);
}

StringRef SCModuleOp::getModuleName() {
  return (*this)
      ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

ParseResult SCModuleOp::parse(OpAsmParser &parser, OperationState &result) {

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr moduleName;
  if (parser.parseSymbolName(moduleName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<Attribute> argNames;
  SmallVector<Attribute> argLocs;
  SmallVector<Attribute> resultNames;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Attribute> resultLocs;
  TypeAttr functionType;
  if (failed(hw::module_like_impl::parseModuleFunctionSignature(
          parser, isVariadic, entryArgs, argNames, argLocs, resultNames,
          resultAttrs, resultLocs, functionType)))
    return failure();

  // Parse the attribute dict.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  result.addAttribute("portNames",
                      ArrayAttr::get(parser.getContext(), argNames));

  result.addAttribute(SCModuleOp::getFunctionTypeAttrName(result.name),
                      functionType);

  mlir::function_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, entryArgs, resultAttrs,
      SCModuleOp::getArgAttrsAttrName(result.name),
      SCModuleOp::getResAttrsAttrName(result.name));

  auto &body = *result.addRegion();
  if (parser.parseRegion(body, entryArgs))
    return failure();
  if (body.empty())
    body.push_back(std::make_unique<Block>().release());

  return success();
}

void SCModuleOp::print(OpAsmPrinter &p) {
  p << ' ';

  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility =
          getOperation()->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());
  p << ' ';

  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(
      p, *this, getFunctionType().getInputs(), false, {}, needArgNamesAttr);
  mlir::function_interface_impl::printFunctionAttributes(
      p, *this,
      {"portNames", getFunctionTypeAttrName(), getArgAttrsAttrName(),
       getResAttrsAttrName()});

  p << ' ';
  p.printRegion(getBody(), false, false);
}

static Type wrapPortType(Type type, hw::ModulePort::Direction direction) {
  if (auto inoutTy = type.dyn_cast<hw::InOutType>())
    type = inoutTy.getElementType();

  switch (direction) {
  case hw::ModulePort::Direction::InOut:
    return InOutType::get(type);
  case hw::ModulePort::Direction::Input:
    return InputType::get(type);
  case hw::ModulePort::Direction::Output:
    return OutputType::get(type);
  }
  llvm_unreachable("Impossible port direction");
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, ArrayAttr portNames,
                       ArrayRef<Type> portTypes,
                       ArrayRef<NamedAttribute> attributes) {
  odsState.addAttribute(getPortNamesAttrName(odsState.name), portNames);
  Region *region = odsState.addRegion();

  auto moduleType = odsBuilder.getFunctionType(portTypes, {});
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name),
                        TypeAttr::get(moduleType));

  odsState.addAttribute(SymbolTable::getSymbolAttrName(), name);
  region->push_back(new Block);
  region->addArguments(
      portTypes,
      SmallVector<Location>(portTypes.size(), odsBuilder.getUnknownLoc()));
  odsState.addAttributes(attributes);
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, ArrayRef<hw::PortInfo> ports,
                       ArrayRef<NamedAttribute> attributes) {
  MLIRContext *ctxt = odsBuilder.getContext();
  SmallVector<Attribute> portNames;
  SmallVector<Type> portTypes;
  for (auto port : ports) {
    portNames.push_back(StringAttr::get(ctxt, port.getName()));
    portTypes.push_back(wrapPortType(port.type, port.dir));
  }
  build(odsBuilder, odsState, name, ArrayAttr::get(ctxt, portNames), portTypes);
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, const hw::ModulePortInfo &ports,
                       ArrayRef<NamedAttribute> attributes) {
  MLIRContext *ctxt = odsBuilder.getContext();
  SmallVector<Attribute> portNames;
  SmallVector<Type> portTypes;
  for (auto port : ports) {
    portNames.push_back(StringAttr::get(ctxt, port.getName()));
    portTypes.push_back(wrapPortType(port.type, port.dir));
  }
  build(odsBuilder, odsState, name, ArrayAttr::get(ctxt, portNames), portTypes);
}

void SCModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  ArrayAttr portNames = getPortNames();
  for (size_t i = 0, e = getNumArguments(); i != e; ++i) {
    auto str = portNames[i].cast<StringAttr>().getValue();
    setNameFn(getArgument(i), str);
  }
}

LogicalResult SCModuleOp::verify() {
  if (getFunctionType().getNumResults() != 0)
    return emitOpError(
        "incorrect number of function results (always has to be 0)");
  if (getPortNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of port names");

  for (auto arg : getArguments()) {
    if (!hw::type_isa<InputType, OutputType, InOutType>(arg.getType()))
      return mlir::emitError(
          arg.getLoc(),
          "module port must be of type 'sc_in', 'sc_out', or 'sc_inout'");
  }

  for (auto portName : getPortNames()) {
    if (portName.cast<StringAttr>().getValue().empty())
      return emitOpError("port name must not be empty");
  }

  return success();
}

LogicalResult SCModuleOp::verifyRegions() {
  auto attachNote = [&](mlir::InFlightDiagnostic &diag) {
    diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
  };
  return verifyUniqueNamesInRegion(getOperation(), getPortNames(), attachNote);
}

CtorOp SCModuleOp::getOrCreateCtor() {
  CtorOp ctor;
  getBody().walk([&](Operation *op) {
    if ((ctor = dyn_cast<CtorOp>(op)))
      return WalkResult::interrupt();

    return WalkResult::skip();
  });

  if (ctor)
    return ctor;

  return OpBuilder(getBody()).create<CtorOp>(getLoc());
}

DestructorOp SCModuleOp::getOrCreateDestructor() {
  DestructorOp destructor;
  getBody().walk([&](Operation *op) {
    if ((destructor = dyn_cast<DestructorOp>(op)))
      return WalkResult::interrupt();

    return WalkResult::skip();
  });

  if (destructor)
    return destructor;

  return OpBuilder::atBlockEnd(getBodyBlock()).create<DestructorOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// SignalOp
//===----------------------------------------------------------------------===//

void SignalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getSignal(), getName());
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

OpFoldResult ConvertOp::fold(FoldAdaptor) {
  if (getInput().getType() == getResult().getType())
    return getInput();

  if (auto other = getInput().getDefiningOp<ConvertOp>()) {
    Type inputType = other.getInput().getType();
    Type intermediateType = getInput().getType();

    if (inputType != getResult().getType())
      return {};

    // Either both the input and intermediate types are signed or both are
    // unsigned.
    bool inputSigned = inputType.isa<SignedType, IntBaseType>();
    bool intermediateSigned = intermediateType.isa<SignedType, IntBaseType>();
    if (inputSigned ^ intermediateSigned)
      return {};

    // Converting 4-valued to 2-valued and back may lose information.
    if (inputType.isa<LogicVectorBaseType, LogicType>() &&
        !intermediateType.isa<LogicVectorBaseType, LogicType>())
      return {};

    auto inputBw = getBitWidth(inputType);
    auto intermediateBw = getBitWidth(intermediateType);

    if (!inputBw && intermediateBw) {
      if (inputType.isa<IntBaseType, UIntBaseType>() && *intermediateBw >= 64)
        return other.getInput();
      // We cannot support input types of signed, unsigned, and vector types
      // since they have no upper bound for the bit-width.
    }

    if (!intermediateBw) {
      if (intermediateType.isa<BitVectorBaseType, LogicVectorBaseType>())
        return other.getInput();

      if (!inputBw && inputType.isa<IntBaseType, UIntBaseType>() &&
          intermediateType.isa<SignedType, UnsignedType>())
        return other.getInput();

      if (inputBw && *inputBw <= 64 &&
          intermediateType
              .isa<IntBaseType, UIntBaseType, SignedType, UnsignedType>())
        return other.getInput();

      // We have to be careful with the signed and unsigned types as they often
      // have a max bit-width defined (that can be customized) and thus folding
      // here could change the behavior.
    }

    if (inputBw && intermediateBw && *inputBw <= *intermediateBw)
      return other.getInput();
  }

  return {};
}

//===----------------------------------------------------------------------===//
// CtorOp
//===----------------------------------------------------------------------===//

LogicalResult CtorOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// SCFuncOp
//===----------------------------------------------------------------------===//

void SCFuncOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getHandle(), getName());
}

LogicalResult SCFuncOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceDeclOp
//===----------------------------------------------------------------------===//

void InstanceDeclOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getInstanceHandle(), getName());
}

StringRef InstanceDeclOp::getInstanceName() { return getName(); }
StringAttr InstanceDeclOp::getInstanceNameAttr() { return getNameAttr(); }

Operation *InstanceDeclOp::getReferencedModule(const hw::HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(getModuleNameAttr()))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(getModuleName());
}

Operation *InstanceDeclOp::getReferencedModule() {
  return getReferencedModule(/*cache=*/nullptr);
}

LogicalResult
InstanceDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("cannot find module definition '")
           << getModuleName() << "'";

  auto emitError = [&](const std::function<void(InFlightDiagnostic & diag)> &fn)
      -> LogicalResult {
    auto diag = emitOpError();
    fn(diag);
    diag.attachNote(module->getLoc()) << "module declared here";
    return failure();
  };

  // It must be a systemc module.
  if (!isa<SCModuleOp>(module))
    return emitError([&](auto &diag) {
      diag << "symbol reference '" << getModuleName()
           << "' isn't a systemc module";
    });

  auto scModule = cast<SCModuleOp>(module);

  // Check that the module name of the symbol and instance type match.
  if (scModule.getModuleName() != getInstanceType().getModuleName())
    return emitError([&](auto &diag) {
      diag << "module names must match; expected '" << scModule.getModuleName()
           << "' but got '" << getInstanceType().getModuleName().getValue()
           << "'";
    });

  // Check that port types and names are consistent with the referenced module.
  ArrayRef<ModuleType::PortInfo> ports = getInstanceType().getPorts();
  ArrayAttr modArgNames = scModule.getPortNames();
  auto numPorts = ports.size();
  auto expectedPortTypes = scModule.getArgumentTypes();

  if (expectedPortTypes.size() != numPorts)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of ports; expected "
           << expectedPortTypes.size() << " but got " << numPorts;
    });

  for (size_t i = 0; i != numPorts; ++i) {
    if (ports[i].type != expectedPortTypes[i]) {
      return emitError([&](auto &diag) {
        diag << "port type #" << i << " must be " << expectedPortTypes[i]
             << ", but got " << ports[i].type;
      });
    }

    if (ports[i].name != modArgNames[i])
      return emitError([&](auto &diag) {
        diag << "port name #" << i << " must be " << modArgNames[i]
             << ", but got " << ports[i].name;
      });
  }

  return success();
}

hw::ModulePortInfo InstanceDeclOp::getPortList() {
  return cast<hw::PortList>(getReferencedModule()).getPortList();
}

//===----------------------------------------------------------------------===//
// DestructorOp
//===----------------------------------------------------------------------===//

LogicalResult DestructorOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// BindPortOp
//===----------------------------------------------------------------------===//

ParseResult BindPortOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand instance, channel;
  std::string portName;
  if (parser.parseOperand(instance) || parser.parseLSquare() ||
      parser.parseString(&portName))
    return failure();

  auto portNameLoc = parser.getCurrentLocation();

  if (parser.parseRSquare() || parser.parseKeyword("to") ||
      parser.parseOperand(channel))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  auto typeListLoc = parser.getCurrentLocation();
  SmallVector<Type> types;
  if (parser.parseColonTypeList(types))
    return failure();

  if (types.size() != 2)
    return parser.emitError(typeListLoc,
                            "expected a list of exactly 2 types, but got ")
           << types.size();

  if (parser.resolveOperand(instance, types[0], result.operands))
    return failure();
  if (parser.resolveOperand(channel, types[1], result.operands))
    return failure();

  if (auto moduleType = types[0].dyn_cast<ModuleType>()) {
    auto ports = moduleType.getPorts();
    uint64_t index = 0;
    for (auto port : ports) {
      if (port.name == portName)
        break;
      index++;
    }
    if (index >= ports.size())
      return parser.emitError(portNameLoc, "port name \"")
             << portName << "\" not found in module";

    result.addAttribute("portId", parser.getBuilder().getIndexAttr(index));

    return success();
  }

  return failure();
}

void BindPortOp::print(OpAsmPrinter &p) {
  p << " " << getInstance() << "["
    << getInstance()
           .getType()
           .cast<ModuleType>()
           .getPorts()[getPortId().getZExtValue()]
           .name
    << "] to " << getChannel();
  p.printOptionalAttrDict((*this)->getAttrs(), {"portId"});
  p << " : " << getInstance().getType() << ", " << getChannel().getType();
}

LogicalResult BindPortOp::verify() {
  auto ports = getInstance().getType().cast<ModuleType>().getPorts();
  if (getPortId().getZExtValue() >= ports.size())
    return emitOpError("port #")
           << getPortId().getZExtValue() << " does not exist, there are only "
           << ports.size() << " ports";

  // Verify that the base types match.
  Type portType = ports[getPortId().getZExtValue()].type;
  Type channelType = getChannel().getType();
  if (getSignalBaseType(portType) != getSignalBaseType(channelType))
    return emitOpError() << portType << " port cannot be bound to "
                         << channelType << " channel due to base type mismatch";

  // Verify that the port/channel directions are valid.
  if ((portType.isa<InputType>() && channelType.isa<OutputType>()) ||
      (portType.isa<OutputType>() && channelType.isa<InputType>()))
    return emitOpError() << portType << " port cannot be bound to "
                         << channelType
                         << " channel due to port direction mismatch";

  return success();
}

StringRef BindPortOp::getPortName() {
  return getInstance()
      .getType()
      .cast<ModuleType>()
      .getPorts()[getPortId().getZExtValue()]
      .name.getValue();
}

//===----------------------------------------------------------------------===//
// SensitiveOp
//===----------------------------------------------------------------------===//

LogicalResult SensitiveOp::canonicalize(SensitiveOp op,
                                        PatternRewriter &rewriter) {
  if (op.getSensitivities().empty()) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getVariable(), getName());
}

ParseResult VariableOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parseImplicitSSAName(parser, nameAttr))
    return failure();
  result.addAttribute("name", nameAttr);

  OpAsmParser::UnresolvedOperand init;
  auto initResult = parser.parseOptionalOperand(init);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  Type variableType;
  if (parser.parseColonType(variableType))
    return failure();

  if (initResult.has_value()) {
    if (parser.resolveOperand(init, variableType, result.operands))
      return failure();
  }
  result.addTypes({variableType});

  return success();
}

void VariableOp::print(::mlir::OpAsmPrinter &p) {
  p << " ";

  if (getInit())
    p << getInit() << " ";

  p.printOptionalAttrDict(getOperation()->getAttrs(), {"name"});
  p << ": " << getVariable().getType();
}

LogicalResult VariableOp::verify() {
  if (getInit() && getInit().getType() != getVariable().getType())
    return emitOpError(
               "'init' and 'variable' must have the same type, but got ")
           << getInit().getType() << " and " << getVariable().getType();

  return success();
}

//===----------------------------------------------------------------------===//
// InteropVerilatedOp
//===----------------------------------------------------------------------===//

/// Create a instance that refers to a known module.
void InteropVerilatedOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               Operation *module, StringAttr name,
                               ArrayRef<Value> inputs) {
  auto [argNames, resultNames] =
      hw::instance_like_impl::getHWModuleArgAndResultNames(module);
  build(odsBuilder, odsState, hw::getModuleType(module).getResults(), name,
        FlatSymbolRefAttr::get(SymbolTable::getSymbolName(module)), argNames,
        resultNames, inputs);
}

LogicalResult
InteropVerilatedOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return hw::instance_like_impl::verifyInstanceOfHWModule(
      *this, getModuleNameAttr(), getInputs(), getResultTypes(),
      getInputNames(), getResultNames(), ArrayAttr(), symbolTable);
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InteropVerilatedOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  hw::instance_like_impl::getAsmResultNames(setNameFn, getInstanceName(),
                                            getResultNames(), getResults());
}

//===----------------------------------------------------------------------===//
// CallOp
//
// TODO: The implementation for this operation was copy-pasted from the
// 'func' dialect. Ideally, this upstream dialect refactored such that we can
// re-use the implementation here.
//===----------------------------------------------------------------------===//

// FIXME: This is an exact copy from upstream
LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

// This verifier was added compared to the upstream implementation.
LogicalResult CallOp::verify() {
  if (getNumResults() > 1)
    return emitOpError(
        "incorrect number of function results (always has to be 0 or 1)");

  return success();
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

// This verifier was added compared to the upstream implementation.
LogicalResult CallIndirectOp::verify() {
  if (getNumResults() > 1)
    return emitOpError(
        "incorrect number of function results (always has to be 0 or 1)");

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//
// TODO: Most of the implementation for this operation was copy-pasted from the
// 'func' dialect. Ideally, this upstream dialect refactored such that we can
// re-use the implementation here.
//===----------------------------------------------------------------------===//

// Note that the create and build operations are taken from upstream, but the
// argNames argument was added.
FuncOp FuncOp::create(Location location, StringRef name, ArrayAttr argNames,
                      FunctionType type, ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  FuncOp::build(builder, state, name, argNames, type, attrs);
  return cast<FuncOp>(Operation::create(state));
}

FuncOp FuncOp::create(Location location, StringRef name, ArrayAttr argNames,
                      FunctionType type, Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, argNames, type, ArrayRef(attrRef));
}

FuncOp FuncOp::create(Location location, StringRef name, ArrayAttr argNames,
                      FunctionType type, ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  FuncOp func = create(location, name, argNames, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                   StringRef name, ArrayAttr argNames, FunctionType type,
                   ArrayRef<NamedAttribute> attrs,
                   ArrayRef<DictionaryAttr> argAttrs) {
  odsState.addAttribute(getArgNamesAttrName(odsState.name), argNames);
  odsState.addAttribute(SymbolTable::getSymbolAttrName(),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute(FuncOp::getFunctionTypeAttrName(odsState.name),
                        TypeAttr::get(type));
  odsState.attributes.append(attrs.begin(), attrs.end());
  odsState.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  mlir::function_interface_impl::addArgAndResultAttrs(
      odsBuilder, odsState, argAttrs,
      /*resultAttrs=*/std::nullopt, FuncOp::getArgAttrsAttrName(odsState.name),
      FuncOp::getResAttrsAttrName(odsState.name));
}

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  // This was added specifically for our implementation, upstream does not have
  // this feature.
  if (succeeded(parser.parseOptionalKeyword("externC")))
    result.addAttribute(getExternCAttrName(result.name),
                        UnitAttr::get(result.getContext()));

  // FIXME: below is an exact copy of the
  // mlir::function_interface_impl::parseFunctionOp implementation, this was
  // needed because we need to access the SSA names of the arguments.
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse visibility.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  mlir::SMLoc signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;
  if (mlir::function_interface_impl::parseFunctionSignature(
          parser, false, entryArgs, isVariadic, resultTypes, resultAttrs))
    return failure();

  std::string errorMessage;
  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  Type type = buildFuncType(
      builder, argTypes, resultTypes,
      mlir::function_interface_impl::VariadicFlag(isVariadic), errorMessage);
  if (!type) {
    return parser.emitError(signatureLocation)
           << "failed to construct function type"
           << (errorMessage.empty() ? "" : ": ") << errorMessage;
  }
  result.addAttribute(FuncOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  mlir::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       {SymbolTable::getVisibilityAttrName(), SymbolTable::getSymbolAttrName(),
        FuncOp::getFunctionTypeAttrName(result.name).getValue()}) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  assert(resultAttrs.size() == resultTypes.size());
  mlir::function_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs,
      FuncOp::getArgAttrsAttrName(result.name),
      FuncOp::getResAttrsAttrName(result.name));

  // Parse the optional function body. The printer will not print the body if
  // its empty, so disallow parsing of empty body in the parser.
  auto *body = result.addRegion();
  mlir::SMLoc loc = parser.getCurrentLocation();
  mlir::OptionalParseResult parseResult =
      parser.parseOptionalRegion(*body, entryArgs,
                                 /*enableNameShadowing=*/false);
  if (parseResult.has_value()) {
    if (failed(*parseResult))
      return failure();
    // Function body was parsed, make sure its not empty.
    if (body->empty())
      return parser.emitError(loc, "expected non-empty function body");
  }

  // Everythink below is added compared to the upstream implemenation to handle
  // argument names.
  SmallVector<Attribute> argNames;
  if (!entryArgs.empty() && !entryArgs.front().ssaName.name.empty()) {
    for (auto &arg : entryArgs)
      argNames.push_back(
          StringAttr::get(parser.getContext(), arg.ssaName.name.drop_front()));
  }

  result.addAttribute(getArgNamesAttrName(result.name),
                      ArrayAttr::get(parser.getContext(), argNames));

  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  if (getExternC())
    p << " externC";

  mlir::FunctionOpInterface op = *this;

  // FIXME: inlined mlir::function_interface_impl::printFunctionOp because we
  // need to elide more attributes

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  p << ' ';

  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';
  p.printSymbolName(funcName);

  ArrayRef<Type> argTypes = op.getArgumentTypes();
  ArrayRef<Type> resultTypes = op.getResultTypes();
  mlir::function_interface_impl::printFunctionSignature(p, op, argTypes, false,
                                                        resultTypes);
  mlir::function_interface_impl::printFunctionAttributes(
      p, op,
      {visibilityAttrName, "externC", "argNames", getFunctionTypeAttrName(),
       getArgAttrsAttrName(), getResAttrsAttrName()});
  // Print the body if this is not an external function.
  Region &body = op->getRegion(0);
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

// FIXME: the below clone operation are exact copies from upstream.

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void FuncOp::cloneInto(FuncOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
FuncOp FuncOp::clone(IRMapping &mapper) {
  // Create the new function.
  FuncOp newFunc = cast<FuncOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i)
      if (!mapper.contains(getArgument(i)))
        newInputs.push_back(oldType.getInput(i));

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs,
                                        oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i)
          if (!mapper.contains(getArgument(i)))
            newArgAttrs.push_back(argAttrs[i]);
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}

FuncOp FuncOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

// The following functions are entirely new additions compared to upstream.

void FuncOp::getAsmBlockArgumentNames(mlir::Region &region,
                                      mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  for (auto [arg, name] : llvm::zip(getArguments(), getArgNames()))
    setNameFn(arg, name.cast<StringAttr>().getValue());
}

LogicalResult FuncOp::verify() {
  if (getFunctionType().getNumResults() > 1)
    return emitOpError(
        "incorrect number of function results (always has to be 0 or 1)");

  if (getBody().empty())
    return success();

  if (getArgNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of argument names");

  for (auto portName : getArgNames()) {
    if (portName.cast<StringAttr>().getValue().empty())
      return emitOpError("arg name must not be empty");
  }

  return success();
}

LogicalResult FuncOp::verifyRegions() {
  auto attachNote = [&](mlir::InFlightDiagnostic &diag) {
    diag.attachNote(getLoc()) << "in function '@" << getName() << "'";
  };
  return verifyUniqueNamesInRegion(getOperation(), getArgNames(), attachNote);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//
// TODO: The implementation for this operation was copy-pasted from the
// 'func' dialect. Ideally, this upstream dialect refactored such that we can
// re-use the implementation here.
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SystemC/SystemC.cpp.inc"
