//===- ExportVerilog.cpp - Verilog Emitter --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Verilog emitter implementation.
//
// CAREFUL: This file covers the emission phase of `ExportVerilog` which mainly
// walks the IR and produces output. Do NOT modify the IR during this walk, as
// emission occurs in a highly parallel fashion. If you need to modify the IR,
// do so during the preparation phase which lives in `PrepareForEmission.cpp`.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportVerilog.h"
#include "../PassDetail.h"
#include "ExportVerilogInternals.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/LTL/LTLVisitors.h"
#include "circt/Dialect/OM/OMOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "circt/Dialect/Verif/VerifVisitors.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/LoweringOptions.h"
#include "circt/Support/Path.h"
#include "circt/Support/PrettyPrinterHelpers.h"
#include "circt/Support/Version.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

using namespace comb;
using namespace hw;
using namespace sv;
using namespace ExportVerilog;

using namespace pretty;

#define DEBUG_TYPE "export-verilog"

StringRef circtHeader = "circt_header.svh";
StringRef circtHeaderInclude = "`include \"circt_header.svh\"\n";

namespace {
/// This enum keeps track of the precedence level of various binary operators,
/// where a lower number binds tighter.
enum VerilogPrecedence {
  // Normal precedence levels.
  Symbol,          // Atomic symbol like "foo" and {a,b}
  Selection,       // () , [] , :: , ., $signed()
  Unary,           // Unary operators like ~foo
  Multiply,        // * , / , %
  Addition,        // + , -
  Shift,           // << , >>, <<<, >>>
  Comparison,      // > , >= , < , <=
  Equality,        // == , !=
  And,             // &
  Xor,             // ^ , ^~
  Or,              // |
  AndShortCircuit, // &&
  Conditional,     // ? :

  LowestPrecedence, // Sentinel which is always the lowest precedence.
};

/// This enum keeps track of whether the emitted subexpression is signed or
/// unsigned as seen from the Verilog language perspective.
enum SubExprSignResult { IsSigned, IsUnsigned };

/// This is information precomputed about each subexpression in the tree we
/// are emitting as a unit.
struct SubExprInfo {
  /// The precedence of this expression.
  VerilogPrecedence precedence;

  /// The signedness of the expression.
  SubExprSignResult signedness;

  SubExprInfo(VerilogPrecedence precedence, SubExprSignResult signedness)
      : precedence(precedence), signedness(signedness) {}
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helper routines
//===----------------------------------------------------------------------===//

static TypedAttr getInt32Attr(MLIRContext *ctx, uint32_t value) {
  return Builder(ctx).getI32IntegerAttr(value);
}

static TypedAttr getIntAttr(MLIRContext *ctx, Type t, const APInt &value) {
  return Builder(ctx).getIntegerAttr(t, value);
}

/// Return true for nullary operations that are better emitted multiple
/// times as inline expression (when they have multiple uses) rather than having
/// a temporary wire.
///
/// This can only handle nullary expressions, because we don't want to replicate
/// subtrees arbitrarily.
static bool isDuplicatableNullaryExpression(Operation *op) {
  // We don't want wires that are just constants aesthetically.
  if (isConstantExpression(op))
    return true;

  // If this is a small verbatim expression with no side effects, duplicate it
  // inline.
  if (isa<VerbatimExprOp>(op)) {
    if (op->getNumOperands() == 0 &&
        op->getAttrOfType<StringAttr>("format_string").getValue().size() <= 32)
      return true;
  }

  // Always duplicate XMRs into their use site.
  if (isa<XMRRefOp>(op))
    return true;

  // If this is a macro reference without side effects, allow duplication.
  if (isa<MacroRefExprOp>(op))
    return true;

  return false;
}

// Return true if the expression can be inlined even when the op has multiple
// uses. Be careful to add operations here since it might cause exponential
// emission without proper restrictions.
static bool isDuplicatableExpression(Operation *op) {
  if (op->getNumOperands() == 0)
    return isDuplicatableNullaryExpression(op);

  // It is cheap to inline extract op.
  if (isa<comb::ExtractOp, hw::StructExtractOp, hw::UnionExtractOp>(op))
    return true;

  // We only inline array_get with a constant, port or wire index.
  if (auto array = dyn_cast<hw::ArrayGetOp>(op)) {
    auto *indexOp = array.getIndex().getDefiningOp();
    if (!indexOp || isa<ConstantOp>(indexOp))
      return true;
    if (auto read = dyn_cast<ReadInOutOp>(indexOp)) {
      auto *readSrc = read.getInput().getDefiningOp();
      // A port or wire is ok to duplicate reads.
      return !readSrc || isa<sv::WireOp, LogicOp>(readSrc);
    }

    return false;
  }

  return false;
}

/// Return the verilog name of the operations that can define a symbol.
/// Legalized names are added to "hw.verilogName" so look up it when the
/// attribute already exists.
StringRef ExportVerilog::getSymOpName(Operation *symOp) {
  // Typeswitch of operation types which can define a symbol.
  // If legalizeNames has renamed it, then the attribute must be set.
  if (auto attr = symOp->getAttrOfType<StringAttr>("hw.verilogName"))
    return attr.getValue();
  return TypeSwitch<Operation *, StringRef>(symOp)
      .Case<HWModuleOp, HWModuleExternOp, HWModuleGeneratedOp, HWTestModuleOp>(
          [](Operation *op) { return getVerilogModuleName(op); })
      .Case<InterfaceOp>([&](InterfaceOp op) {
        return getVerilogModuleNameAttr(op).getValue();
      })
      .Case<InterfaceSignalOp>(
          [&](InterfaceSignalOp op) { return op.getSymName(); })
      .Case<InterfaceModportOp>(
          [&](InterfaceModportOp op) { return op.getSymName(); })
      .Default([&](Operation *op) {
        if (auto attr = op->getAttrOfType<StringAttr>("name"))
          return attr.getValue();
        if (auto attr = op->getAttrOfType<StringAttr>("instanceName"))
          return attr.getValue();
        if (auto attr = op->getAttrOfType<StringAttr>("sv.namehint"))
          return attr.getValue();
        if (auto attr =
                op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
          return attr.getValue();
        return StringRef("");
      });
}

/// Emits a known-safe token that is legal when indexing into singleton arrays.
template <typename PPS>
static void emitZeroWidthIndexingValue(PPS &os) {
  os << "/*Zero width*/ 1\'b0";
}

static StringRef getPortVerilogName(Operation *module, PortInfo port) {
  char verilogNameAttr[] = "hw.verilogName";
  if (port.attrs)
    if (auto updatedName = port.attrs.get(verilogNameAttr))
      return updatedName.cast<StringAttr>().getValue();
  return port.name;
}

/// Return the verilog name of the port for the module.
static StringRef getPortVerilogName(Operation *module, size_t portArgNum) {
  if (auto htmo = dyn_cast<HWTestModuleOp>(module))
    return getPortVerilogName(module, htmo.getPortList().at(portArgNum));
  auto numInputs = hw::getModuleNumInputs(module);
  // portArgNum is the index into the result of getAllModulePortInfos.
  // Also ensure the correct index into the input/output list is computed.
  ssize_t portId = portArgNum;
  char verilogNameAttr[] = "hw.verilogName";
  // Check for input ports.
  if (portArgNum < numInputs) {
    if (auto argAttr =
            cast<mlir::FunctionOpInterface>(module).getArgAttrsAttr())
      if (auto argDict = argAttr[portArgNum].cast<DictionaryAttr>())
        if (auto updatedName = argDict.get(verilogNameAttr))
          return updatedName.cast<StringAttr>().getValue();
    // Get the original name of input port if no renaming.
    return module->getAttrOfType<ArrayAttr>("argNames")[portArgNum]
        .cast<StringAttr>()
        .getValue();
  }

  // If its an output port, get the index into the output port array.
  portId = portArgNum - numInputs;
  if (auto argAttr = cast<mlir::FunctionOpInterface>(module).getResAttrsAttr())
    if (auto argDict = argAttr[portId].cast<DictionaryAttr>())
      if (auto updatedName = argDict.get(verilogNameAttr))
        return updatedName.cast<StringAttr>().getValue();
  // Get the original name of output port if no renaming.
  return module->getAttrOfType<ArrayAttr>("resultNames")[portId]
      .cast<StringAttr>()
      .getValue();
}

/// This predicate returns true if the specified operation is considered a
/// potentially inlinable Verilog expression.  These nodes always have a single
/// result, but may have side effects (e.g. `sv.verbatim.expr.se`).
/// MemoryEffects should be checked if a client cares.
bool ExportVerilog::isVerilogExpression(Operation *op) {
  // These are SV dialect expressions.
  if (isa<ReadInOutOp, AggregateConstantOp, ArrayIndexInOutOp,
          IndexedPartSelectInOutOp, StructFieldInOutOp, IndexedPartSelectOp,
          ParamValueOp, XMROp, XMRRefOp, SampledOp, EnumConstantOp,
          SystemFunctionOp>(op))
    return true;

  // All HW combinational logic ops and SV expression ops are Verilog
  // expressions.
  return isCombinational(op) || isExpression(op);
}

// NOLINTBEGIN(misc-no-recursion)
/// Push this type's dimension into a vector.
static void getTypeDims(SmallVectorImpl<Attribute> &dims, Type type,
                        Location loc) {
  if (auto integer = hw::type_dyn_cast<IntegerType>(type)) {
    if (integer.getWidth() != 1)
      dims.push_back(getInt32Attr(type.getContext(), integer.getWidth()));
    return;
  }
  if (auto array = hw::type_dyn_cast<ArrayType>(type)) {
    dims.push_back(getInt32Attr(type.getContext(), array.getSize()));
    getTypeDims(dims, array.getElementType(), loc);

    return;
  }
  if (auto intType = hw::type_dyn_cast<IntType>(type)) {
    dims.push_back(intType.getWidth());
    return;
  }

  if (auto inout = hw::type_dyn_cast<InOutType>(type))
    return getTypeDims(dims, inout.getElementType(), loc);
  if (auto uarray = hw::type_dyn_cast<hw::UnpackedArrayType>(type))
    return getTypeDims(dims, uarray.getElementType(), loc);
  if (hw::type_isa<InterfaceType, StructType, EnumType>(type))
    return;

  mlir::emitError(loc, "value has an unsupported verilog type ") << type;
}
// NOLINTEND(misc-no-recursion)

/// True iff 'a' and 'b' have the same wire dims.
static bool haveMatchingDims(Type a, Type b, Location loc) {
  SmallVector<Attribute, 4> aDims;
  getTypeDims(aDims, a, loc);

  SmallVector<Attribute, 4> bDims;
  getTypeDims(bDims, b, loc);

  return aDims == bDims;
}

// NOLINTBEGIN(misc-no-recursion)
bool ExportVerilog::isZeroBitType(Type type) {
  type = getCanonicalType(type);
  if (auto intType = type.dyn_cast<IntegerType>())
    return intType.getWidth() == 0;
  if (auto inout = type.dyn_cast<hw::InOutType>())
    return isZeroBitType(inout.getElementType());
  if (auto uarray = type.dyn_cast<hw::UnpackedArrayType>())
    return uarray.getSize() == 0 || isZeroBitType(uarray.getElementType());
  if (auto array = type.dyn_cast<hw::ArrayType>())
    return array.getSize() == 0 || isZeroBitType(array.getElementType());
  if (auto structType = type.dyn_cast<hw::StructType>())
    return llvm::all_of(structType.getElements(),
                        [](auto elem) { return isZeroBitType(elem.type); });
  if (auto enumType = type.dyn_cast<hw::EnumType>())
    return enumType.getFields().empty();
  if (auto unionType = type.dyn_cast<hw::UnionType>())
    return hw::getBitWidth(unionType) == 0;

  // We have an open type system, so assume it is ok.
  return false;
}
// NOLINTEND(misc-no-recursion)

/// Given a set of known nested types (those supported by this pass), strip off
/// leading unpacked types.  This strips off portions of the type that are
/// printed to the right of the name in verilog.
// NOLINTBEGIN(misc-no-recursion)
static Type stripUnpackedTypes(Type type) {
  return TypeSwitch<Type, Type>(type)
      .Case<InOutType>([](InOutType inoutType) {
        return stripUnpackedTypes(inoutType.getElementType());
      })
      .Case<UnpackedArrayType>([](UnpackedArrayType arrayType) {
        return stripUnpackedTypes(arrayType.getElementType());
      })
      .Default([](Type type) { return type; });
}

/// Return true if type has a struct type as a subtype.
static bool hasStructType(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<InOutType, UnpackedArrayType, ArrayType>([](auto parentType) {
        return hasStructType(parentType.getElementType());
      })
      .Case<StructType>([](auto) { return true; })
      .Default([](auto) { return false; });
}
// NOLINTEND(misc-no-recursion)

/// Return the word (e.g. "reg") in Verilog to declare the specified thing.
static StringRef getVerilogDeclWord(Operation *op,
                                    const LoweringOptions &options) {
  if (isa<RegOp>(op)) {
    // Check if the type stored in this register is a struct or array of
    // structs. In this case, according to spec section 6.8, the "reg" prefix
    // should be left off.
    auto elementType =
        op->getResult(0).getType().cast<InOutType>().getElementType();
    if (elementType.isa<StructType>())
      return "";
    if (elementType.isa<UnionType>())
      return "";
    if (elementType.isa<EnumType>())
      return "";
    if (auto innerType = elementType.dyn_cast<ArrayType>()) {
      while (innerType.getElementType().isa<ArrayType>())
        innerType = innerType.getElementType().cast<ArrayType>();
      if (innerType.getElementType().isa<StructType>() ||
          innerType.getElementType().isa<TypeAliasType>())
        return "";
    }
    if (elementType.isa<TypeAliasType>())
      return "";

    return "reg";
  }
  if (isa<sv::WireOp>(op))
    return "wire";
  if (isa<ConstantOp, AggregateConstantOp, LocalParamOp, ParamValueOp>(op))
    return "localparam";

  // Interfaces instances use the name of the declared interface.
  if (auto interface = dyn_cast<InterfaceInstanceOp>(op))
    return interface.getInterfaceType().getInterface().getValue();

  // If 'op' is in a module, output 'wire'. If 'op' is in a procedural block,
  // fall through to default.
  bool isProcedural = op->getParentOp()->hasTrait<ProceduralRegion>();

  if (isa<LogicOp>(op)) {
    // If the logic op is defined in a procedural region, add 'automatic'
    // keyword. If the op has a struct type, 'logic' keyword is already emitted
    // within a struct type definition (e.g. struct packed {logic foo;}). So we
    // should not emit extra 'logic'.
    bool hasStruct = hasStructType(op->getResult(0).getType());
    if (isProcedural)
      return hasStruct ? "automatic" : "automatic logic";
    return hasStruct ? "" : "logic";
  }

  if (!isProcedural)
    return "wire";

  // "automatic" values aren't allowed in disallowLocalVariables mode.
  assert(!options.disallowLocalVariables && "automatic variables not allowed");

  // If the type contains a struct type, we have to use only "automatic" because
  // "automatic struct" is syntactically correct.
  return hasStructType(op->getResult(0).getType()) ? "automatic"
                                                   : "automatic logic";
}

/// Pull any FileLineCol locs out of the specified location and add it to the
/// specified set.
// NOLINTBEGIN(misc-no-recursion)
static void collectFileLineColLocs(Location loc,
                                   SmallPtrSetImpl<Attribute> &locationSet) {
  if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
    locationSet.insert(fileLoc);

  if (auto fusedLoc = loc.dyn_cast<FusedLoc>())
    for (auto loc : fusedLoc.getLocations())
      collectFileLineColLocs(loc, locationSet);
}
// NOLINTEND(misc-no-recursion)

/// Return the location information as a (potentially empty) string.
static std::string
getLocationInfoAsStringImpl(const SmallPtrSetImpl<Attribute> &locationSet) {
  std::string resultStr;
  llvm::raw_string_ostream sstr(resultStr);

  auto printLoc = [&](FileLineColLoc loc) {
    sstr << loc.getFilename().getValue();
    if (auto line = loc.getLine()) {
      sstr << ':' << line;
      if (auto col = loc.getColumn())
        sstr << ':' << col;
    }
  };

  // Fast pass some common cases.
  switch (locationSet.size()) {
  case 1:
    printLoc((*locationSet.begin()).cast<FileLineColLoc>());
    [[fallthrough]];
  case 0:
    return sstr.str();
  default:
    break;
  }

  // Sort the entries.
  SmallVector<FileLineColLoc, 8> locVector;
  locVector.reserve(locationSet.size());
  for (auto loc : locationSet)
    locVector.push_back(loc.cast<FileLineColLoc>());

  llvm::array_pod_sort(
      locVector.begin(), locVector.end(),
      [](const FileLineColLoc *lhs, const FileLineColLoc *rhs) -> int {
        if (auto fn = lhs->getFilename().compare(rhs->getFilename()))
          return fn;
        if (lhs->getLine() != rhs->getLine())
          return lhs->getLine() < rhs->getLine() ? -1 : 1;
        return lhs->getColumn() < rhs->getColumn() ? -1 : 1;
      });

  // The entries are sorted by filename, line, col.  Try to merge together
  // entries to reduce verbosity on the column info.
  StringRef lastFileName;
  for (size_t i = 0, e = locVector.size(); i != e;) {
    if (i != 0)
      sstr << ", ";

    // Print the filename if it changed.
    auto first = locVector[i];
    if (first.getFilename() != lastFileName) {
      lastFileName = first.getFilename();
      sstr << lastFileName;
    }

    // Scan for entries with the same file/line.
    size_t end = i + 1;
    while (end != e && first.getFilename() == locVector[end].getFilename() &&
           first.getLine() == locVector[end].getLine())
      ++end;

    // If we have one entry, print it normally.
    if (end == i + 1) {
      if (auto line = first.getLine()) {
        sstr << ':' << line;
        if (auto col = first.getColumn())
          sstr << ':' << col;
      }
      ++i;
      continue;
    }

    // Otherwise print a brace enclosed list.
    sstr << ':' << first.getLine() << ":{";
    while (i != end) {
      sstr << locVector[i++].getColumn();

      if (i != end)
        sstr << ',';
    }
    sstr << '}';
  }
  return sstr.str();
}

static std::string
getLocationInfoAsStringImpl(const SmallPtrSetImpl<Attribute> &locationSet,
                            LoweringOptions::LocationInfoStyle style) {

  if (style == LoweringOptions::LocationInfoStyle::None)
    return "";
  auto str = getLocationInfoAsStringImpl(locationSet);
  if (str.empty() || style == LoweringOptions::LocationInfoStyle::Plain)
    return str;
  assert(style == LoweringOptions::LocationInfoStyle::WrapInAtSquareBracket &&
         "other styles must be already handled");
  return "@[" + str + "]";
}

/// Return the location information in the specified style.
static std::string
getLocationInfoAsString(Location loc,
                        LoweringOptions::LocationInfoStyle style) {
  SmallPtrSet<Attribute, 8> locationSet;
  collectFileLineColLocs(loc, locationSet);
  return getLocationInfoAsStringImpl(locationSet, style);
}

/// Return the location information in the specified style.
static std::string
getLocationInfoAsString(const SmallPtrSetImpl<Operation *> &ops,
                        LoweringOptions::LocationInfoStyle style) {
  // Multiple operations may come from the same location or may not have useful
  // location info.  Unique it now.
  SmallPtrSet<Attribute, 8> locationSet;
  for (auto *op : ops)
    collectFileLineColLocs(op->getLoc(), locationSet);
  return getLocationInfoAsStringImpl(locationSet, style);
}

/// Most expressions are invalid to bit-select from in Verilog, but some
/// things are ok.  Return true if it is ok to inline bitselect from the
/// result of this expression.  It is conservatively correct to return false.
static bool isOkToBitSelectFrom(Value v) {
  // Module ports are always ok to bit select from.
  if (v.isa<BlockArgument>())
    return true;

  // Read_inout is valid to inline for bit-select. See `select` syntax on
  // SV spec A.8.4 (P1174).
  if (auto read = v.getDefiningOp<ReadInOutOp>())
    return true;

  // Aggregate access can be inlined.
  if (isa_and_nonnull<StructExtractOp, UnionExtractOp, ArrayGetOp>(
          v.getDefiningOp()))
    return true;

  // Interface signal can be inlined.
  if (v.getDefiningOp<ReadInterfaceSignalOp>())
    return true;

  // TODO: We could handle concat and other operators here.
  return false;
}

/// Return true if we are unable to ever inline the specified operation.  This
/// happens because not all Verilog expressions are composable, notably you
/// can only use bit selects like x[4:6] on simple expressions, you cannot use
/// expressions in the sensitivity list of always blocks, etc.
static bool isExpressionUnableToInline(Operation *op,
                                       const LoweringOptions &options) {
  if (auto cast = dyn_cast<BitcastOp>(op))
    if (!haveMatchingDims(cast.getInput().getType(), cast.getResult().getType(),
                          op->getLoc())) {
      // Even if dimentions don't match, we can inline when its user doesn't
      // rely on the type.
      if (op->hasOneUse() &&
          isa<comb::ConcatOp, hw::ArrayConcatOp>(*op->getUsers().begin()))
        return false;
      // Bitcasts rely on the type being assigned to, so we cannot inline.
      return true;
    }

  // StructCreateOp needs to be assigning to a named temporary so that types
  // are inferred properly by verilog
  if (isa<StructCreateOp, UnionCreateOp>(op))
    return true;

  // Aggregate literal syntax only works in an assignment expression, where
  // the Verilog expression's type is determined by the LHS.
  if (auto aggConstantOp = dyn_cast<AggregateConstantOp>(op))
    return true;

  // Verbatim with a long string should be emitted as an out-of-line declration.
  if (auto verbatim = dyn_cast<VerbatimExprOp>(op))
    if (verbatim.getFormatString().size() > 32)
      return true;

  // Scan the users of the operation to see if any of them need this to be
  // emitted out-of-line.
  for (auto *user : op->getUsers()) {
    // Verilog bit selection is required by the standard to be:
    // "a vector, packed array, packed structure, parameter or concatenation".
    //
    // It cannot be an arbitrary expression, e.g. this is invalid:
    //     assign bar = {{a}, {b}, {c}, {d}}[idx];
    //
    // To handle these, we push the subexpression into a temporary.
    if (isa<ExtractOp, ArraySliceOp, ArrayGetOp, StructExtractOp,
            UnionExtractOp, IndexedPartSelectOp>(user))
      if (op->getResult(0) == user->getOperand(0) && // ignore index operands.
          !isOkToBitSelectFrom(op->getResult(0)))
        return true;

    // Always blocks must have a name in their sensitivity list, not an expr.
    if (!options.allowExprInEventControl && isa<AlwaysOp, AlwaysFFOp>(user)) {
      // Anything other than a read of a wire must be out of line.
      if (auto read = dyn_cast<ReadInOutOp>(op))
        if (read.getInput().getDefiningOp<sv::WireOp>() ||
            read.getInput().getDefiningOp<RegOp>())
          continue;
      return true;
    }
  }
  return false;
}

enum class BlockStatementCount { Zero, One, TwoOrMore };

/// Compute how many statements are within this block, for begin/end markers.
static BlockStatementCount countStatements(Block &block) {
  unsigned numStatements = 0;
  block.walk([&](Operation *op) {
    if (isVerilogExpression(op) || isa<ltl::LTLDialect>(op->getDialect()))
      return WalkResult::advance();
    numStatements +=
        TypeSwitch<Operation *, unsigned>(op)
            .Case<VerbatimOp>([&](auto) {
              // We don't know how many statements we emitted, so assume
              // conservatively that a lot got put out. This will make sure we
              // get a begin/end block around this.
              return 3;
            })
            .Case<IfOp>([&](auto) {
              // We count if as multiple statements to make sure it is always
              // surrounded by a begin/end so we don't get if/else confusion in
              // cases like this:
              // if (cond)
              //   if (otherCond)    // This should force a begin!
              //     stmt
              // else                // Goes with the outer if!
              //   thing;
              return 2;
            })
            .Case<IfDefOp, IfDefProceduralOp>([&](auto) { return 3; })
            .Case<OutputOp>([&](OutputOp oop) {
              // Skip single-use instance outputs, they don't get statements.
              // Keep this synchronized with visitStmt(InstanceOp,OutputOp).
              return llvm::count_if(oop->getOperands(), [&](auto operand) {
                return !operand.hasOneUse() ||
                       !dyn_cast_or_null<InstanceOp>(operand.getDefiningOp());
              });
            })
            .Default([](auto) { return 1; });
    if (numStatements > 1)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (numStatements == 0)
    return BlockStatementCount::Zero;
  if (numStatements == 1)
    return BlockStatementCount::One;
  return BlockStatementCount::TwoOrMore;
}

/// Return true if this expression should be emitted inline into any statement
/// that uses it.
bool ExportVerilog::isExpressionEmittedInline(Operation *op,
                                              const LoweringOptions &options) {
  // Never create a temporary for a dead expression.
  if (op->getResult(0).use_empty())
    return true;

  // Never create a temporary which is only going to be assigned to an output
  // port, wire, or reg.
  if (op->hasOneUse() &&
      isa<hw::OutputOp, sv::AssignOp, sv::BPAssignOp, sv::PAssignOp>(
          *op->getUsers().begin()))
    return true;

  // If mux inlining is dissallowed, we cannot inline muxes.
  if (options.disallowMuxInlining && isa<MuxOp>(op))
    return false;

  // If this operation has multiple uses, we can't generally inline it unless
  // the op is duplicatable.
  if (!op->getResult(0).hasOneUse() && !isDuplicatableExpression(op))
    return false;

  // If it isn't structurally possible to inline this expression, emit it out
  // of line.
  return !isExpressionUnableToInline(op, options);
}

/// Find a nested IfOp in an else block that can be printed as `else if`
/// instead of nesting it into a new `begin` - `end` block.  The block must
/// contain a single IfOp and optionally expressions which can be hoisted out.
static IfOp findNestedElseIf(Block *elseBlock) {
  IfOp ifOp;
  for (auto &op : *elseBlock) {
    if (auto opIf = dyn_cast<IfOp>(op)) {
      if (ifOp)
        return {};
      ifOp = opIf;
      continue;
    }
    if (!isVerilogExpression(&op))
      return {};
  }
  // SV attributes cannot be attached to `else if` so reject when ifOp has SV
  // attributes.
  if (ifOp && hasSVAttributes(ifOp))
    return {};
  return ifOp;
}

/// Emit SystemVerilog attributes.
template <typename PPS>
static void emitSVAttributesImpl(PPS &ps, ArrayAttr attrs, bool mayBreak) {
  enum Container { NoContainer, InComment, InAttr };
  Container currentContainer = NoContainer;

  auto closeContainer = [&] {
    if (currentContainer == NoContainer)
      return;
    if (currentContainer == InComment)
      ps << " */";
    else if (currentContainer == InAttr)
      ps << " *)";
    ps << PP::end << PP::end;

    currentContainer = NoContainer;
  };

  bool isFirstContainer = true;
  auto openContainer = [&](Container newContainer) {
    assert(newContainer != NoContainer);
    if (currentContainer == newContainer)
      return false;
    closeContainer();
    // If not first container, insert break point but no space.
    if (!isFirstContainer)
      ps << (mayBreak ? PP::space : PP::nbsp);
    isFirstContainer = false;
    // fit container on one line if possible, break if needed.
    ps << PP::ibox0;
    if (newContainer == InComment)
      ps << "/* ";
    else if (newContainer == InAttr)
      ps << "(* ";
    currentContainer = newContainer;
    // Pack attributes within to fit, align to current column when breaking.
    ps << PP::ibox0;
    return true;
  };

  // Break containers to starting column (0), put all on same line OR
  // put each on their own line (cbox).
  ps.scopedBox(PP::cbox0, [&]() {
    for (auto attr : attrs.getAsRange<SVAttributeAttr>()) {
      if (!openContainer(attr.getEmitAsComment().getValue() ? InComment
                                                            : InAttr))
        ps << "," << (mayBreak ? PP::space : PP::nbsp);
      ps << PPExtString(attr.getName().getValue());
      if (attr.getExpression())
        ps << " = " << PPExtString(attr.getExpression().getValue());
    }
    closeContainer();
  });
}

/// Retrieve value's verilog name from IR. The name must already have been
/// added in pre-pass and passed through "hw.verilogName" attr.
StringRef getVerilogValueName(Value val) {
  if (auto *op = val.getDefiningOp())
    return getSymOpName(op);

  if (auto port = val.dyn_cast<BlockArgument>()) {
    // If the value is defined by for op, use its associated verilog name.
    if (auto forOp = dyn_cast<ForOp>(port.getParentBlock()->getParentOp()))
      return forOp->getAttrOfType<StringAttr>("hw.verilogName");
    return getPortVerilogName(port.getParentBlock()->getParentOp(),
                              port.getArgNumber());
  }
  assert(false && "unhandled value");
  return {};
}

//===----------------------------------------------------------------------===//
// VerilogEmitterState
//===----------------------------------------------------------------------===//

namespace {

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class VerilogEmitterState {
public:
  explicit VerilogEmitterState(ModuleOp designOp,
                               const SharedEmitterState &shared,
                               const LoweringOptions &options,
                               const HWSymbolCache &symbolCache,
                               const GlobalNameTable &globalNames,
                               raw_ostream &os)
      : designOp(designOp), shared(shared), options(options),
        symbolCache(symbolCache), globalNames(globalNames), os(os),
        pp(os, options.emittedLineLength) {
    pp.setListener(&saver);
  }
  /// This is the root mlir::ModuleOp that holds the whole design being emitted.
  ModuleOp designOp;

  const SharedEmitterState &shared;

  /// The emitter options which control verilog emission.
  const LoweringOptions &options;

  /// This is a cache of various information about the IR, in frozen state.
  const HWSymbolCache &symbolCache;

  /// This tracks global names where the Verilog name needs to be different than
  /// the IR name.
  const GlobalNameTable &globalNames;

  /// The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  /// Pretty printing:

  /// Whether a newline is expected, emitted late to provide opportunity to
  /// open/close boxes we don't know we need at level of individual statement.
  /// Every statement should set this instead of directly emitting (last)
  /// newline. Most statements end with emitLocationInfoAndNewLine which handles
  /// this.
  bool pendingNewline = false;

  /// String storage backing Tokens built from temporary strings.
  /// PrettyPrinter will clear this as appropriate.
  TokenStringSaver saver;

  /// Pretty printer.
  PrettyPrinter pp;

private:
  VerilogEmitterState(const VerilogEmitterState &) = delete;
  void operator=(const VerilogEmitterState &) = delete;
};
} // namespace

//===----------------------------------------------------------------------===//
// EmitterBase
//===----------------------------------------------------------------------===//

namespace {

class EmitterBase {
public:
  // All of the mutable state we are maintaining.
  VerilogEmitterState &state;

  /// Stream helper (pp, saver).
  TokenStream<> ps;

  explicit EmitterBase(VerilogEmitterState &state)
      : state(state), ps(state.pp, state.saver) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  InFlightDiagnostic emitOpError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitOpError(message);
  }

  void emitLocationImpl(const std::string &location) {
    // Break so previous content is not impacted by following,
    // but use a 'neverbreak' so it always fits.
    ps << PP::neverbreak;
    if (!location.empty())
      ps << "\t// " << location; // (don't use tabs in normal pretty-printing)
  }

  void emitLocationInfo(Location loc) {
    emitLocationImpl(
        getLocationInfoAsString(loc, state.options.locationInfoStyle));
  }

  /// If we have location information for any of the specified operations,
  /// aggregate it together and print a pretty comment specifying where the
  /// operations came from.  In any case, print a newline.
  void emitLocationInfoAndNewLine(const SmallPtrSetImpl<Operation *> &ops) {
    emitLocationImpl(
        getLocationInfoAsString(ops, state.options.locationInfoStyle));
    setPendingNewline();
  }

  template <typename PPS>
  void emitTextWithSubstitutions(PPS &ps, StringRef string, Operation *op,
                                 llvm::function_ref<void(Value)> operandEmitter,
                                 ArrayAttr symAttrs);

  /// Emit the value of a StringAttr as one or more Verilog "one-line" comments
  /// ("//").  Break the comment to respect the emittedLineLength and trim
  /// whitespace after a line break.  Do nothing if the StringAttr is null or
  /// the value is empty.
  void emitComment(StringAttr comment);

  /// If previous emission requires a newline, emit it now.
  /// This gives us opportunity to open/close boxes before linebreak.
  void emitPendingNewlineIfNeeded() {
    if (state.pendingNewline) {
      state.pendingNewline = false;
      ps << PP::newline;
    }
  }
  void setPendingNewline() {
    assert(!state.pendingNewline);
    state.pendingNewline = true;
  }

  void startStatement() { emitPendingNewlineIfNeeded(); }

private:
  void operator=(const EmitterBase &) = delete;
  EmitterBase(const EmitterBase &) = delete;
};
} // end anonymous namespace

template <typename PPS>
void EmitterBase::emitTextWithSubstitutions(
    PPS &ps, StringRef string, Operation *op,
    llvm::function_ref<void(Value)> operandEmitter, ArrayAttr symAttrs) {

  // Perform operand substitions as we emit the line string.  We turn {{42}}
  // into the value of operand 42.
  auto namify = [&](Attribute sym, HWSymbolCache::Item item) {
    // CAVEAT: These accesses can reach into other modules through inner name
    // references, which are currently being processed. Do not add those remote
    // operations to this module's `names`, which is reserved for things named
    // *within* this module. Instead, you have to rely on those remote
    // operations to have been named inside the global names table. If they
    // haven't, take a look at name legalization first.
    if (auto *itemOp = item.getOp()) {
      if (item.hasPort()) {
        return getPortVerilogName(itemOp, item.getPort());
      }
      StringRef symOpName = getSymOpName(itemOp);
      if (!symOpName.empty())
        return symOpName;
      emitError(itemOp, "cannot get name for symbol ") << sym;
    } else {
      emitError(op, "cannot get name for symbol ") << sym;
    }
    return StringRef("<INVALID>");
  };

  // Scan 'line' for a substitution, emitting any non-substitution prefix,
  // then the mentioned operand, chopping the relevant text off 'line' and
  // returning true.  This returns false if no substitution is found.
  unsigned numSymOps = symAttrs.size();
  auto emitUntilSubstitution = [&](size_t next = 0) -> bool {
    size_t start = 0;
    while (true) {
      next = string.find("{{", next);
      if (next == StringRef::npos)
        return false;

      // Check to make sure we have a number followed by }}.  If not, we
      // ignore the {{ sequence as something that could happen in Verilog.
      next += 2;
      start = next;
      while (next < string.size() && isdigit(string[next]))
        ++next;
      // We need at least one digit.
      if (start == next) {
        next--;
        continue;
      }

      // We must have a }} right after the digits.
      if (!string.substr(next).startswith("}}"))
        continue;

      // We must be able to decode the integer into an unsigned.
      unsigned operandNo = 0;
      if (string.drop_front(start)
              .take_front(next - start)
              .getAsInteger(10, operandNo)) {
        emitError(op, "operand substitution too large");
        continue;
      }
      next += 2;

      // Emit any text before the substitution.
      auto before = string.take_front(start - 2);
      if (!before.empty())
        ps << PPExtString(before);

      // operandNo can either refer to Operands or symOps.  symOps are
      // numbered after the operands.
      if (operandNo < op->getNumOperands())
        // Emit the operand.
        operandEmitter(op->getOperand(operandNo));
      else if ((operandNo - op->getNumOperands()) < numSymOps) {
        unsigned symOpNum = operandNo - op->getNumOperands();
        auto sym = symAttrs[symOpNum];
        StringRef symVerilogName;
        if (auto fsym = sym.dyn_cast<FlatSymbolRefAttr>()) {
          if (auto *symOp = state.symbolCache.getDefinition(fsym)) {
            if (auto globalRef = dyn_cast<HierPathOp>(symOp)) {
              auto namepath = globalRef.getNamepathAttr().getValue();
              for (auto [index, sym] : llvm::enumerate(namepath)) {
                if (index > 0)
                  ps << ".";

                auto innerRef = cast<InnerRefAttr>(sym);
                auto ref = state.symbolCache.getInnerDefinition(
                    innerRef.getModule(), innerRef.getName());
                ps << namify(innerRef, ref);
              }
            } else {
              symVerilogName = namify(sym, symOp);
            }
          }
        } else if (auto isym = sym.dyn_cast<InnerRefAttr>()) {
          auto symOp = state.symbolCache.getInnerDefinition(isym.getModule(),
                                                            isym.getName());
          symVerilogName = namify(sym, symOp);
        }
        if (!symVerilogName.empty())
          ps << PPExtString(symVerilogName);
      } else {
        emitError(op, "operand " + llvm::utostr(operandNo) + " isn't valid");
        continue;
      }
      // Forget about the part we emitted.
      string = string.drop_front(next);
      return true;
    }
  };

  // Emit all the substitutions.
  while (emitUntilSubstitution())
    ;

  // Emit any text after the last substitution.
  if (!string.empty())
    ps << PPExtString(string);
}

void EmitterBase::emitComment(StringAttr comment) {
  if (!comment)
    return;

  // Set a line length for the comment.  Subtract off the leading comment and
  // space ("// ") as well as the current indent level to simplify later
  // arithmetic.  Ensure that this line length doesn't go below zero.
  auto lineLength = state.options.emittedLineLength - state.currentIndent - 3;
  if (lineLength > state.options.emittedLineLength)
    lineLength = 0;

  // Process the comment in line chunks extracted from manually specified line
  // breaks.  This is done to preserve user-specified line breaking if used.
  auto ref = comment.getValue();
  StringRef line;
  while (!ref.empty()) {
    std::tie(line, ref) = ref.split("\n");
    // Emit each comment line breaking it if it exceeds the emittedLineLength.
    for (;;) {
      startStatement();
      ps << "// ";

      // Base case 1: the entire comment fits on one line.
      if (line.size() <= lineLength) {
        ps << PPExtString(line);
        setPendingNewline();
        break;
      }

      // The comment does NOT fit on one line.  Use a simple algorithm to find
      // a position to break the line:
      //   1) Search backwards for whitespace and break there if you find it.
      //   2) If no whitespace exists in (1), search forward for whitespace
      //      and break there.
      // This algorithm violates the emittedLineLength if (2) ever occurrs,
      // but it's dead simple.
      auto breakPos = line.rfind(' ', lineLength);
      // No whitespace exists looking backwards.
      if (breakPos == StringRef::npos) {
        breakPos = line.find(' ', lineLength);
        // No whitespace exists looking forward (you hit the end of the
        // string).
        if (breakPos == StringRef::npos)
          breakPos = line.size();
      }

      // Emit up to the break position.  Trim any whitespace after the break
      // position.  Exit if nothing is left to emit.  Otherwise, update the
      // comment ref and continue;
      ps << PPExtString(line.take_front(breakPos));
      setPendingNewline();
      breakPos = line.find_first_not_of(' ', breakPos);
      // Base Case 2: nothing left except whitespace.
      if (breakPos == StringRef::npos)
        break;

      line = line.drop_front(breakPos);
    }
  }
}

/// Given an expression that is spilled into a temporary wire, try to synthesize
/// a better name than "_T_42" based on the structure of the expression.
// NOLINTBEGIN(misc-no-recursion)
StringAttr ExportVerilog::inferStructuralNameForTemporary(Value expr) {
  StringAttr result;
  bool addPrefixUnderScore = true;

  // Look through read_inout.
  if (auto read = expr.getDefiningOp<ReadInOutOp>())
    return inferStructuralNameForTemporary(read.getInput());

  // Module ports carry names!
  if (auto blockArg = expr.dyn_cast<BlockArgument>()) {
    auto moduleOp = cast<HWModuleOp>(blockArg.getOwner()->getParentOp());
    StringRef name = getPortVerilogName(moduleOp, blockArg.getArgNumber());
    result = StringAttr::get(expr.getContext(), name);

  } else if (auto *op = expr.getDefiningOp()) {
    // Uses of a wire, register or logic can be done inline.
    if (isa<sv::WireOp, RegOp, LogicOp>(op)) {
      StringRef name = getSymOpName(op);
      result = StringAttr::get(expr.getContext(), name);

    } else if (auto nameHint = op->getAttrOfType<StringAttr>("sv.namehint")) {
      // Use a dialect (sv) attribute to get a hint for the name if the op
      // doesn't explicitly specify it. Do this last
      result = nameHint;

      // If there is a namehint, don't add underscores to the name.
      addPrefixUnderScore = false;
    } else {
      TypeSwitch<Operation *>(op)
          // Generate a pretty name for VerbatimExpr's that look macro-like
          // using the same logic that generates the MLIR syntax name.
          .Case([&result](VerbatimExprOp verbatim) {
            verbatim.getAsmResultNames([&](Value, StringRef name) {
              result = StringAttr::get(verbatim.getContext(), name);
            });
          })
          .Case([&result](VerbatimExprSEOp verbatim) {
            verbatim.getAsmResultNames([&](Value, StringRef name) {
              result = StringAttr::get(verbatim.getContext(), name);
            });
          })

          // If this is an extract from a namable object, derive a name from it.
          .Case([&result](ExtractOp extract) {
            if (auto operandName =
                    inferStructuralNameForTemporary(extract.getInput())) {
              unsigned numBits =
                  extract.getType().cast<IntegerType>().getWidth();
              if (numBits == 1)
                result = StringAttr::get(extract.getContext(),
                                         operandName.strref() + "_" +
                                             Twine(extract.getLowBit()));
              else
                result = StringAttr::get(
                    extract.getContext(),
                    operandName.strref() + "_" +
                        Twine(extract.getLowBit() + numBits - 1) + "to" +
                        Twine(extract.getLowBit()));
            }
          });
      // TODO: handle other common patterns.
    }
  }

  // Make sure any synthesized name starts with an _.
  if (!result || result.strref().empty())
    return {};

  // Make sure that all temporary names start with an underscore.
  if (addPrefixUnderScore && result.strref().front() != '_')
    result = StringAttr::get(expr.getContext(), "_" + result.strref());

  return result;
}
// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// ModuleEmitter
//===----------------------------------------------------------------------===//

namespace {

class ModuleEmitter : public EmitterBase {
public:
  explicit ModuleEmitter(VerilogEmitterState &state)
      : EmitterBase(state),
        fieldNameResolver(FieldNameResolver(state.globalNames)) {}
  ~ModuleEmitter() {
    emitPendingNewlineIfNeeded();
    ps.eof();
  };

  void emitParameters(Operation *module, ArrayAttr params);
  void emitPortList(Operation *module, const ModulePortInfo &portInfo);

  void emitHWModule(HWModuleOp module);
  void emitHWTestModule(HWTestModuleOp module);
  void emitHWExternModule(HWModuleExternOp module);
  void emitHWGeneratedModule(HWModuleGeneratedOp module);

  // Statements.
  void emitStatement(Operation *op);
  void emitBind(BindOp op);
  void emitBindInterface(BindInterfaceOp op);

  void emitSVAttributes(Operation *op);

  /// Legalize the given field name if it is an invalid verilog name.
  StringRef getVerilogStructFieldName(StringAttr field) {
    return fieldNameResolver.getRenamedFieldName(field).getValue();
  }

  //===--------------------------------------------------------------------===//
  // Methods for formatting types.

  /// Emit a type's packed dimensions.
  void emitTypeDims(Type type, Location loc, raw_ostream &os);

  /// Print the specified packed portion of the type to the specified stream,
  ///
  ///  * 'optionalAliasType' can be provided to perform any alias-aware printing
  ///    of the inner type.
  ///  * When `implicitIntType` is false, a "logic" is printed.  This is used in
  ///        struct fields and typedefs.
  ///  * When `singleBitDefaultType` is false, single bit values are printed as
  ///       `[0:0]`.  This is used in parameter lists.
  ///
  /// This returns true if anything was printed.
  bool printPackedType(Type type, raw_ostream &os, Location loc,
                       Type optionalAliasType = {}, bool implicitIntType = true,
                       bool singleBitDefaultType = true);

  /// Output the unpacked array dimensions.  This is the part of the type that
  /// is to the right of the name.
  void printUnpackedTypePostfix(Type type, raw_ostream &os);

  //===--------------------------------------------------------------------===//
  // Methods for formatting parameters.

  /// Prints a parameter attribute expression in a Verilog compatible way to the
  /// specified stream.  This returns the precedence of the generated string.
  SubExprInfo printParamValue(Attribute value, raw_ostream &os,
                              function_ref<InFlightDiagnostic()> emitError);

  SubExprInfo printParamValue(Attribute value, raw_ostream &os,
                              VerilogPrecedence parenthesizeIfLooserThan,
                              function_ref<InFlightDiagnostic()> emitError);

  //===--------------------------------------------------------------------===//
  // Mutable state while emitting a module body.

  /// This is the current module being emitted for a HWModuleOp.
  Operation *currentModuleOp;

  /// This set keeps track of expressions that were emitted into their
  /// 'automatic logic' or 'localparam' declaration.  This is only used for
  /// expressions in a procedural region, because we otherwise just emit wires
  /// on demand.
  SmallPtrSet<Operation *, 16> expressionsEmittedIntoDecl;

  /// This class keeps track of field name renamings in the module scope.
  FieldNameResolver fieldNameResolver;

  /// This keeps track of assignments folded into wire emissions
  SmallPtrSet<Operation *, 16> assignsInlined;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Methods for formatting types.

/// Emit a list of dimensions.
static void emitDims(ArrayRef<Attribute> dims, raw_ostream &os, Location loc,
                     ModuleEmitter &emitter) {
  for (Attribute width : dims) {
    if (!width) {
      os << "<<invalid type>>";
      continue;
    }
    if (auto intAttr = width.dyn_cast<IntegerAttr>()) {
      if (intAttr.getValue().isZero())
        os << "/*Zero Width*/";
      else
        os << '[' << (intAttr.getValue().getZExtValue() - 1) << ":0]";
      continue;
    }

    // Otherwise it must be a parameterized dimension.  Shove the "-1" into the
    // attribute so it gets printed in canonical form.
    auto typedAttr = width.dyn_cast<TypedAttr>();
    if (!typedAttr) {
      mlir::emitError(loc, "untyped dimension attribute ") << width;
      continue;
    }
    auto negOne = getIntAttr(
        loc.getContext(), typedAttr.getType(),
        APInt(typedAttr.getType().getIntOrFloatBitWidth(), -1L, true));
    width = ParamExprAttr::get(PEO::Add, typedAttr, negOne);
    os << '[';
    emitter.printParamValue(width, os, [loc]() {
      return mlir::emitError(loc, "invalid parameter in type");
    });
    os << ":0]";
  }
}

/// Emit a type's packed dimensions.
void ModuleEmitter::emitTypeDims(Type type, Location loc, raw_ostream &os) {
  SmallVector<Attribute, 4> dims;
  getTypeDims(dims, type, loc);
  emitDims(dims, os, loc, *this);
}

/// Output the basic type that consists of packed and primitive types.  This is
/// those to the left of the name in verilog. implicitIntType controls whether
/// to print a base type for (logic) for inteters or whether the caller will
/// have handled this (with logic, wire, reg, etc).
/// optionalAliasType can be provided to perform any necessary alias-aware
/// printing of 'type'.
///
/// Returns true when anything was printed out.
// NOLINTBEGIN(misc-no-recursion)
static bool printPackedTypeImpl(Type type, raw_ostream &os, Location loc,
                                SmallVectorImpl<Attribute> &dims,
                                bool implicitIntType, bool singleBitDefaultType,
                                ModuleEmitter &emitter,
                                Type optionalAliasType = {}) {
  return TypeSwitch<Type, bool>(type)
      .Case<IntegerType>([&](IntegerType integerType) {
        if (!implicitIntType)
          os << "logic";
        if (integerType.getWidth() != 1 || !singleBitDefaultType)
          dims.push_back(
              getInt32Attr(type.getContext(), integerType.getWidth()));
        if (!dims.empty() && !implicitIntType)
          os << ' ';

        emitDims(dims, os, loc, emitter);
        return !dims.empty() || !implicitIntType;
      })
      .Case<IntType>([&](IntType intType) {
        if (!implicitIntType)
          os << "logic ";
        dims.push_back(intType.getWidth());
        emitDims(dims, os, loc, emitter);
        return true;
      })
      .Case<ArrayType>([&](ArrayType arrayType) {
        dims.push_back(arrayType.getSizeAttr());
        return printPackedTypeImpl(arrayType.getElementType(), os, loc, dims,
                                   implicitIntType, singleBitDefaultType,
                                   emitter);
      })
      .Case<InOutType>([&](InOutType inoutType) {
        return printPackedTypeImpl(inoutType.getElementType(), os, loc, dims,
                                   implicitIntType, singleBitDefaultType,
                                   emitter);
      })
      .Case<EnumType>([&](EnumType enumType) {
        os << "enum ";
        if (enumType.getBitWidth() != 32)
          os << "bit [" << enumType.getBitWidth() - 1 << ":0] ";
        os << "{";
        Type enumPrefixType = optionalAliasType ? optionalAliasType : enumType;
        llvm::interleaveComma(
            enumType.getFields().getAsRange<StringAttr>(), os,
            [&](auto enumerator) {
              os << emitter.fieldNameResolver.getEnumFieldName(
                  hw::EnumFieldAttr::get(loc, enumerator, enumPrefixType));
            });
        os << "}";
        return true;
      })
      .Case<StructType>([&](StructType structType) {
        if (structType.getElements().empty() || isZeroBitType(structType)) {
          os << "/*Zero Width*/";
          return true;
        }
        os << "struct packed {";
        for (auto &element : structType.getElements()) {
          if (isZeroBitType(element.type)) {
            os << "/*" << emitter.getVerilogStructFieldName(element.name)
               << ": Zero Width;*/ ";
            continue;
          }
          SmallVector<Attribute, 8> structDims;
          printPackedTypeImpl(stripUnpackedTypes(element.type), os, loc,
                              structDims,
                              /*implicitIntType=*/false,
                              /*singleBitDefaultType=*/true, emitter);
          os << ' ' << emitter.getVerilogStructFieldName(element.name);
          emitter.printUnpackedTypePostfix(element.type, os);
          os << "; ";
        }
        os << '}';
        emitDims(dims, os, loc, emitter);
        return true;
      })
      .Case<UnionType>([&](UnionType unionType) {
        if (unionType.getElements().empty() || isZeroBitType(unionType)) {
          os << "/*Zero Width*/";
          return true;
        }

        int64_t unionWidth = hw::getBitWidth(unionType);
        os << "union packed {";
        for (auto &element : unionType.getElements()) {
          if (isZeroBitType(element.type)) {
            os << "/*" << emitter.getVerilogStructFieldName(element.name)
               << ": Zero Width;*/ ";
            continue;
          }
          int64_t elementWidth = hw::getBitWidth(element.type);
          bool needsPadding = elementWidth < unionWidth || element.offset > 0;
          if (needsPadding) {
            os << " struct packed {";
            if (element.offset) {
              os << "logic [" << element.offset - 1 << ":0] "
                 << "__pre_padding_" << element.name.getValue() << "; ";
            }
          }

          SmallVector<Attribute, 8> structDims;
          printPackedTypeImpl(stripUnpackedTypes(element.type), os, loc,
                              structDims,
                              /*implicitIntType=*/false,
                              /*singleBitDefaultType=*/true, emitter);
          os << ' ' << emitter.getVerilogStructFieldName(element.name);
          emitter.printUnpackedTypePostfix(element.type, os);
          os << ";";

          if (needsPadding) {
            if (elementWidth + (int64_t)element.offset < unionWidth) {
              os << " logic ["
                 << unionWidth - (elementWidth + element.offset) - 1 << ":0] "
                 << "__post_padding_" << element.name.getValue() << ";";
            }
            os << "} " << emitter.getVerilogStructFieldName(element.name)
               << ";";
          }
        }
        os << '}';
        emitDims(dims, os, loc, emitter);
        return true;
      })

      .Case<InterfaceType>([](InterfaceType ifaceType) { return false; })
      .Case<UnpackedArrayType>([&](UnpackedArrayType arrayType) {
        os << "<<unexpected unpacked array>>";
        mlir::emitError(loc, "Unexpected unpacked array in packed type ")
            << arrayType;
        return true;
      })
      .Case<TypeAliasType>([&](TypeAliasType typeRef) {
        auto typedecl = typeRef.getTypeDecl(emitter.state.symbolCache);
        if (!typedecl) {
          mlir::emitError(loc, "unresolvable type reference");
          return false;
        }
        if (typedecl.getType() != typeRef.getInnerType()) {
          mlir::emitError(loc, "declared type did not match aliased type");
          return false;
        }

        os << typedecl.getPreferredName();
        emitDims(dims, os, typedecl->getLoc(), emitter);
        return true;
      })
      .Default([&](Type type) {
        os << "<<invalid type '" << type << "'>>";
        mlir::emitError(loc, "value has an unsupported verilog type ") << type;
        return true;
      });
}
// NOLINTEND(misc-no-recursion)

/// Print the specified packed portion of the type to the specified stream,
///
///  * When `implicitIntType` is false, a "logic" is printed.  This is used in
///        struct fields and typedefs.
///  * When `singleBitDefaultType` is false, single bit values are printed as
///       `[0:0]`.  This is used in parameter lists.
///
/// This returns true if anything was printed.
bool ModuleEmitter::printPackedType(Type type, raw_ostream &os, Location loc,
                                    Type optionalAliasType,
                                    bool implicitIntType,
                                    bool singleBitDefaultType) {
  SmallVector<Attribute, 8> packedDimensions;
  return printPackedTypeImpl(type, os, loc, packedDimensions, implicitIntType,
                             singleBitDefaultType, *this, optionalAliasType);
}

/// Output the unpacked array dimensions.  This is the part of the type that is
/// to the right of the name.
// NOLINTBEGIN(misc-no-recursion)
void ModuleEmitter::printUnpackedTypePostfix(Type type, raw_ostream &os) {
  TypeSwitch<Type, void>(type)
      .Case<InOutType>([&](InOutType inoutType) {
        printUnpackedTypePostfix(inoutType.getElementType(), os);
      })
      .Case<UnpackedArrayType>([&](UnpackedArrayType arrayType) {
        os << "[0:" << (arrayType.getSize() - 1) << "]";
        printUnpackedTypePostfix(arrayType.getElementType(), os);
      })
      .Case<InterfaceType>([&](auto) {
        // Interface instantiations have parentheses like a module with no
        // ports.
        os << "()";
      });
}
// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// Methods for formatting parameters.

/// Prints a parameter attribute expression in a Verilog compatible way to the
/// specified stream.  This returns the precedence of the generated string.
SubExprInfo
ModuleEmitter::printParamValue(Attribute value, raw_ostream &os,
                               function_ref<InFlightDiagnostic()> emitError) {
  return printParamValue(value, os, VerilogPrecedence::LowestPrecedence,
                         emitError);
}

/// Helper that prints a parameter constant value in a Verilog compatible way.
/// This returns the precedence of the generated string.
// NOLINTBEGIN(misc-no-recursion)
SubExprInfo
ModuleEmitter::printParamValue(Attribute value, raw_ostream &os,
                               VerilogPrecedence parenthesizeIfLooserThan,
                               function_ref<InFlightDiagnostic()> emitError) {
  if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
    IntegerType intTy = intAttr.getType().cast<IntegerType>();
    APInt value = intAttr.getValue();

    // We omit the width specifier if the value is <= 32-bits in size, which
    // makes this more compatible with unknown width extmodules.
    if (intTy.getWidth() > 32) {
      // Sign comes out before any width specifier.
      if (value.isNegative() && (intTy.isSigned() || intTy.isSignless())) {
        os << '-';
        value = -value;
      }
      if (intTy.isSigned())
        os << intTy.getWidth() << "'sd";
      else
        os << intTy.getWidth() << "'d";
    }
    value.print(os, intTy.isSigned());
    return {Symbol, intTy.isSigned() ? IsSigned : IsUnsigned};
  }
  if (auto strAttr = value.dyn_cast<StringAttr>()) {
    os << '"';
    os.write_escaped(strAttr.getValue());
    os << '"';
    return {Symbol, IsUnsigned};
  }
  if (auto fpAttr = value.dyn_cast<FloatAttr>()) {
    // TODO: relying on float printing to be precise is not a good idea.
    os << fpAttr.getValueAsDouble();
    return {Symbol, IsUnsigned};
  }
  if (auto verbatimParam = value.dyn_cast<ParamVerbatimAttr>()) {
    os << verbatimParam.getValue().getValue();
    return {Symbol, IsUnsigned};
  }
  if (auto parameterRef = value.dyn_cast<ParamDeclRefAttr>()) {
    // Get the name of this parameter (in case it got renamed).
    os << state.globalNames.getParameterVerilogName(currentModuleOp,
                                                    parameterRef.getName());

    // TODO: Should we support signed parameters?
    return {Symbol, IsUnsigned};
  }

  // Handle nested expressions.
  auto expr = value.dyn_cast<ParamExprAttr>();
  if (!expr) {
    os << "<<UNKNOWN MLIRATTR: " << value << ">>";
    emitError() << " = " << value;
    return {LowestPrecedence, IsUnsigned};
  }

  StringRef operatorStr;
  StringRef openStr, closeStr;
  VerilogPrecedence subprecedence = LowestPrecedence;
  VerilogPrecedence prec; // precedence of the emitted expression.
  std::optional<SubExprSignResult> operandSign;
  bool isUnary = false;
  bool hasOpenClose = false;

  switch (expr.getOpcode()) {
  case PEO::Add:
    operatorStr = " + ";
    subprecedence = Addition;
    break;
  case PEO::Mul:
    operatorStr = " * ";
    subprecedence = Multiply;
    break;
  case PEO::And:
    operatorStr = " & ";
    subprecedence = And;
    break;
  case PEO::Or:
    operatorStr = " | ";
    subprecedence = Or;
    break;
  case PEO::Xor:
    operatorStr = " ^ ";
    subprecedence = Xor;
    break;
  case PEO::Shl:
    operatorStr = " << ";
    subprecedence = Shift;
    break;
  case PEO::ShrU:
    // >> in verilog is always a logical shift even if operands are signed.
    operatorStr = " >> ";
    subprecedence = Shift;
    break;
  case PEO::ShrS:
    // >>> in verilog is an arithmetic shift if both operands are signed.
    operatorStr = " >>> ";
    subprecedence = Shift;
    operandSign = IsSigned;
    break;
  case PEO::DivU:
    operatorStr = " / ";
    subprecedence = Multiply;
    operandSign = IsUnsigned;
    break;
  case PEO::DivS:
    operatorStr = " / ";
    subprecedence = Multiply;
    operandSign = IsSigned;
    break;
  case PEO::ModU:
    operatorStr = " % ";
    subprecedence = Multiply;
    operandSign = IsUnsigned;
    break;
  case PEO::ModS:
    operatorStr = " % ";
    subprecedence = Multiply;
    operandSign = IsSigned;
    break;
  case PEO::CLog2:
    openStr = "$clog2(";
    closeStr = ")";
    operandSign = IsUnsigned;
    hasOpenClose = true;
    prec = Symbol;
    break;
  case PEO::StrConcat:
    openStr = "{";
    closeStr = "}";
    hasOpenClose = true;
    operatorStr = ", ";
    // We don't have Concat precedence, but it's lowest anyway. (SV Table 11-2).
    subprecedence = LowestPrecedence;
    prec = Symbol;
    break;
  }
  if (!hasOpenClose)
    prec = subprecedence;

  // unary -> one element.
  assert(!isUnary || llvm::hasSingleElement(expr.getOperands()));
  // one element -> {unary || open/close}.
  assert(isUnary || hasOpenClose ||
         !llvm::hasSingleElement(expr.getOperands()));

  // Emit the specified operand with a $signed() or $unsigned() wrapper around
  // it if context requires a specific signedness to compute the right value.
  // This returns true if the operand is signed.
  // TODO: This could try harder to omit redundant casts like the mainline
  // expression emitter.
  auto emitOperand = [&](Attribute operand) -> bool {
    // If surrounding with signed/unsigned, inner expr doesn't need parens.
    auto subprec = operandSign.has_value() ? LowestPrecedence : subprecedence;
    if (operandSign.has_value())
      os << (*operandSign == IsSigned ? "$signed(" : "$unsigned(");
    auto signedness =
        printParamValue(operand, os, subprec, emitError).signedness;
    if (operandSign.has_value()) {
      os << ')';
      signedness = *operandSign;
    }
    return signedness == IsSigned;
  };

  // Check outer precedence, wrap in parentheses if needed.
  if (prec > parenthesizeIfLooserThan)
    os << '(';

  // Emit opening portion of the operation.
  if (hasOpenClose)
    os << openStr;
  else if (isUnary)
    os << operatorStr;

  bool allOperandsSigned = emitOperand(expr.getOperands()[0]);
  for (auto op : expr.getOperands().drop_front()) {
    // Handle the special case of (a + b + -42) as (a + b - 42).
    // TODO: Also handle (a + b + x*-1).
    if (expr.getOpcode() == PEO::Add) {
      if (auto integer = op.dyn_cast<IntegerAttr>()) {
        const APInt &value = integer.getValue();
        if (value.isNegative() && !value.isMinSignedValue()) {
          os << " - ";
          allOperandsSigned &=
              emitOperand(IntegerAttr::get(op.getType(), -value));
          continue;
        }
      }
    }

    os << operatorStr;
    allOperandsSigned &= emitOperand(op);
  }
  if (hasOpenClose)
    os << closeStr;
  if (prec > parenthesizeIfLooserThan) {
    os << ')';
    prec = Selection;
  }
  return {prec, allOperandsSigned ? IsSigned : IsUnsigned};
}
// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// Expression Emission
//===----------------------------------------------------------------------===//

namespace {
/// This builds a recursively nested expression from an SSA use-def graph.  This
/// uses a post-order walk, but it needs to obey precedence and signedness
/// constraints that depend on the behavior of the child nodes.
/// To handle this, we must buffer all output so we can insert parentheses
/// and other things if we find out that it was needed later.
// NOLINTBEGIN(misc-no-recursion)
class ExprEmitter : public EmitterBase,
                    public TypeOpVisitor<ExprEmitter, SubExprInfo>,
                    public CombinationalVisitor<ExprEmitter, SubExprInfo>,
                    public Visitor<ExprEmitter, SubExprInfo> {
public:
  /// Create an ExprEmitter for the specified module emitter, and keeping track
  /// of any emitted expressions in the specified set.
  ExprEmitter(ModuleEmitter &emitter,
              SmallPtrSetImpl<Operation *> &emittedExprs)
      : ExprEmitter(emitter, emittedExprs, localTokens) {}

  ExprEmitter(ModuleEmitter &emitter,
              SmallPtrSetImpl<Operation *> &emittedExprs,
              BufferingPP::BufferVec &tokens)
      : EmitterBase(emitter.state), emitter(emitter),
        emittedExprs(emittedExprs), buffer(tokens), ps(buffer, state.saver) {
    assert(state.pp.getListener() == &state.saver);
  }

  /// Emit the specified value as an expression.  If this is an inline-emitted
  /// expression, we emit that expression, otherwise we emit a reference to the
  /// already computed name.
  ///
  void emitExpression(Value exp, VerilogPrecedence parenthesizeIfLooserThan) {
    assert(localTokens.empty());
    // Wrap to this column.
    ps.scopedBox(PP::ibox0, [&]() {
      emitSubExpr(exp, parenthesizeIfLooserThan,
                  /*signRequirement*/ NoRequirement,
                  /*isSelfDeterminedUnsignedValue*/ false);
    });
    // If we are not using an external token buffer provided through the
    // constructor, but we're using the default `ExprEmitter`-scoped buffer,
    // flush it.
    if (&buffer.tokens == &localTokens)
      buffer.flush(state.pp);
  }

private:
  friend class TypeOpVisitor<ExprEmitter, SubExprInfo>;
  friend class CombinationalVisitor<ExprEmitter, SubExprInfo>;
  friend class Visitor<ExprEmitter, SubExprInfo>;

  enum SubExprSignRequirement { NoRequirement, RequireSigned, RequireUnsigned };

  /// Emit the specified value `exp` as a subexpression to the stream.  The
  /// `parenthesizeIfLooserThan` parameter indicates when parentheses should be
  /// added aroun the subexpression.  The `signReq` flag can cause emitSubExpr
  /// to emit a subexpression that is guaranteed to be signed or unsigned, and
  /// the `isSelfDeterminedUnsignedValue` flag indicates whether the value is
  /// known to be have "self determined" width, allowing us to omit extensions.
  SubExprInfo emitSubExpr(Value exp, VerilogPrecedence parenthesizeIfLooserThan,
                          SubExprSignRequirement signReq = NoRequirement,
                          bool isSelfDeterminedUnsignedValue = false);

  /// Emit SystemVerilog attributes attached to the expression op as dialect
  /// attributes.
  void emitSVAttributes(Operation *op);

  SubExprInfo visitUnhandledExpr(Operation *op);
  SubExprInfo visitInvalidComb(Operation *op) {
    return dispatchTypeOpVisitor(op);
  }
  SubExprInfo visitUnhandledComb(Operation *op) {
    return visitUnhandledExpr(op);
  }
  SubExprInfo visitInvalidTypeOp(Operation *op) {
    return dispatchSVVisitor(op);
  }
  SubExprInfo visitUnhandledTypeOp(Operation *op) {
    return visitUnhandledExpr(op);
  }
  SubExprInfo visitUnhandledSV(Operation *op) { return visitUnhandledExpr(op); }

  using Visitor::visitSV;

  /// These are flags that control `emitBinary`.
  enum EmitBinaryFlags {
    EB_RequireSignedOperands = RequireSigned,     /* 0x1*/
    EB_RequireUnsignedOperands = RequireUnsigned, /* 0x2*/
    EB_OperandSignRequirementMask = 0x3,

    /// This flag indicates that the RHS operand is an unsigned value that has
    /// "self determined" width.  This means that we can omit explicit zero
    /// extensions from it, and don't impose a sign on it.
    EB_RHS_UnsignedWithSelfDeterminedWidth = 0x4,

    /// This flag indicates that the result should be wrapped in a $signed(x)
    /// expression to force the result to signed.
    EB_ForceResultSigned = 0x8,
  };

  /// Emit a binary expression.  The "emitBinaryFlags" are a bitset from
  /// EmitBinaryFlags.
  SubExprInfo emitBinary(Operation *op, VerilogPrecedence prec,
                         const char *syntax, unsigned emitBinaryFlags = 0);

  SubExprInfo emitUnary(Operation *op, const char *syntax,
                        bool resultAlwaysUnsigned = false);

  /// Emit the specified value as a subexpression, wrapping in an ibox2.
  void emitSubExprIBox2(
      Value v, VerilogPrecedence parenthesizeIfLooserThan = LowestPrecedence) {
    ps.scopedBox(PP::ibox2,
                 [&]() { emitSubExpr(v, parenthesizeIfLooserThan); });
  }

  /// Emit a range of values separated by commas and a breakable space.
  /// Each value is emitted by invoking `eachFn`.
  template <typename Container, typename EachFn>
  void interleaveComma(const Container &c, EachFn eachFn) {
    llvm::interleave(c, eachFn, [&]() { ps << "," << PP::space; });
  }

  /// Emit a range of values separated by commas and a breakable space.
  /// Each value is emitted in an ibox2.
  void interleaveComma(ValueRange ops) {
    return interleaveComma(ops, [&](Value v) { emitSubExprIBox2(v); });
  }

  /// Emit an array-literal-like structure, separated by commas.
  /// Use callbacks to emit open tokens, closing tokens, and handle each value.
  /// If it fits, will be emitted on a single line with no space between
  /// list and surrounding open and close.
  /// Otherwise, each item is placed on its own line.
  /// This has property that if any element requires breaking, all elements
  /// are emitted on separate lines (with open/close attached to first/last).
  /// `{a + b, x + y, c}`
  /// OR
  /// ```
  /// {a + b,
  ///  x + y,
  ///  c}
  /// ```
  template <typename Container, typename OpenFunc, typename CloseFunc,
            typename EachFunc>
  void emitBracedList(const Container &c, OpenFunc openFn, EachFunc eachFn,
                      CloseFunc closeFn) {
    openFn();
    ps.scopedBox(PP::cbox0, [&]() {
      interleaveComma(c, eachFn);
      closeFn();
    });
  }

  /// Emit braced list of values surrounded by specified open/close.
  template <typename OpenFunc, typename CloseFunc>
  void emitBracedList(ValueRange ops, OpenFunc openFn, CloseFunc closeFn) {
    return emitBracedList(
        ops, openFn, [&](Value v) { emitSubExprIBox2(v); }, closeFn);
  }

  /// Emit braced list of values surrounded by `{` and `}`.
  void emitBracedList(ValueRange ops) {
    return emitBracedList(
        ops, [&]() { ps << "{"; }, [&]() { ps << "}"; });
  }

  /// Print an APInt constant.
  SubExprInfo printConstantScalar(APInt &value, IntegerType type);

  SubExprInfo visitSV(GetModportOp op);
  SubExprInfo visitSV(SystemFunctionOp op);
  SubExprInfo visitSV(ReadInterfaceSignalOp op);
  SubExprInfo visitSV(XMROp op);
  SubExprInfo visitSV(XMRRefOp op);
  SubExprInfo visitVerbatimExprOp(Operation *op, ArrayAttr symbols);
  SubExprInfo visitSV(VerbatimExprOp op) {
    return visitVerbatimExprOp(op, op.getSymbols());
  }
  SubExprInfo visitSV(VerbatimExprSEOp op) {
    return visitVerbatimExprOp(op, op.getSymbols());
  }
  SubExprInfo visitSV(MacroRefExprOp op);
  SubExprInfo visitSV(MacroRefExprSEOp op);
  template <typename MacroTy>
  SubExprInfo emitMacroCall(MacroTy op);

  SubExprInfo visitSV(ConstantXOp op);
  SubExprInfo visitSV(ConstantZOp op);
  SubExprInfo visitSV(ConstantStrOp op);

  // Noop cast operators.
  SubExprInfo visitSV(ReadInOutOp op) {
    auto result = emitSubExpr(op->getOperand(0), LowestPrecedence);
    emitSVAttributes(op);
    return result;
  }
  SubExprInfo visitSV(ArrayIndexInOutOp op);
  SubExprInfo visitSV(IndexedPartSelectInOutOp op);
  SubExprInfo visitSV(IndexedPartSelectOp op);
  SubExprInfo visitSV(StructFieldInOutOp op);

  // Sampled value functions
  SubExprInfo visitSV(SampledOp op);

  // Other
  using TypeOpVisitor::visitTypeOp;
  SubExprInfo visitTypeOp(ConstantOp op);
  SubExprInfo visitTypeOp(AggregateConstantOp op);
  SubExprInfo visitTypeOp(BitcastOp op);
  SubExprInfo visitTypeOp(ParamValueOp op);
  SubExprInfo visitTypeOp(ArraySliceOp op);
  SubExprInfo visitTypeOp(ArrayGetOp op);
  SubExprInfo visitTypeOp(ArrayCreateOp op);
  SubExprInfo visitTypeOp(ArrayConcatOp op);
  SubExprInfo visitTypeOp(StructCreateOp op);
  SubExprInfo visitTypeOp(StructExtractOp op);
  SubExprInfo visitTypeOp(StructInjectOp op);
  SubExprInfo visitTypeOp(UnionCreateOp op);
  SubExprInfo visitTypeOp(UnionExtractOp op);
  SubExprInfo visitTypeOp(EnumCmpOp op);
  SubExprInfo visitTypeOp(EnumConstantOp op);

  // Comb Dialect Operations
  using CombinationalVisitor::visitComb;
  SubExprInfo visitComb(MuxOp op);
  SubExprInfo visitComb(AddOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Addition, "+");
  }
  SubExprInfo visitComb(SubOp op) { return emitBinary(op, Addition, "-"); }
  SubExprInfo visitComb(MulOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Multiply, "*");
  }
  SubExprInfo visitComb(DivUOp op) {
    return emitBinary(op, Multiply, "/", EB_RequireUnsignedOperands);
  }
  SubExprInfo visitComb(DivSOp op) {
    return emitBinary(op, Multiply, "/", EB_RequireSignedOperands);
  }
  SubExprInfo visitComb(ModUOp op) {
    return emitBinary(op, Multiply, "%", EB_RequireUnsignedOperands);
  }
  SubExprInfo visitComb(ModSOp op) {
    return emitBinary(op, Multiply, "%", EB_RequireSignedOperands);
  }
  SubExprInfo visitComb(ShlOp op) {
    return emitBinary(op, Shift, "<<", EB_RHS_UnsignedWithSelfDeterminedWidth);
  }
  SubExprInfo visitComb(ShrUOp op) {
    // >> in Verilog is always an unsigned right shift.
    return emitBinary(op, Shift, ">>", EB_RHS_UnsignedWithSelfDeterminedWidth);
  }
  SubExprInfo visitComb(ShrSOp op) {
    // >>> is only an arithmetic shift right when both operands are signed.
    // Otherwise it does a logical shift.
    return emitBinary(op, Shift, ">>>",
                      EB_RequireSignedOperands | EB_ForceResultSigned |
                          EB_RHS_UnsignedWithSelfDeterminedWidth);
  }
  SubExprInfo visitComb(AndOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, And, "&");
  }
  SubExprInfo visitComb(OrOp op) {
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Or, "|");
  }
  SubExprInfo visitComb(XorOp op) {
    if (op.isBinaryNot())
      return emitUnary(op, "~");
    assert(op.getNumOperands() == 2 && "prelowering should handle variadics");
    return emitBinary(op, Xor, "^");
  }

  // SystemVerilog spec 11.8.1: "Reduction operator results are unsigned,
  // regardless of the operands."
  SubExprInfo visitComb(ParityOp op) { return emitUnary(op, "^", true); }

  SubExprInfo visitComb(ReplicateOp op);
  SubExprInfo visitComb(ConcatOp op);
  SubExprInfo visitComb(ExtractOp op);
  SubExprInfo visitComb(ICmpOp op);

public:
  ModuleEmitter &emitter;

private:
  /// This is set (before a visit method is called) if emitSubExpr would
  /// prefer to get an output of a specific sign.  This is a hint to cause the
  /// visitor to change its emission strategy, but the visit method can ignore
  /// it without a correctness problem.
  SubExprSignRequirement signPreference = NoRequirement;

  /// Keep track of all operations emitted within this subexpression for
  /// location information tracking.
  SmallPtrSetImpl<Operation *> &emittedExprs;

  /// Tokens buffered for inserting casts/parens after emitting children.
  SmallVector<Token> localTokens;

  /// Stores tokens until told to flush.  Uses provided buffer (tokens).
  BufferingPP buffer;

  /// Stream to emit expressions into, will add to buffer.
  TokenStream<BufferingPP> ps;
};
} // end anonymous namespace

SubExprInfo ExprEmitter::emitBinary(Operation *op, VerilogPrecedence prec,
                                    const char *syntax,
                                    unsigned emitBinaryFlags) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  // It's tempting to wrap expressions in groups as we emit them,
  // but that can cause bad wrapping as-is:
  // add(a, add(b, add(c, add(d, e))))
  //   ->
  // group(a + (group(b + group(c + group(d + e)))))
  // Which will break after 'a +' first.
  // TODO: Build tree capturing precedence/fixity at same level, group those!
  // Maybe like: https://www.tweag.io/blog/2022-02-10-ormolu-and-operators/ .
  // For now, only group within punctuation, such as parens + braces.
  if (emitBinaryFlags & EB_ForceResultSigned)
    ps << "$signed(" << PP::ibox0;
  auto operandSignReq =
      SubExprSignRequirement(emitBinaryFlags & EB_OperandSignRequirementMask);
  auto lhsInfo = emitSubExpr(op->getOperand(0), prec, operandSignReq);
  // Bit of a kludge: if this is a comparison, don't break on either side.
  auto lhsSpace = prec == VerilogPrecedence::Comparison ? PP::nbsp : PP::space;
  // Use non-breaking space between op and RHS so breaking is consistent.
  ps << lhsSpace << syntax << PP::nbsp; // PP::space;

  // Right associative operators are already generally variadic, we need to
  // handle things like: (a<4> == b<4>) == (c<3> == d<3>).  When processing the
  // top operation of the tree, the rhs needs parens.  When processing
  // known-reassociative operators like +, ^, etc we don't need parens.
  // TODO: MLIR should have general "Associative" trait.
  auto rhsPrec = prec;
  if (!isa<AddOp, MulOp, AndOp, OrOp, XorOp>(op))
    rhsPrec = VerilogPrecedence(prec - 1);

  // If the RHS operand has self-determined width and always treated as
  // unsigned, inform emitSubExpr of this.  This is true for the shift amount in
  // a shift operation.
  bool rhsIsUnsignedValueWithSelfDeterminedWidth = false;
  if (emitBinaryFlags & EB_RHS_UnsignedWithSelfDeterminedWidth) {
    rhsIsUnsignedValueWithSelfDeterminedWidth = true;
    operandSignReq = NoRequirement;
  }

  auto rhsInfo = emitSubExpr(op->getOperand(1), rhsPrec, operandSignReq,
                             rhsIsUnsignedValueWithSelfDeterminedWidth);

  // SystemVerilog 11.8.1 says that the result of a binary expression is signed
  // only if both operands are signed.
  SubExprSignResult signedness = IsUnsigned;
  if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
    signedness = IsSigned;

  if (emitBinaryFlags & EB_ForceResultSigned) {
    ps << PP::end << ")";
    signedness = IsSigned;
    prec = Selection;
  }

  return {prec, signedness};
}

SubExprInfo ExprEmitter::emitUnary(Operation *op, const char *syntax,
                                   bool resultAlwaysUnsigned) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  ps << syntax;
  auto signedness = emitSubExpr(op->getOperand(0), Selection).signedness;
  // For reduction operators "&" and "|", make precedence lowest to avoid
  // emitting an expression like `a & &b`, which is syntactically valid but some
  // tools produce LINT warnings.
  return {isa<ICmpOp>(op) ? LowestPrecedence : Unary,
          resultAlwaysUnsigned ? IsUnsigned : signedness};
}

/// Emit SystemVerilog attributes attached to the expression op as dialect
/// attributes.
void ExprEmitter::emitSVAttributes(Operation *op) {
  // SystemVerilog 2017 Section 5.12.
  auto svAttrs = getSVAttributes(op);
  if (!svAttrs)
    return;

  // For now, no breaks for attributes.
  ps << PP::nbsp;
  emitSVAttributesImpl(ps, svAttrs, /*mayBreak=*/false);
}

/// If the specified extension is a zero extended version of another value,
/// return the shorter value, otherwise return null.
static Value isZeroExtension(Value value) {
  auto concat = value.getDefiningOp<ConcatOp>();
  if (!concat || concat.getNumOperands() != 2)
    return {};

  auto constant = concat.getOperand(0).getDefiningOp<ConstantOp>();
  if (constant && constant.getValue().isZero())
    return concat.getOperand(1);
  return {};
}

/// Emit the specified value `exp` as a subexpression to the stream.  The
/// `parenthesizeIfLooserThan` parameter indicates when parentheses should be
/// added aroun the subexpression.  The `signReq` flag can cause emitSubExpr
/// to emit a subexpression that is guaranteed to be signed or unsigned, and
/// the `isSelfDeterminedUnsignedValue` flag indicates whether the value is
/// known to be have "self determined" width, allowing us to omit extensions.
SubExprInfo ExprEmitter::emitSubExpr(Value exp,
                                     VerilogPrecedence parenthesizeIfLooserThan,
                                     SubExprSignRequirement signRequirement,
                                     bool isSelfDeterminedUnsignedValue) {
  // If this is a self-determined unsigned value, look through any inline zero
  // extensions.  This occurs on the RHS of a shift operation for example.
  if (isSelfDeterminedUnsignedValue && exp.hasOneUse()) {
    if (auto smaller = isZeroExtension(exp))
      exp = smaller;
  }

  auto *op = exp.getDefiningOp();
  bool shouldEmitInlineExpr = op && isVerilogExpression(op);

  // If this is a non-expr or shouldn't be done inline, just refer to its name.
  if (!shouldEmitInlineExpr) {
    // All wires are declared as unsigned, so if the client needed it signed,
    // emit a conversion.
    if (signRequirement == RequireSigned) {
      ps << "$signed(" << PPExtString(getVerilogValueName(exp)) << ")";
      return {Symbol, IsSigned};
    }

    ps << PPExtString(getVerilogValueName(exp));
    return {Symbol, IsUnsigned};
  }

  unsigned subExprStartIndex = buffer.tokens.size();

  // Inform the visit method about the preferred sign we want from the result.
  // It may choose to ignore this, but some emitters can change behavior based
  // on contextual desired sign.
  signPreference = signRequirement;

  bool bitCastAdded = false;
  if (state.options.explicitBitcast && isa<AddOp, MulOp, SubOp>(op))
    if (auto inType =
            (op->getResult(0).getType().dyn_cast_or_null<IntegerType>())) {
      ps.addAsString(inType.getWidth());
      ps << "'(" << PP::ibox0;
      bitCastAdded = true;
    }
  // Okay, this is an expression we should emit inline.  Do this through our
  // visitor.
  auto expInfo = dispatchCombinationalVisitor(exp.getDefiningOp());

  // Check cases where we have to insert things before the expression now that
  // we know things about it.
  auto addPrefix = [&](StringToken &&t) {
    // insert {Prefix, ibox0}.
    buffer.tokens.insert(buffer.tokens.begin() + subExprStartIndex,
                         BeginToken(0));
    buffer.tokens.insert(buffer.tokens.begin() + subExprStartIndex, t);
  };
  auto closeBoxAndParen = [&]() { ps << PP::end << ")"; };
  if (signRequirement == RequireSigned && expInfo.signedness == IsUnsigned) {
    addPrefix(StringToken("$signed("));
    closeBoxAndParen();
    expInfo.signedness = IsSigned;
    expInfo.precedence = Selection;
  } else if (signRequirement == RequireUnsigned &&
             expInfo.signedness == IsSigned) {
    addPrefix(StringToken("$unsigned("));
    closeBoxAndParen();
    expInfo.signedness = IsUnsigned;
    expInfo.precedence = Selection;
  } else if (expInfo.precedence > parenthesizeIfLooserThan) {
    // If this subexpression would bind looser than the expression it is bound
    // into, then we need to parenthesize it.  Insert the parentheses
    // retroactively.
    addPrefix(StringToken("("));
    closeBoxAndParen();
    // Reset the precedence to the () level.
    expInfo.precedence = Selection;
  }
  if (bitCastAdded) {
    closeBoxAndParen();
  }

  // Remember that we emitted this.
  emittedExprs.insert(exp.getDefiningOp());
  return expInfo;
}

SubExprInfo ExprEmitter::visitComb(ReplicateOp op) {
  auto openFn = [&]() {
    ps << "{";
    ps.addAsString(op.getMultiple());
    ps << "{";
  };
  auto closeFn = [&]() { ps << "}}"; };

  // If the subexpression is an inline concat, we can emit it as part of the
  // replicate.
  if (auto concatOp = op.getOperand().getDefiningOp<ConcatOp>()) {
    if (op.getOperand().hasOneUse()) {
      emitBracedList(concatOp.getOperands(), openFn, closeFn);
      return {Symbol, IsUnsigned};
    }
  }
  emitBracedList(op.getOperand(), openFn, closeFn);
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitComb(ConcatOp op) {
  emitBracedList(op.getOperands());
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(BitcastOp op) {
  // NOTE: Bitcasts are emitted out-of-line with their own wire declaration when
  // their dimensions don't match. SystemVerilog uses the wire declaration to
  // know what type this value is being casted to.
  Type toType = op.getType();
  if (!haveMatchingDims(toType, op.getInput().getType(), op.getLoc())) {
    ps << "/*cast(bit";
    ps.invokeWithStringOS(
        [&](auto &os) { emitter.emitTypeDims(toType, op.getLoc(), os); });
    ps << ")*/";
  }
  return emitSubExpr(op.getInput(), LowestPrecedence);
}

SubExprInfo ExprEmitter::visitComb(ICmpOp op) {
  const char *symop[] = {"==", "!=", "<",  "<=",  ">",   ">=",  "<",
                         "<=", ">",  ">=", "===", "!==", "==?", "!=?"};
  SubExprSignRequirement signop[] = {
      // Equality
      NoRequirement, NoRequirement,
      // Signed Comparisons
      RequireSigned, RequireSigned, RequireSigned, RequireSigned,
      // Unsigned Comparisons
      RequireUnsigned, RequireUnsigned, RequireUnsigned, RequireUnsigned,
      // Weird Comparisons
      NoRequirement, NoRequirement, NoRequirement, NoRequirement};

  auto pred = static_cast<uint64_t>(op.getPredicate());
  assert(pred < sizeof(symop) / sizeof(symop[0]));

  // Lower "== -1" to Reduction And.
  if (op.isEqualAllOnes())
    return emitUnary(op, "&", true);

  // Lower "!= 0" to Reduction Or.
  if (op.isNotEqualZero())
    return emitUnary(op, "|", true);

  auto result = emitBinary(op, Comparison, symop[pred], signop[pred]);

  // SystemVerilog 11.8.1: "Comparison... operator results are unsigned,
  // regardless of the operands".
  result.signedness = IsUnsigned;
  return result;
}

SubExprInfo ExprEmitter::visitComb(ExtractOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  unsigned loBit = op.getLowBit();
  unsigned hiBit = loBit + op.getType().cast<IntegerType>().getWidth() - 1;

  auto x = emitSubExpr(op.getInput(), LowestPrecedence);
  assert((x.precedence == Symbol ||
          (x.precedence == Selection && isOkToBitSelectFrom(op.getInput()))) &&
         "should be handled by isExpressionUnableToInline");

  // If we're extracting the whole input, just return it.  This is valid but
  // non-canonical IR, and we don't want to generate invalid Verilog.
  if (loBit == 0 &&
      op.getInput().getType().getIntOrFloatBitWidth() == hiBit + 1)
    return x;

  ps << "[";
  ps.addAsString(hiBit);
  if (hiBit != loBit) { // Emit x[4] instead of x[4:4].
    ps << ":";
    ps.addAsString(loBit);
  }
  ps << "]";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(GetModportOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto decl = op.getReferencedDecl(state.symbolCache);
  ps << PPExtString(getVerilogValueName(op.getIface())) << "."
     << PPExtString(getSymOpName(decl));
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(SystemFunctionOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  ps << "$" << PPExtString(op.getFnName()) << "(";
  ps.scopedBox(PP::ibox0, [&]() {
    llvm::interleave(
        op.getOperands(), [&](Value v) { emitSubExpr(v, LowestPrecedence); },
        [&]() { ps << "," << PP::space; });
    ps << ")";
  });
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ReadInterfaceSignalOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto decl = op.getReferencedDecl(state.symbolCache);

  ps << PPExtString(getVerilogValueName(op.getIface())) << "."
     << PPExtString(getSymOpName(decl));
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(XMROp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  if (op.getIsRooted())
    ps << "$root.";
  for (auto s : op.getPath())
    ps << PPExtString(s.cast<StringAttr>().getValue()) << ".";
  ps << PPExtString(op.getTerminal());
  return {Selection, IsUnsigned};
}

// TODO: This shares a lot of code with the getNameRemotely mtehod. Combine
// these to share logic.
SubExprInfo ExprEmitter::visitSV(XMRRefOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  // The XMR is pointing at a GlobalRef.
  auto globalRef = op.getReferencedPath(&state.symbolCache);
  auto namepath = globalRef.getNamepathAttr().getValue();
  auto *module = state.symbolCache.getDefinition(
      cast<InnerRefAttr>(namepath.front()).getModule());
  ps << PPExtString(getSymOpName(module));
  for (auto sym : namepath) {
    ps << ".";
    auto innerRef = cast<InnerRefAttr>(sym);
    auto ref = state.symbolCache.getInnerDefinition(innerRef.getModule(),
                                                    innerRef.getName());
    if (ref.hasPort()) {
      ps << PPExtString(getPortVerilogName(ref.getOp(), ref.getPort()));
      continue;
    }
    ps << PPExtString(getSymOpName(ref.getOp()));
  }
  auto leaf = op.getVerbatimSuffixAttr();
  if (leaf && leaf.size())
    ps << PPExtString(leaf);
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitVerbatimExprOp(Operation *op, ArrayAttr symbols) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  emitTextWithSubstitutions(
      ps, op->getAttrOfType<StringAttr>("format_string").getValue(), op,
      [&](Value operand) { emitSubExpr(operand, LowestPrecedence); }, symbols);

  return {Unary, IsUnsigned};
}

template <typename MacroTy>
SubExprInfo ExprEmitter::emitMacroCall(MacroTy op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  // Use the specified name or the symbol name as appropriate.
  auto macroOp = op.getReferencedMacro(&state.symbolCache);
  assert(macroOp && "Invalid IR");
  StringRef name =
      macroOp.getVerilogName() ? *macroOp.getVerilogName() : macroOp.getName();
  ps << "`" << PPExtString(name);
  if (!op.getInputs().empty()) {
    ps << "(";
    llvm::interleaveComma(op.getInputs(), ps, [&](Value val) {
      emitExpression(val, LowestPrecedence);
    });
    ps << ")";
  }
  return {LowestPrecedence, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(MacroRefExprOp op) {
  return emitMacroCall(op);
}

SubExprInfo ExprEmitter::visitSV(MacroRefExprSEOp op) {
  return emitMacroCall(op);
}

SubExprInfo ExprEmitter::visitSV(ConstantXOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  ps.addAsString(op.getWidth());
  ps << "'bx";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ConstantStrOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  ps.writeQuotedEscaped(op.getStr());
  return {Symbol, IsUnsigned}; // is a string unsigned?  Yes! SV 5.9
}

SubExprInfo ExprEmitter::visitSV(ConstantZOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  ps.addAsString(op.getWidth());
  ps << "'bz";
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::printConstantScalar(APInt &value, IntegerType type) {
  bool isNegated = false;
  // If this is a negative signed number and not MININT (e.g. -128), then print
  // it as a negated positive number.
  if (signPreference == RequireSigned && value.isNegative() &&
      !value.isMinSignedValue()) {
    ps << "-";
    isNegated = true;
  }

  ps.addAsString(type.getWidth());
  ps << "'";

  // Emit this as a signed constant if the caller would prefer that.
  if (signPreference == RequireSigned)
    ps << "sh";
  else
    ps << "h";

  // Print negated if required.
  SmallString<32> valueStr;
  if (isNegated) {
    (-value).toStringUnsigned(valueStr, 16);
  } else {
    value.toStringUnsigned(valueStr, 16);
  }
  ps << valueStr;
  return {Unary, signPreference == RequireSigned ? IsSigned : IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(ConstantOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto value = op.getValue();
  // We currently only allow zero width values to be handled as special cases in
  // the various operations that may come across them. If we reached this point
  // in the emitter, the value should be considered illegal to emit.
  if (value.getBitWidth() == 0) {
    emitOpError(op, "will not emit zero width constants in the general case");
    ps << "<<unsupported zero width constant: "
       << PPExtString(op->getName().getStringRef()) << ">>";
    return {Unary, IsUnsigned};
  }

  return printConstantScalar(value, op.getType().cast<IntegerType>());
}

SubExprInfo ExprEmitter::visitTypeOp(AggregateConstantOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  // If the constant op as a whole is zero-width, it is an error.
  assert(!isZeroBitType(op.getType()) &&
         "zero-bit types not allowed at this point");

  std::function<void(Attribute, Type)> printAggregate = [&](Attribute attr,
                                                            Type type) {
    if (auto arrayType = hw::type_dyn_cast<ArrayType>(type)) {
      auto elementType = arrayType.getElementType();
      emitBracedList(
          attr.cast<ArrayAttr>(), [&]() { ps << "{"; },
          [&](Attribute attr) { printAggregate(attr, elementType); },
          [&]() { ps << "}"; });
    } else if (auto arrayType = hw::type_dyn_cast<UnpackedArrayType>(type)) {
      auto elementType = arrayType.getElementType();
      emitBracedList(
          attr.cast<ArrayAttr>(), [&]() { ps << "'{"; },
          [&](Attribute attr) { printAggregate(attr, elementType); },
          [&]() { ps << "}"; });
    } else if (auto structType = hw::type_dyn_cast<StructType>(type)) {
      // Only emit elements with non-zero bit width.
      // TODO: Ideally we should emit zero bit values as comments, e.g. `{/*a:
      // ZeroBit,*/ b: foo, /* c: ZeroBit*/ d: bar}`. However it's tedious to
      // nicely emit all edge cases hence currently we just elide zero bit
      // values.
      emitBracedList(
          llvm::make_filter_range(
              llvm::zip(structType.getElements(), attr.cast<ArrayAttr>()),
              [](const auto &fieldAndAttr) {
                // Elide zero bit elements.
                return !isZeroBitType(std::get<0>(fieldAndAttr).type);
              }),
          [&]() { ps << "'{"; },
          [&](const auto &fieldAndAttr) {
            ps.scopedBox(PP::ibox2, [&]() {
              const auto &[field, attr] = fieldAndAttr;
              ps << PPExtString(emitter.getVerilogStructFieldName(field.name))
                 << ":" << PP::space;
              printAggregate(attr, field.type);
            });
          },
          [&]() { ps << "}"; });
    } else if (auto enumType = hw::type_dyn_cast<EnumType>(type)) {
      assert(false && "unsupported");
      auto value = attr.cast<StringAttr>();
      ps << value.getValue();
    } else if (auto intType = hw::type_dyn_cast<IntegerType>(type)) {
      auto value = attr.cast<IntegerAttr>().getValue();
      printConstantScalar(value, intType);
    } else {
      assert(false && "unknown constant kind");
    }
  };

  printAggregate(op.getFields(), op.getType());
  return {Symbol, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(ParamValueOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  return ps.invokeWithStringOS([&](auto &os) {
    return emitter.printParamValue(op.getValue(), os, [&]() {
      return op->emitOpError("invalid parameter use");
    });
  });
}

// 11.5.1 "Vector bit-select and part-select addressing" allows a '+:' syntax
// for slicing operations.
SubExprInfo ExprEmitter::visitTypeOp(ArraySliceOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto arrayPrec = emitSubExpr(op.getInput(), Selection);

  unsigned dstWidth = type_cast<ArrayType>(op.getType()).getSize();
  ps << "[";
  emitSubExpr(op.getLowIndex(), LowestPrecedence);
  ps << " +: ";
  ps.addAsString(dstWidth);
  ps << "]";
  return {Selection, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitTypeOp(ArrayGetOp op) {
  emitSubExpr(op.getInput(), Selection);
  ps << "[";
  if (isZeroBitType(op.getIndex().getType()))
    emitZeroWidthIndexingValue(ps);
  else
    emitSubExpr(op.getIndex(), LowestPrecedence);
  ps << "]";
  emitSVAttributes(op);
  return {Selection, IsUnsigned};
}

// Syntax from: section 5.11 "Array literals".
SubExprInfo ExprEmitter::visitTypeOp(ArrayCreateOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  if (op.isUniform()) {
    ps << "{";
    ps.addAsString(op.getInputs().size());
    ps << "{";
    emitSubExpr(op.getUniformElement(), LowestPrecedence);
    ps << "}}";
  } else {
    emitBracedList(
        op.getInputs(), [&]() { ps << "{"; },
        [&](Value v) {
          ps << "{";
          emitSubExprIBox2(v);
          ps << "}";
        },
        [&]() { ps << "}"; });
  }
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(ArrayConcatOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  emitBracedList(op.getOperands());
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitSV(ArrayIndexInOutOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto index = op.getIndex();
  auto arrayPrec = emitSubExpr(op.getInput(), Selection);
  ps << "[";
  if (isZeroBitType(index.getType()))
    emitZeroWidthIndexingValue(ps);
  else
    emitSubExpr(index, LowestPrecedence);
  ps << "]";
  return {Selection, arrayPrec.signedness};
}

SubExprInfo ExprEmitter::visitSV(IndexedPartSelectInOutOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto prec = emitSubExpr(op.getInput(), Selection);
  ps << "[";
  emitSubExpr(op.getBase(), LowestPrecedence);
  if (op.getDecrement())
    ps << " -: ";
  else
    ps << " +: ";
  ps.addAsString(op.getWidth());
  ps << "]";
  return {Selection, prec.signedness};
}

SubExprInfo ExprEmitter::visitSV(IndexedPartSelectOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto info = emitSubExpr(op.getInput(), LowestPrecedence);
  ps << "[";
  emitSubExpr(op.getBase(), LowestPrecedence);
  if (op.getDecrement())
    ps << " -: ";
  else
    ps << " +: ";
  ps.addAsString(op.getWidth());
  ps << "]";
  return info;
}

SubExprInfo ExprEmitter::visitSV(StructFieldInOutOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto prec = emitSubExpr(op.getInput(), Selection);
  ps << "."
     << PPExtString(emitter.getVerilogStructFieldName(op.getFieldAttr()));
  return {Selection, prec.signedness};
}

SubExprInfo ExprEmitter::visitSV(SampledOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  ps << "$sampled(";
  auto info = emitSubExpr(op.getExpression(), LowestPrecedence);
  ps << ")";
  return info;
}

SubExprInfo ExprEmitter::visitComb(MuxOp op) {
  // The ?: operator is right associative.

  // Layout:
  // cond ? a : b
  // (long
  //  + cond) ? a : b
  // long
  // + cond
  //   ? a : b
  // long
  // + cond
  //   ? a
  //   : b
  return ps.scopedBox(PP::cbox0, [&]() -> SubExprInfo {
    ps.scopedBox(PP::ibox0, [&]() {
      emitSubExpr(op.getCond(), VerilogPrecedence(Conditional - 1));
    });
    ps << BreakToken(1, 2);
    ps << "?";
    emitSVAttributes(op);
    ps << " ";
    auto lhsInfo = ps.scopedBox(PP::ibox0, [&]() {
      return emitSubExpr(op.getTrueValue(), VerilogPrecedence(Conditional - 1));
    });
    ps << BreakToken(1, 2) << ": ";

    auto rhsInfo = ps.scopedBox(PP::ibox0, [&]() {
      return emitSubExpr(op.getFalseValue(), Conditional);
    });

    SubExprSignResult signedness = IsUnsigned;
    if (lhsInfo.signedness == IsSigned && rhsInfo.signedness == IsSigned)
      signedness = IsSigned;

    return {Conditional, signedness};
  });
}

SubExprInfo ExprEmitter::visitTypeOp(StructCreateOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  StructType stype = op.getType();
  // Only emit elements with non-zero bit width.
  // TODO: Ideally we should emit zero bit values as comments, e.g. `{/*a:
  // ZeroBit,*/ b: foo, /* c: ZeroBit*/ d: bar}`. However it's tedious to nicely
  // emit all edge cases hence currently we just elide zero bit values.
  emitBracedList(
      llvm::make_filter_range(llvm::zip(stype.getElements(), op.getOperands()),
                              [](const auto &fieldAndOperand) {
                                // Elide zero bit elements.
                                const auto &[field, _] = fieldAndOperand;
                                return !isZeroBitType(field.type);
                              }),
      [&]() { ps << "'{"; },
      [&](const auto &fieldAndOperand) {
        ps.scopedBox(PP::ibox2, [&]() {
          const auto &[field, operand] = fieldAndOperand;
          ps << PPExtString(emitter.getVerilogStructFieldName(field.name))
             << ":" << PP::space;
          emitSubExpr(operand, Selection);
        });
      },
      [&]() { ps << "}"; });
  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(StructExtractOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  emitSubExpr(op.getInput(), Selection);
  ps << "."
     << PPExtString(emitter.getVerilogStructFieldName(op.getFieldAttr()));
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(StructInjectOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  StructType stype = op.getType().cast<StructType>();
  // Only emit elements with non-zero bit width.
  emitBracedList(

      llvm::make_filter_range(
          stype.getElements(),
          [](const auto &field) { return !isZeroBitType(field.type); }),
      [&]() { ps << "'{"; },
      [&](const StructType::FieldInfo &field) {
        ps.scopedBox(PP::ibox2, [&]() {
          ps << PPExtString(emitter.getVerilogStructFieldName(field.name))
             << ":" << PP::space;
          if (field.name == op.getField()) {
            emitSubExpr(op.getNewValue(), Selection);
          } else {
            emitSubExpr(op.getInput(), Selection);
            ps << "." << PPExtString(field.name.getValue());
          }
        });
      },
      [&]() { ps << "}"; });
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(EnumConstantOp op) {
  ps << PPSaveString(emitter.fieldNameResolver.getEnumFieldName(op.getField()));
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(EnumCmpOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");
  auto result = emitBinary(op, Comparison, "==", NoRequirement);
  // SystemVerilog 11.8.1: "Comparison... operator results are unsigned,
  // regardless of the operands".
  result.signedness = IsUnsigned;
  return result;
}

SubExprInfo ExprEmitter::visitTypeOp(UnionCreateOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  // Check if this union type has been padded.
  auto fieldName = op.getFieldAttr();
  auto unionType = cast<UnionType>(getCanonicalType(op.getType()));
  auto unionWidth = hw::getBitWidth(unionType);
  auto element = unionType.getFieldInfo(fieldName.getValue());
  auto elementWidth = hw::getBitWidth(element.type);

  // If the element is 0 width, just fill the union with 0s.
  if (!elementWidth) {
    ps.addAsString(unionWidth);
    ps << "'h0";
    return {Unary, IsUnsigned};
  }

  // If the element has no padding, emit it directly.
  if (elementWidth == unionWidth) {
    emitSubExpr(op.getInput(), LowestPrecedence);
    return {Unary, IsUnsigned};
  }

  // Emit the value as a bitconcat, supplying 0 for the padding bits.
  ps << "{";
  ps.scopedBox(PP::ibox0, [&]() {
    if (auto prePadding = element.offset) {
      ps.addAsString(prePadding);
      ps << "'h0," << PP::space;
    }
    emitSubExpr(op.getInput(), Selection);
    if (auto postPadding = unionWidth - elementWidth - element.offset) {
      ps << "," << PP::space;
      ps.addAsString(postPadding);
      ps << "'h0";
    }
    ps << "}";
  });

  return {Unary, IsUnsigned};
}

SubExprInfo ExprEmitter::visitTypeOp(UnionExtractOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");
  emitSubExpr(op.getInput(), Selection);

  // Check if this union type has been padded.
  auto fieldName = op.getFieldAttr();
  auto unionType = cast<UnionType>(getCanonicalType(op.getInput().getType()));
  auto unionWidth = hw::getBitWidth(unionType);
  auto element = unionType.getFieldInfo(fieldName.getValue());
  auto elementWidth = hw::getBitWidth(element.type);
  bool needsPadding = elementWidth < unionWidth || element.offset > 0;
  auto verilogFieldName = emitter.getVerilogStructFieldName(fieldName);

  // If the element needs padding then we need to get the actual element out
  // of an anonymous structure.
  if (needsPadding)
    ps << "." << PPExtString(verilogFieldName);

  // Get the correct member from the union.
  ps << "." << PPExtString(verilogFieldName);
  return {Selection, IsUnsigned};
}

SubExprInfo ExprEmitter::visitUnhandledExpr(Operation *op) {
  emitOpError(op, "cannot emit this expression to Verilog");
  ps << "<<unsupported expr: " << PPExtString(op->getName().getStringRef())
     << ">>";
  return {Symbol, IsUnsigned};
}
// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// Property Emission
//===----------------------------------------------------------------------===//

// NOLINTBEGIN(misc-no-recursion)

namespace {
/// Precedence level of various property and sequence expressions. Lower numbers
/// bind tighter.
///
/// See IEEE 1800-2017 section 16.12 "Declaring properties", specifically table
/// 16-3 on "Sequence and property operator precedence and associativity".
enum class PropertyPrecedence {
  Symbol,      // Atomic symbol like `foo` and regular boolean expressions
  Repeat,      // Sequence `[*]`, `[=]`, `[->]`
  Concat,      // Sequence `##`
  Throughout,  // Sequence `throughout`
  Within,      // Sequence `within`
  Intersect,   // Sequence `intersect`
  Unary,       // Property `not`, `nexttime`-like
  And,         // Sequence and property `and`
  Or,          // Sequence and property `or`
  Iff,         // Property `iff`
  Until,       // Property `until`-like, `implies`
  Implication, // Property `|->`, `|=>`, `#-#`, `#=#`
  Qualifier,   // Property `always`-like, `eventually`-like, `if`, `case`,
               // `accept`-like, `reject`-like
  Clocking,    // `@(...)`, `disable iff` (not specified in the standard)
  Lowest,      // Sentinel which is always the lowest precedence.
};

/// Additional information on emitted property and sequence expressions.
struct EmittedProperty {
  /// The precedence of this expression.
  PropertyPrecedence precedence;
};

/// A helper to emit recursively nested property and sequence expressions for
/// SystemVerilog assertions.
class PropertyEmitter : public EmitterBase,
                        public ltl::Visitor<PropertyEmitter, EmittedProperty> {
public:
  /// Create a PropertyEmitter for the specified module emitter, and keeping
  /// track of any emitted expressions in the specified set.
  PropertyEmitter(ModuleEmitter &emitter,
                  SmallPtrSetImpl<Operation *> &emittedOps)
      : PropertyEmitter(emitter, emittedOps, localTokens) {}
  PropertyEmitter(ModuleEmitter &emitter,
                  SmallPtrSetImpl<Operation *> &emittedOps,
                  BufferingPP::BufferVec &tokens)
      : EmitterBase(emitter.state), emitter(emitter), emittedOps(emittedOps),
        buffer(tokens), ps(buffer, state.saver) {
    assert(state.pp.getListener() == &state.saver);
  }

  /// Emit the specified value as an SVA property or sequence. This is the entry
  /// point to print an entire tree of property or sequence expressions in one
  /// go.
  void emitProperty(
      Value property,
      PropertyPrecedence parenthesizeIfLooserThan = PropertyPrecedence::Lowest);

private:
  using ltl::Visitor<PropertyEmitter, EmittedProperty>::visitLTL;
  friend class ltl::Visitor<PropertyEmitter, EmittedProperty>;

  /// Emit the specified value as an SVA property or sequence.
  EmittedProperty
  emitNestedProperty(Value property,
                     PropertyPrecedence parenthesizeIfLooserThan);

  EmittedProperty visitUnhandledLTL(Operation *op);
  EmittedProperty visitLTL(ltl::AndOp op);
  EmittedProperty visitLTL(ltl::OrOp op);
  EmittedProperty visitLTL(ltl::DelayOp op);
  EmittedProperty visitLTL(ltl::ConcatOp op);
  EmittedProperty visitLTL(ltl::NotOp op);
  EmittedProperty visitLTL(ltl::ImplicationOp op);
  EmittedProperty visitLTL(ltl::EventuallyOp op);
  EmittedProperty visitLTL(ltl::ClockOp op);
  EmittedProperty visitLTL(ltl::DisableOp op);

  void emitLTLConcat(ValueRange inputs);

public:
  ModuleEmitter &emitter;

private:
  /// Keep track of all operations emitted within this subexpression for
  /// location information tracking.
  SmallPtrSetImpl<Operation *> &emittedOps;

  /// Tokens buffered for inserting casts/parens after emitting children.
  SmallVector<Token> localTokens;

  /// Stores tokens until told to flush.  Uses provided buffer (tokens).
  BufferingPP buffer;

  /// Stream to emit expressions into, will add to buffer.
  TokenStream<BufferingPP> ps;
};
} // end anonymous namespace

void PropertyEmitter::emitProperty(
    Value property, PropertyPrecedence parenthesizeIfLooserThan) {
  assert(localTokens.empty());
  // Wrap to this column.
  ps.scopedBox(PP::ibox0,
               [&] { emitNestedProperty(property, parenthesizeIfLooserThan); });
  // If we are not using an external token buffer provided through the
  // constructor, but we're using the default `PropertyEmitter`-scoped buffer,
  // flush it.
  if (&buffer.tokens == &localTokens)
    buffer.flush(state.pp);
}

EmittedProperty PropertyEmitter::emitNestedProperty(
    Value property, PropertyPrecedence parenthesizeIfLooserThan) {
  // Emit the property as a plain expression if it doesn't have a property or
  // sequence type, in which case it is just a boolean expression.
  //
  // We use the `LowestPrecedence` for the boolean expression such that it never
  // gets parenthesized. According to IEEE 1800-2017, "the operators described
  // in Table 11-2 have higher precedence than the sequence and property
  // operators". Therefore any boolean expression behaves just like a
  // `PropertyPrecedence::Symbol` and needs no parantheses, which is equivalent
  // to `VerilogPrecedence::LowestPrecedence`.
  if (!isa<ltl::SequenceType, ltl::PropertyType>(property.getType())) {
    ExprEmitter(emitter, emittedOps, buffer.tokens)
        .emitExpression(property, LowestPrecedence);
    return {PropertyPrecedence::Symbol};
  }

  unsigned startIndex = buffer.tokens.size();
  auto info = dispatchLTLVisitor(property.getDefiningOp());

  // If this subexpression would bind looser than the expression it is bound
  // into, then we need to parenthesize it. Insert the parentheses
  // retroactively.
  if (info.precedence > parenthesizeIfLooserThan) {
    // Insert {"(", ibox0} before the subexpression.
    buffer.tokens.insert(buffer.tokens.begin() + startIndex, BeginToken(0));
    buffer.tokens.insert(buffer.tokens.begin() + startIndex, StringToken("("));
    // Insert {end, ")" } after the subexpression.
    ps << PP::end << ")";
    // Reset the precedence level.
    info.precedence = PropertyPrecedence::Symbol;
  }

  // Remember that we emitted this.
  emittedOps.insert(property.getDefiningOp());
  return info;
}

EmittedProperty PropertyEmitter::visitUnhandledLTL(Operation *op) {
  emitOpError(op, "emission as Verilog property or sequence not supported");
  ps << "<<unsupported: " << PPExtString(op->getName().getStringRef()) << ">>";
  return {PropertyPrecedence::Symbol};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::AndOp op) {
  llvm::interleave(
      op.getInputs(),
      [&](auto input) { emitNestedProperty(input, PropertyPrecedence::And); },
      [&]() { ps << PP::space << "and" << PP::nbsp; });
  return {PropertyPrecedence::And};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::OrOp op) {
  llvm::interleave(
      op.getInputs(),
      [&](auto input) { emitNestedProperty(input, PropertyPrecedence::Or); },
      [&]() { ps << PP::space << "or" << PP::nbsp; });
  return {PropertyPrecedence::Or};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::DelayOp op) {
  ps << "##";
  if (auto length = op.getLength()) {
    if (*length == 0) {
      ps.addAsString(op.getDelay());
    } else {
      ps << "[";
      ps.addAsString(op.getDelay());
      ps << ":";
      ps.addAsString(op.getDelay() + *length);
      ps << "]";
    }
  } else {
    if (op.getDelay() == 0) {
      ps << "[*]";
    } else if (op.getDelay() == 1) {
      ps << "[+]";
    } else {
      ps << "[";
      ps.addAsString(op.getDelay());
      ps << ":$]";
    }
  }
  ps << PP::space;
  emitNestedProperty(op.getInput(), PropertyPrecedence::Concat);
  return {PropertyPrecedence::Concat};
}

void PropertyEmitter::emitLTLConcat(ValueRange inputs) {
  bool addSeparator = false;
  for (auto input : inputs) {
    if (addSeparator) {
      ps << PP::space;
      if (!input.getDefiningOp<ltl::DelayOp>())
        ps << "##0" << PP::space;
    }
    addSeparator = true;
    emitNestedProperty(input, PropertyPrecedence::Concat);
  }
}

EmittedProperty PropertyEmitter::visitLTL(ltl::ConcatOp op) {
  emitLTLConcat(op.getInputs());
  return {PropertyPrecedence::Concat};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::NotOp op) {
  ps << "not" << PP::space;
  emitNestedProperty(op.getInput(), PropertyPrecedence::Unary);
  return {PropertyPrecedence::Unary};
}

/// For a value `concat(..., delay(const(true), 1, 0))`, return `...`. This is
/// useful for emitting `(seq ##1 true) |-> prop` as `seq |=> prop`.
static ValueRange getNonOverlappingConcatSubrange(Value value) {
  auto concatOp = value.getDefiningOp<ltl::ConcatOp>();
  if (!concatOp || concatOp.getInputs().size() < 2)
    return {};
  auto delayOp = concatOp.getInputs().back().getDefiningOp<ltl::DelayOp>();
  if (!delayOp || delayOp.getDelay() != 1 || delayOp.getLength() != 0)
    return {};
  auto constOp = delayOp.getInput().getDefiningOp<ConstantOp>();
  if (!constOp || !constOp.getValue().isOne())
    return {};
  return concatOp.getInputs().drop_back();
}

EmittedProperty PropertyEmitter::visitLTL(ltl::ImplicationOp op) {
  // Emit `(seq ##1 true) |-> prop` as `seq |=> prop`.
  if (auto range = getNonOverlappingConcatSubrange(op.getAntecedent());
      !range.empty()) {
    emitLTLConcat(range);
    ps << PP::space << "|=>" << PP::nbsp;
  } else {
    emitNestedProperty(op.getAntecedent(), PropertyPrecedence::Implication);
    ps << PP::space << "|->" << PP::nbsp;
  }
  emitNestedProperty(op.getConsequent(), PropertyPrecedence::Implication);
  return {PropertyPrecedence::Implication};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::EventuallyOp op) {
  ps << "s_eventually" << PP::space;
  emitNestedProperty(op.getInput(), PropertyPrecedence::Qualifier);
  return {PropertyPrecedence::Qualifier};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::ClockOp op) {
  ps << "@(";
  ps.scopedBox(PP::ibox2, [&] {
    ps << PPExtString(stringifyClockEdge(op.getEdge())) << PP::space;
    emitNestedProperty(op.getClock(), PropertyPrecedence::Lowest);
    ps << ")";
  });
  ps << PP::space;
  emitNestedProperty(op.getInput(), PropertyPrecedence::Clocking);
  return {PropertyPrecedence::Clocking};
}

EmittedProperty PropertyEmitter::visitLTL(ltl::DisableOp op) {
  ps << "disable iff" << PP::nbsp << "(";
  ps.scopedBox(PP::ibox2, [&] {
    emitNestedProperty(op.getCondition(), PropertyPrecedence::Lowest);
    ps << ")";
  });
  ps << PP::space;
  emitNestedProperty(op.getInput(), PropertyPrecedence::Clocking);
  return {PropertyPrecedence::Clocking};
}

// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// NameCollector
//===----------------------------------------------------------------------===//

namespace {
class NameCollector {
public:
  NameCollector(ModuleEmitter &moduleEmitter) : moduleEmitter(moduleEmitter) {}

  // Scan operations in the specified block, collecting information about
  // those that need to be emitted as declarations.
  void collectNames(Block &block);

  size_t getMaxDeclNameWidth() const { return maxDeclNameWidth; }
  size_t getMaxTypeWidth() const { return maxTypeWidth; }

private:
  size_t maxDeclNameWidth = 0, maxTypeWidth = 0;
  ModuleEmitter &moduleEmitter;
};
} // namespace

// NOLINTNEXTLINE(misc-no-recursion)
void NameCollector::collectNames(Block &block) {
  // Loop over all of the results of all of the ops. Anything that defines a
  // value needs to be noticed.
  for (auto &op : block) {
    // Instances have an instance name to recognize but we don't need to look
    // at the result values since wires used by instances should be traversed
    // anyway.
    if (isa<InstanceOp, InterfaceInstanceOp>(op))
      continue;
    if (isa<ltl::LTLDialect>(op.getDialect()))
      continue;

    bool isExpr = isVerilogExpression(&op);
    assert((!isExpr ||
            isExpressionEmittedInline(&op, moduleEmitter.state.options)) &&
           "If 'op' is a verilog expression, the expression must be inlinable. "
           "Otherwise, it is a bug of PrepareForEmission");

    if (!isExpr) {
      for (auto result : op.getResults()) {
        StringRef declName =
            getVerilogDeclWord(&op, moduleEmitter.state.options);
        maxDeclNameWidth = std::max(declName.size(), maxDeclNameWidth);
        SmallString<16> typeString;

        // Convert the port's type to a string and measure it.
        {
          llvm::raw_svector_ostream stringStream(typeString);
          moduleEmitter.printPackedType(stripUnpackedTypes(result.getType()),
                                        stringStream, op.getLoc());
        }
        maxTypeWidth = std::max(typeString.size(), maxTypeWidth);
      }
    }

    // Recursively process any regions under the op iff this is a procedural
    // #ifdef region: we need to emit automatic logic values at the top of the
    // enclosing region.
    if (isa<IfDefProceduralOp, OrderedOutputOp>(op)) {
      for (auto &region : op.getRegions()) {
        if (!region.empty())
          collectNames(region.front());
      }
      continue;
    }

    // Recursively process any expressions in else blocks that can be emitted
    // as `else if`.
    if (auto ifOp = dyn_cast<IfOp>(op)) {
      if (ifOp.hasElse() && findNestedElseIf(ifOp.getElseBlock()))
        collectNames(*ifOp.getElseBlock());
      continue;
    }
  }
}

//===----------------------------------------------------------------------===//
// StmtEmitter
//===----------------------------------------------------------------------===//

namespace {
/// This emits statement-related operations.
// NOLINTBEGIN(misc-no-recursion)
class StmtEmitter : public EmitterBase,
                    public hw::StmtVisitor<StmtEmitter, LogicalResult>,
                    public sv::Visitor<StmtEmitter, LogicalResult>,
                    public verif::Visitor<StmtEmitter, LogicalResult> {
public:
  /// Create an ExprEmitter for the specified module emitter, and keeping track
  /// of any emitted expressions in the specified set.
  StmtEmitter(ModuleEmitter &emitter)
      : EmitterBase(emitter.state), emitter(emitter) {}

  void emitStatement(Operation *op);
  void emitStatementBlock(Block &body);

  /// Emit a declaration.
  LogicalResult emitDeclaration(Operation *op);

private:
  void collectNamesAndCalculateDeclarationWidths(Block &block);

  void
  emitExpression(Value exp, SmallPtrSetImpl<Operation *> &emittedExprs,
                 VerilogPrecedence parenthesizeIfLooserThan = LowestPrecedence);
  void emitSVAttributes(Operation *op);

  using hw::StmtVisitor<StmtEmitter, LogicalResult>::visitStmt;
  using sv::Visitor<StmtEmitter, LogicalResult>::visitSV;
  using verif::Visitor<StmtEmitter, LogicalResult>::visitVerif;
  friend class hw::StmtVisitor<StmtEmitter, LogicalResult>;
  friend class sv::Visitor<StmtEmitter, LogicalResult>;
  friend class verif::Visitor<StmtEmitter, LogicalResult>;

  // Visitor methods.
  LogicalResult visitUnhandledStmt(Operation *op) { return failure(); }
  LogicalResult visitInvalidStmt(Operation *op) { return failure(); }
  LogicalResult visitUnhandledSV(Operation *op) { return failure(); }
  LogicalResult visitInvalidSV(Operation *op) { return failure(); }
  LogicalResult visitUnhandledVerif(Operation *op) { return failure(); }
  LogicalResult visitInvalidVerif(Operation *op) { return failure(); }

  LogicalResult visitSV(sv::WireOp op) { return emitDeclaration(op); }
  LogicalResult visitSV(RegOp op) { return emitDeclaration(op); }
  LogicalResult visitSV(LogicOp op) { return emitDeclaration(op); }
  LogicalResult visitSV(LocalParamOp op) { return emitDeclaration(op); }
  template <typename Op>
  LogicalResult
  emitAssignLike(Op op, PPExtString syntax,
                 std::optional<PPExtString> wordBeforeLHS = std::nullopt);
  void emitAssignLike(llvm::function_ref<void()> emitLHS,
                      llvm::function_ref<void()> emitRHS, PPExtString syntax,
                      PPExtString postSyntax = PPExtString(";"),
                      std::optional<PPExtString> wordBeforeLHS = std::nullopt);
  LogicalResult visitSV(AssignOp op);
  LogicalResult visitSV(BPAssignOp op);
  LogicalResult visitSV(PAssignOp op);
  LogicalResult visitSV(ForceOp op);
  LogicalResult visitSV(ReleaseOp op);
  LogicalResult visitSV(AliasOp op);
  LogicalResult visitSV(InterfaceInstanceOp op);
  LogicalResult visitStmt(OutputOp op);
  LogicalResult visitStmt(InstanceOp op);
  LogicalResult visitStmt(TypeScopeOp op);
  LogicalResult visitStmt(TypedeclOp op);

  LogicalResult emitIfDef(Operation *op, MacroIdentAttr cond);
  LogicalResult visitSV(OrderedOutputOp op);
  LogicalResult visitSV(IfDefOp op) { return emitIfDef(op, op.getCond()); }
  LogicalResult visitSV(IfDefProceduralOp op) {
    return emitIfDef(op, op.getCond());
  }
  LogicalResult visitSV(IfOp op);
  LogicalResult visitSV(AlwaysOp op);
  LogicalResult visitSV(AlwaysCombOp op);
  LogicalResult visitSV(AlwaysFFOp op);
  LogicalResult visitSV(InitialOp op);
  LogicalResult visitSV(CaseOp op);
  LogicalResult visitSV(FWriteOp op);
  LogicalResult visitSV(VerbatimOp op);

  LogicalResult emitSimulationControlTask(Operation *op, PPExtString taskName,
                                          std::optional<unsigned> verbosity);
  LogicalResult visitSV(StopOp op);
  LogicalResult visitSV(FinishOp op);
  LogicalResult visitSV(ExitOp op);

  LogicalResult emitSeverityMessageTask(Operation *op, PPExtString taskName,
                                        std::optional<unsigned> verbosity,
                                        StringAttr message,
                                        ValueRange operands);
  LogicalResult visitSV(FatalOp op);
  LogicalResult visitSV(ErrorOp op);
  LogicalResult visitSV(WarningOp op);
  LogicalResult visitSV(InfoOp op);

  LogicalResult visitSV(ReadMemOp op);

  LogicalResult visitSV(GenerateOp op);
  LogicalResult visitSV(GenerateCaseOp op);

  LogicalResult visitSV(ForOp op);

  void emitAssertionLabel(Operation *op);
  void emitAssertionMessage(StringAttr message, ValueRange args,
                            SmallPtrSetImpl<Operation *> &ops,
                            bool isConcurrent);
  template <typename Op>
  LogicalResult emitImmediateAssertion(Op op, PPExtString opName);
  LogicalResult visitSV(AssertOp op);
  LogicalResult visitSV(AssumeOp op);
  LogicalResult visitSV(CoverOp op);
  template <typename Op>
  LogicalResult emitConcurrentAssertion(Op op, PPExtString opName);
  LogicalResult visitSV(AssertConcurrentOp op);
  LogicalResult visitSV(AssumeConcurrentOp op);
  LogicalResult visitSV(CoverConcurrentOp op);

  LogicalResult visitSV(BindOp op);
  LogicalResult visitSV(InterfaceOp op);
  LogicalResult visitSV(InterfaceSignalOp op);
  LogicalResult visitSV(InterfaceModportOp op);
  LogicalResult visitSV(AssignInterfaceSignalOp op);
  LogicalResult visitSV(MacroDefOp op);

  void emitBlockAsStatement(Block *block,
                            const SmallPtrSetImpl<Operation *> &locationOps,
                            StringRef multiLineComment = StringRef());

  LogicalResult emitVerifAssertLike(Operation *op, Value property,
                                    PPExtString opName);
  LogicalResult visitVerif(verif::AssertOp op);
  LogicalResult visitVerif(verif::AssumeOp op);
  LogicalResult visitVerif(verif::CoverOp op);

public:
  ModuleEmitter &emitter;

private:
  /// These keep track of the maximum length of name width and type width in the
  /// current statement scope.
  size_t maxDeclNameWidth = 0;
  size_t maxTypeWidth = 0;
};

} // end anonymous namespace

/// Emit the specified value as an expression.  If this is an inline-emitted
/// expression, we emit that expression, otherwise we emit a reference to the
/// already computed name.
///
void StmtEmitter::emitExpression(Value exp,
                                 SmallPtrSetImpl<Operation *> &emittedExprs,
                                 VerilogPrecedence parenthesizeIfLooserThan) {
  ExprEmitter(emitter, emittedExprs)
      .emitExpression(exp, parenthesizeIfLooserThan);
}

/// Emit SystemVerilog attributes attached to the statement op as dialect
/// attributes.
void StmtEmitter::emitSVAttributes(Operation *op) {
  // SystemVerilog 2017 Section 5.12.
  auto svAttrs = getSVAttributes(op);
  if (!svAttrs)
    return;

  startStatement(); // For attributes.
  emitSVAttributesImpl(ps, svAttrs, /*mayBreak=*/true);
  setPendingNewline();
}

void StmtEmitter::emitAssignLike(llvm::function_ref<void()> emitLHS,
                                 llvm::function_ref<void()> emitRHS,
                                 PPExtString syntax, PPExtString postSyntax,
                                 std::optional<PPExtString> wordBeforeLHS) {
  // If wraps, indent.
  ps.scopedBox(PP::ibox2, [&]() {
    if (wordBeforeLHS) {
      ps << *wordBeforeLHS << PP::space;
    }
    emitLHS();
    // Allow breaking before 'syntax' (e.g., '=') if long assignment.
    ps << PP::space << syntax << PP::space;
    // RHS is boxed to right of the syntax.
    ps.scopedBox(PP::ibox0, [&]() {
      emitRHS();
      ps << postSyntax;
    });
  });
}

template <typename Op>
LogicalResult
StmtEmitter::emitAssignLike(Op op, PPExtString syntax,
                            std::optional<PPExtString> wordBeforeLHS) {
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  startStatement();
  emitAssignLike([&]() { emitExpression(op.getDest(), ops); },
                 [&]() { emitExpression(op.getSrc(), ops); }, syntax,
                 PPExtString(";"), wordBeforeLHS);

  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AssignOp op) {
  // prepare assigns wires to instance outputs, but these are logically handled
  // in the port binding list when outputing an instance.
  if (dyn_cast_or_null<InstanceOp>(op.getSrc().getDefiningOp()))
    return success();

  if (emitter.assignsInlined.count(op))
    return success();

  // Emit SV attributes. See Spec 12.3.
  emitSVAttributes(op);

  return emitAssignLike(op, PPExtString("="), PPExtString("assign"));
}

LogicalResult StmtEmitter::visitSV(BPAssignOp op) {
  // If the assign is emitted into logic declaration, we must not emit again.
  if (emitter.assignsInlined.count(op))
    return success();

  // Emit SV attributes. See Spec 12.3.
  emitSVAttributes(op);

  return emitAssignLike(op, PPExtString("="));
}

LogicalResult StmtEmitter::visitSV(PAssignOp op) {
  // Emit SV attributes. See Spec 12.3.
  emitSVAttributes(op);

  return emitAssignLike(op, PPExtString("<="));
}

LogicalResult StmtEmitter::visitSV(ForceOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  return emitAssignLike(op, PPExtString("="), PPExtString("force"));
}

LogicalResult StmtEmitter::visitSV(ReleaseOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "release" << PP::space;
    emitExpression(op.getDest(), ops);
    ps << ";";
  });
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AliasOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "alias" << PP::space;
    ps.scopedBox(PP::cbox0, [&]() { // If any breaks, all break.
      llvm::interleave(
          op.getOperands(), [&](Value v) { emitExpression(v, ops); },
          [&]() { ps << PP::nbsp << "=" << PP::space; });
      ps << ";";
    });
  });
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceInstanceOp op) {
  auto doNotPrint = op->hasAttr("doNotPrint");
  if (doNotPrint && !state.options.emitBindComments)
    return success();

  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  StringRef prefix = "";
  if (doNotPrint) {
    prefix = "// ";
    ps << "// This interface is elsewhere emitted as a bind statement."
       << PP::newline;
  }

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  auto *interfaceOp = op.getReferencedInterface(&state.symbolCache);
  assert(interfaceOp && "InterfaceInstanceOp has invalid symbol that does not "
                        "point to an interface");

  auto verilogName = getSymOpName(interfaceOp);
  if (!prefix.empty())
    ps << PPExtString(prefix);
  ps << PPExtString(verilogName)
     << PP::nbsp /* don't break, may be comment line */
     << PPExtString(op.getName()) << "();";

  emitLocationInfoAndNewLine(ops);

  return success();
}

/// For OutputOp we put "assign" statements at the end of the Verilog module to
/// assign the module outputs to intermediate wires.
LogicalResult StmtEmitter::visitStmt(OutputOp op) {
  SmallPtrSet<Operation *, 8> ops;
  auto parent = op->getParentOfType<PortList>();

  size_t operandIndex = 0;
  auto ports = parent.getPortList();
  for (PortInfo port : ports.getOutputs()) {
    auto operand = op.getOperand(operandIndex);
    // Outputs that are set by the output port of an instance are handled
    // directly when the instance is emitted.
    // Keep synced with countStatements() and visitStmt(InstanceOp).
    if (operand.hasOneUse() &&
        dyn_cast_or_null<InstanceOp>(operand.getDefiningOp())) {
      ++operandIndex;
      continue;
    }

    ops.clear();
    ops.insert(op);

    startStatement();
    bool isZeroBit = isZeroBitType(port.type);
    ps.scopedBox(isZeroBit ? PP::neverbox : PP::ibox2, [&]() {
      if (isZeroBit)
        ps << "// Zero width: ";

      ps << "assign" << PP::space;
      ps << PPExtString(getPortVerilogName(parent, port));
      ps << PP::space << "=" << PP::space;
      ps.scopedBox(PP::ibox0, [&]() {
        // If this is a zero-width constant then don't emit it (illegal). Else,
        // emit the expression - even for zero width - for traceability.
        if (isZeroBit &&
            isa_and_nonnull<hw::ConstantOp>(operand.getDefiningOp()))
          ps << "/*Zero width*/";
        else
          emitExpression(operand, ops, LowestPrecedence);
        ps << ";";
      });
    });
    emitLocationInfoAndNewLine(ops);

    ++operandIndex;
  }
  return success();
}

LogicalResult StmtEmitter::visitStmt(TypeScopeOp op) {
  startStatement();
  auto typescopeDef = ("_TYPESCOPE_" + op.getSymName()).str();
  ps << "`ifndef " << typescopeDef << PP::newline;
  ps << "`define " << typescopeDef;
  setPendingNewline();
  emitStatementBlock(*op.getBodyBlock());
  startStatement();
  ps << "`endif // " << typescopeDef;
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitStmt(TypedeclOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  auto zeroBitType = isZeroBitType(op.getType());
  if (zeroBitType)
    ps << PP::neverbox << "// ";

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps.scopedBox(PP::ibox2, [&]() {
    ps << "typedef" << PP::space;
    ps.invokeWithStringOS([&](auto &os) {
      emitter.printPackedType(stripUnpackedTypes(op.getType()), os, op.getLoc(),
                              op.getAliasType(), false);
    });
    ps << PP::space << PPExtString(op.getPreferredName());
    ps.invokeWithStringOS(
        [&](auto &os) { emitter.printUnpackedTypePostfix(op.getType(), os); });
    ps << ";";
  });
  if (zeroBitType)
    ps << PP::end;
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(FWriteOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  ps << "$fwrite(";
  ps.scopedBox(PP::ibox0, [&]() {
    emitExpression(op.getFd(), ops);

    ps << "," << PP::space;
    ps.writeQuotedEscaped(op.getFormatString());

    // TODO: if any of these breaks, it'd be "nice" to break
    // after the comma, instead of:
    // $fwrite(5, "...", a + b,
    //         longexpr_goes
    //         + here, c);
    // (without forcing breaking between all elements, like braced list)
    for (auto operand : op.getSubstitutions()) {
      ps << "," << PP::space;
      emitExpression(operand, ops);
    }
    ps << ");";
  });
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(VerbatimOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps << PP::neverbox;

  // Drop an extraneous \n off the end of the string if present.
  StringRef string = op.getFormatString();
  if (string.endswith("\n"))
    string = string.drop_back();

  // Emit each \n separated piece of the string with each piece properly
  // indented.  The convention is to not emit the \n so
  // emitLocationInfoAndNewLine can do that for the last line.
  bool isFirst = true;

  // Emit each line of the string at a time.
  while (!string.empty()) {
    auto lhsRhs = string.split('\n');
    if (isFirst)
      isFirst = false;
    else {
      ps << PP::end << PP::newline << PP::neverbox;
    }

    // Emit each chunk of the line.
    emitTextWithSubstitutions(
        ps, lhsRhs.first, op,
        [&](Value operand) { emitExpression(operand, ops); }, op.getSymbols());
    string = lhsRhs.second;
  }

  ps << PP::end;

  emitLocationInfoAndNewLine(ops);
  return success();
}

/// Emit one of the simulation control tasks `$stop`, `$finish`, or `$exit`.
LogicalResult
StmtEmitter::emitSimulationControlTask(Operation *op, PPExtString taskName,
                                       std::optional<unsigned> verbosity) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps << taskName;
  if (verbosity && *verbosity != 1) {
    ps << "(";
    ps.addAsString(*verbosity);
    ps << ")";
  }
  ps << ";";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(StopOp op) {
  return emitSimulationControlTask(op, PPExtString("$stop"), op.getVerbosity());
}

LogicalResult StmtEmitter::visitSV(FinishOp op) {
  return emitSimulationControlTask(op, PPExtString("$finish"),
                                   op.getVerbosity());
}

LogicalResult StmtEmitter::visitSV(ExitOp op) {
  return emitSimulationControlTask(op, PPExtString("$exit"), {});
}

/// Emit one of the severity message tasks `$fatal`, `$error`, `$warning`, or
/// `$info`.
LogicalResult
StmtEmitter::emitSeverityMessageTask(Operation *op, PPExtString taskName,
                                     std::optional<unsigned> verbosity,
                                     StringAttr message, ValueRange operands) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps << taskName;

  // In case we have a message to print, or the operation has an optional
  // verbosity and that verbosity is present, print the parenthesized parameter
  // list.
  if ((verbosity && *verbosity != 1) || message) {
    ps << "(";
    ps.scopedBox(PP::ibox0, [&]() {
      // If the operation takes a verbosity, print it if it is set, or print the
      // default "1".
      if (verbosity)
        ps.addAsString(*verbosity);

      // Print the message and interpolation operands if present.
      if (message) {
        if (verbosity)
          ps << "," << PP::space;
        ps.writeQuotedEscaped(message.getValue());
        // TODO: good comma/wrapping behavior as elsewhere.
        for (auto operand : operands) {
          ps << "," << PP::space;
          emitExpression(operand, ops);
        }
      }

      ps << ")";
    });
  }

  ps << ";";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(FatalOp op) {
  return emitSeverityMessageTask(op, PPExtString("$fatal"), op.getVerbosity(),
                                 op.getMessageAttr(), op.getSubstitutions());
}

LogicalResult StmtEmitter::visitSV(ErrorOp op) {
  return emitSeverityMessageTask(op, PPExtString("$error"), {},
                                 op.getMessageAttr(), op.getSubstitutions());
}

LogicalResult StmtEmitter::visitSV(WarningOp op) {
  return emitSeverityMessageTask(op, PPExtString("$warning"), {},
                                 op.getMessageAttr(), op.getSubstitutions());
}

LogicalResult StmtEmitter::visitSV(InfoOp op) {
  return emitSeverityMessageTask(op, PPExtString("$info"), {},
                                 op.getMessageAttr(), op.getSubstitutions());
}

LogicalResult StmtEmitter::visitSV(ReadMemOp op) {
  SmallPtrSet<Operation *, 8> ops({op});

  startStatement();
  ps << "$readmem";
  switch (op.getBaseAttr().getValue()) {
  case MemBaseTypeAttr::MemBaseBin:
    ps << "b";
    break;
  case MemBaseTypeAttr::MemBaseHex:
    ps << "h";
    break;
  }
  ps << "(";
  ps.scopedBox(PP::ibox0, [&]() {
    ps.writeQuotedEscaped(op.getFilename());
    ps << "," << PP::space;
    emitExpression(op.getDest(), ops);
  });

  ps << ");";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(GenerateOp op) {
  emitSVAttributes(op);
  // TODO: location info?
  startStatement();
  ps << "generate" << PP::newline;
  ps << "begin: " << PPExtString(getSymOpName(op));
  setPendingNewline();
  emitStatementBlock(op.getBody().getBlocks().front());
  startStatement();
  ps << "end: " << PPExtString(getSymOpName(op)) << PP::newline;
  ps << "endgenerate";
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitSV(GenerateCaseOp op) {
  emitSVAttributes(op);
  // TODO: location info?
  startStatement();
  ps << "case (";
  ps.invokeWithStringOS([&](auto &os) {
    emitter.printParamValue(
        op.getCond(), os, VerilogPrecedence::Selection,
        [&]() { return op->emitOpError("invalid case parameter"); });
  });
  ps << ")";
  setPendingNewline();

  // Ensure that all of the per-case arrays are the same length.
  ArrayAttr patterns = op.getCasePatterns();
  ArrayAttr caseNames = op.getCaseNames();
  MutableArrayRef<Region> regions = op.getCaseRegions();
  assert(patterns.size() == regions.size());
  assert(patterns.size() == caseNames.size());

  // TODO: We'll probably need to store the legalized names somewhere for
  // `verbose` formatting. Set up the infra for storing names recursively. Just
  // store this locally for now.
  llvm::StringMap<size_t> nextGenIds;
  ps.scopedBox(PP::bbox2, [&]() {
    // Emit each case.
    for (size_t i = 0, e = patterns.size(); i < e; ++i) {
      auto &region = regions[i];
      assert(region.hasOneBlock());
      Attribute patternAttr = patterns[i];

      startStatement();
      if (!patternAttr.isa<mlir::TypedAttr>())
        ps << "default";
      else
        ps.invokeWithStringOS([&](auto &os) {
          emitter.printParamValue(
              patternAttr, os, VerilogPrecedence::LowestPrecedence,
              [&]() { return op->emitOpError("invalid case value"); });
        });

      StringRef legalName =
          legalizeName(caseNames[i].cast<StringAttr>().getValue(), nextGenIds);
      ps << ": begin: " << PPExtString(legalName);
      setPendingNewline();
      emitStatementBlock(region.getBlocks().front());
      startStatement();
      ps << "end: " << PPExtString(legalName);
      setPendingNewline();
    }
  });

  startStatement();
  ps << "endcase";
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitSV(ForOp op) {
  emitSVAttributes(op);
  llvm::SmallPtrSet<Operation *, 8> ops;
  startStatement();
  auto inductionVarName = op->getAttrOfType<StringAttr>("hw.verilogName");
  ps << "for (";
  // Emit statements on same line if possible, or put each on own line.
  ps.scopedBox(PP::cbox0, [&]() {
    // Emit initialization assignment.
    emitAssignLike(
        [&]() {
          ps << "logic" << PP::nbsp;
          ps.invokeWithStringOS([&](auto &os) {
            emitter.emitTypeDims(op.getInductionVar().getType(), op.getLoc(),
                                 os);
          });
          ps << PP::nbsp << PPExtString(inductionVarName);
        },
        [&]() { emitExpression(op.getLowerBound(), ops); }, PPExtString("="));
    // Break between statements.
    ps << PP::space;

    // Emit bounds-check statement.
    emitAssignLike([&]() { ps << PPExtString(inductionVarName); },
                   [&]() { emitExpression(op.getUpperBound(), ops); },
                   PPExtString("<"));
    // Break between statements.
    ps << PP::space;

    // Emit update statement and trailing syntax.
    emitAssignLike([&]() { ps << PPExtString(inductionVarName); },
                   [&]() { emitExpression(op.getStep(), ops); },
                   PPExtString("+="), PPExtString(") begin"));
  });
  // Don't break for because of newline.
  ps << PP::neverbreak;
  setPendingNewline();
  emitStatementBlock(op.getBody().getBlocks().front());
  startStatement();
  ps << "end";
  emitLocationInfoAndNewLine(ops);
  return success();
}

/// Emit the `<label>:` portion of a verification operation.
void StmtEmitter::emitAssertionLabel(Operation *op) {
  if (auto label = op->getAttrOfType<StringAttr>("hw.verilogName"))
    ps << PPExtString(label) << ":" << PP::space;
}

/// Emit the optional ` else $error(...)` portion of an immediate or concurrent
/// verification operation.
void StmtEmitter::emitAssertionMessage(StringAttr message, ValueRange args,
                                       SmallPtrSetImpl<Operation *> &ops,
                                       bool isConcurrent = false) {
  if (!message)
    return;
  ps << PP::space << "else" << PP::nbsp << "$error(";
  ps.scopedBox(PP::ibox0, [&]() {
    ps.writeQuotedEscaped(message.getValue());
    // TODO: box, break/wrap behavior!
    for (auto arg : args) {
      ps << "," << PP::space;
      emitExpression(arg, ops);
    }
    ps << ")";
  });
}

template <typename Op>
LogicalResult StmtEmitter::emitImmediateAssertion(Op op, PPExtString opName) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps.scopedBox(PP::ibox2, [&]() {
    emitAssertionLabel(op);
    ps.scopedBox(PP::cbox0, [&]() {
      ps << opName;
      switch (op.getDefer()) {
      case DeferAssert::Immediate:
        break;
      case DeferAssert::Observed:
        ps << " #0 ";
        break;
      case DeferAssert::Final:
        ps << " final ";
        break;
      }
      ps << "(";
      ps.scopedBox(PP::ibox0, [&]() {
        emitExpression(op.getExpression(), ops);
        ps << ")";
      });
      emitAssertionMessage(op.getMessageAttr(), op.getSubstitutions(), ops);
      ps << ";";
    });
  });
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AssertOp op) {
  return emitImmediateAssertion(op, PPExtString("assert"));
}

LogicalResult StmtEmitter::visitSV(AssumeOp op) {
  return emitImmediateAssertion(op, PPExtString("assume"));
}

LogicalResult StmtEmitter::visitSV(CoverOp op) {
  return emitImmediateAssertion(op, PPExtString("cover"));
}

template <typename Op>
LogicalResult StmtEmitter::emitConcurrentAssertion(Op op, PPExtString opName) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps.scopedBox(PP::ibox2, [&]() {
    emitAssertionLabel(op);
    ps.scopedBox(PP::cbox0, [&]() {
      ps << opName << PP::nbsp << "property (";
      ps.scopedBox(PP::ibox0, [&]() {
        ps << "@(" << PPExtString(stringifyEventControl(op.getEvent()))
           << PP::nbsp;
        emitExpression(op.getClock(), ops);
        ps << ")" << PP::space;
        emitExpression(op.getProperty(), ops);
        ps << ")";
      });
      emitAssertionMessage(op.getMessageAttr(), op.getSubstitutions(), ops,
                           true);
      ps << ";";
    });
  });
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitSV(AssertConcurrentOp op) {
  return emitConcurrentAssertion(op, PPExtString("assert"));
}

LogicalResult StmtEmitter::visitSV(AssumeConcurrentOp op) {
  return emitConcurrentAssertion(op, PPExtString("assume"));
}

LogicalResult StmtEmitter::visitSV(CoverConcurrentOp op) {
  return emitConcurrentAssertion(op, PPExtString("cover"));
}

/// Emit an assert-like operation from the `verif` dialect. This covers
/// `verif.assert`, `verif.assume`, and `verif.cover`.
LogicalResult StmtEmitter::emitVerifAssertLike(Operation *op, Value property,
                                               PPExtString opName) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  // If we are inside a procedural region we have the option of emitting either
  // an `assert` or `assert property`. If we are in a non-procedural region,
  // e.g., the body of a module, we have to use the concurrent form `assert
  // property` (which also supports plain booleans).
  //
  // See IEEE 1800-2017 section 16.14.5 "Using concurrent assertion statements
  // outside procedural code" and 16.14.6 "Embedding concurrent assertions in
  // procedural code".
  bool isTemporal = !property.getType().isSignlessInteger(1);
  bool isProcedural = op->getParentOp()->hasTrait<ProceduralRegion>();
  bool emitAsImmediate = !isTemporal && isProcedural;

  startStatement();
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  ps.scopedBox(PP::ibox2, [&]() {
    emitAssertionLabel(op);
    ps.scopedBox(PP::cbox0, [&]() {
      if (emitAsImmediate)
        ps << opName << "(";
      else
        ps << opName << PP::nbsp << "property" << PP::nbsp << "(";
      ps.scopedBox(PP::ibox2, [&]() {
        PropertyEmitter(emitter, ops).emitProperty(property);
        ps << ");";
      });
    });
  });
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitVerif(verif::AssertOp op) {
  return emitVerifAssertLike(op, op.getProperty(), PPExtString("assert"));
}

LogicalResult StmtEmitter::visitVerif(verif::AssumeOp op) {
  return emitVerifAssertLike(op, op.getProperty(), PPExtString("assume"));
}

LogicalResult StmtEmitter::visitVerif(verif::CoverOp op) {
  return emitVerifAssertLike(op, op.getProperty(), PPExtString("cover"));
}

LogicalResult StmtEmitter::emitIfDef(Operation *op, MacroIdentAttr cond) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto ident = PPExtString(cond.getName());

  startStatement();
  bool hasEmptyThen = op->getRegion(0).front().empty();
  if (hasEmptyThen)
    ps << "`ifndef " << ident;
  else
    ps << "`ifdef " << ident;

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  emitLocationInfoAndNewLine(ops);

  if (!hasEmptyThen)
    emitStatementBlock(op->getRegion(0).front());

  if (!op->getRegion(1).empty()) {
    if (!hasEmptyThen) {
      startStatement();
      ps << "`else  // " << ident;
      setPendingNewline();
    }
    emitStatementBlock(op->getRegion(1).front());
  }
  startStatement();
  ps << "`endif // ";
  if (hasEmptyThen)
    ps << "not def ";
  ps << ident;
  setPendingNewline();
  return success();
}

/// Emit the body of a control flow statement that is surrounded by begin/end
/// markers if non-singular.  If the control flow construct is multi-line and
/// if multiLineComment is non-null, the string is included in a comment after
/// the 'end' to make it easier to associate.
void StmtEmitter::emitBlockAsStatement(
    Block *block, const SmallPtrSetImpl<Operation *> &locationOps,
    StringRef multiLineComment) {

  // Determine if we need begin/end by scanning the block.
  auto count = countStatements(*block);
  auto needsBeginEnd = count != BlockStatementCount::One;
  if (needsBeginEnd)
    ps << " begin";
  emitLocationInfoAndNewLine(locationOps);

  if (count != BlockStatementCount::Zero)
    emitStatementBlock(*block);

  if (needsBeginEnd) {
    startStatement();
    ps << "end";
    // Emit comment if there's an 'end', regardless of line count.
    if (!multiLineComment.empty())
      ps << " // " << multiLineComment;
    setPendingNewline();
  }
}

LogicalResult StmtEmitter::visitSV(OrderedOutputOp ooop) {
  // Emit the body.
  for (auto &op : ooop.getBody().front())
    emitStatement(&op);
  return success();
}

LogicalResult StmtEmitter::visitSV(IfOp op) {
  SmallPtrSet<Operation *, 8> ops;

  auto ifcondBox = PP::ibox2;

  emitSVAttributes(op);
  startStatement();
  ps << "if (" << ifcondBox;

  // In the loop, emit an if statement assuming the keyword introducing
  // it (either "if (" or "else if (") was printed already.
  IfOp ifOp = op;
  for (;;) {
    ops.clear();
    ops.insert(ifOp);

    // Emit the condition and the then block.
    emitExpression(ifOp.getCond(), ops);
    ps << PP::end << ")";
    emitBlockAsStatement(ifOp.getThenBlock(), ops);

    if (!ifOp.hasElse())
      break;

    startStatement();
    Block *elseBlock = ifOp.getElseBlock();
    auto nestedElseIfOp = findNestedElseIf(elseBlock);
    if (!nestedElseIfOp) {
      // The else block does not contain an if-else that can be flattened.
      ops.clear();
      ops.insert(ifOp);
      ps << "else";
      emitBlockAsStatement(elseBlock, ops);
      break;
    }

    // Introduce the 'else if', and iteratively continue unfolding any if-else
    // statements inside of it.
    ifOp = nestedElseIfOp;
    ps << "else if (" << ifcondBox;
  }

  return success();
}

LogicalResult StmtEmitter::visitSV(AlwaysOp op) {
  emitSVAttributes(op);
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  startStatement();

  auto printEvent = [&](AlwaysOp::Condition cond) {
    ps << PPExtString(stringifyEventControl(cond.event)) << PP::nbsp;
    ps.scopedBox(PP::cbox0, [&]() { emitExpression(cond.value, ops); });
  };

  switch (op.getNumConditions()) {
  case 0:
    ps << "always @*";
    break;
  case 1:
    ps << "always @(";
    printEvent(op.getCondition(0));
    ps << ")";
    break;
  default:
    ps << "always @(";
    ps.scopedBox(PP::cbox0, [&]() {
      printEvent(op.getCondition(0));
      for (size_t i = 1, e = op.getNumConditions(); i != e; ++i) {
        ps << PP::space << "or" << PP::space;
        printEvent(op.getCondition(i));
      }
      ps << ")";
    });
    break;
  }

  // Build the comment string, leave out the signal expressions (since they
  // can be large).
  std::string comment;
  if (op.getNumConditions() == 0) {
    comment = "always @*";
  } else {
    comment = "always @(";
    llvm::interleave(
        op.getEvents(),
        [&](Attribute eventAttr) {
          auto event = sv::EventControl(eventAttr.cast<IntegerAttr>().getInt());
          comment += stringifyEventControl(event);
        },
        [&]() { comment += ", "; });
    comment += ')';
  }

  emitBlockAsStatement(op.getBodyBlock(), ops, comment);
  return success();
}

LogicalResult StmtEmitter::visitSV(AlwaysCombOp op) {
  emitSVAttributes(op);
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  startStatement();

  StringRef opString = "always_comb";
  if (state.options.noAlwaysComb)
    opString = "always @(*)";

  ps << PPExtString(opString);
  emitBlockAsStatement(op.getBodyBlock(), ops, opString);
  return success();
}

LogicalResult StmtEmitter::visitSV(AlwaysFFOp op) {
  emitSVAttributes(op);

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  startStatement();

  ps << "always_ff @(";
  ps.scopedBox(PP::cbox0, [&]() {
    ps << PPExtString(stringifyEventControl(op.getClockEdge())) << PP::nbsp;
    emitExpression(op.getClock(), ops);
    if (op.getResetStyle() == ResetType::AsyncReset) {
      ps << PP::nbsp << "or" << PP::space
         << PPExtString(stringifyEventControl(*op.getResetEdge())) << PP::nbsp;
      emitExpression(op.getReset(), ops);
    }
    ps << ")";
  });

  // Build the comment string, leave out the signal expressions (since they
  // can be large).
  std::string comment;
  comment += "always_ff @(";
  comment += stringifyEventControl(op.getClockEdge());
  if (op.getResetStyle() == ResetType::AsyncReset) {
    comment += " or ";
    comment += stringifyEventControl(*op.getResetEdge());
  }
  comment += ')';

  if (op.getResetStyle() == ResetType::NoReset)
    emitBlockAsStatement(op.getBodyBlock(), ops, comment);
  else {
    ps << " begin";
    emitLocationInfoAndNewLine(ops);
    ps.scopedBox(PP::bbox2, [&]() {
      startStatement();
      ps << "if (";
      // TODO: group, like normal 'if'.
      // Negative edge async resets need to invert the reset condition. This
      // is noted in the op description.
      if (op.getResetStyle() == ResetType::AsyncReset &&
          *op.getResetEdge() == sv::EventControl::AtNegEdge)
        ps << "!";
      emitExpression(op.getReset(), ops);
      ps << ")";
      emitBlockAsStatement(op.getResetBlock(), ops);
      startStatement();
      ps << "else";
      emitBlockAsStatement(op.getBodyBlock(), ops);
    });

    startStatement();
    ps << "end";
    ps << " // " << comment;
    setPendingNewline();
  }
  return success();
}

LogicalResult StmtEmitter::visitSV(InitialOp op) {
  emitSVAttributes(op);
  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);
  startStatement();
  ps << "initial";
  emitBlockAsStatement(op.getBodyBlock(), ops, "initial");
  return success();
}

LogicalResult StmtEmitter::visitSV(CaseOp op) {
  emitSVAttributes(op);
  SmallPtrSet<Operation *, 8> ops, emptyOps;
  ops.insert(op);
  startStatement();
  if (op.getValidationQualifier() !=
      ValidationQualifierTypeEnum::ValidationQualifierPlain)
    ps << PPExtString(circt::sv::stringifyValidationQualifierTypeEnum(
              op.getValidationQualifier()))
       << PP::nbsp;
  const char *opname = nullptr;
  switch (op.getCaseStyle()) {
  case CaseStmtType::CaseStmt:
    opname = "case";
    break;
  case CaseStmtType::CaseXStmt:
    opname = "casex";
    break;
  case CaseStmtType::CaseZStmt:
    opname = "casez";
    break;
  }
  ps << opname << " (";
  ps.scopedBox(PP::ibox0, [&]() {
    emitExpression(op.getCond(), ops);
    ps << ")";
  });
  emitLocationInfoAndNewLine(ops);

  ps.scopedBox(PP::bbox2, [&]() {
    for (auto &caseInfo : op.getCases()) {
      startStatement();
      auto &pattern = caseInfo.pattern;

      llvm::TypeSwitch<CasePattern *>(pattern.get())
          .Case<CaseBitPattern>([&](auto bitPattern) {
            // TODO: We could emit in hex if/when the size is a multiple of
            // 4 and there are no x's crossing nibble boundaries.
            ps.invokeWithStringOS([&](auto &os) {
              os << bitPattern->getWidth() << "'b";
              for (size_t bit = 0, e = bitPattern->getWidth(); bit != e; ++bit)
                os << getLetter(bitPattern->getBit(e - bit - 1));
            });
          })
          .Case<CaseEnumPattern>([&](auto enumPattern) {
            ps << PPExtString(emitter.fieldNameResolver.getEnumFieldName(
                enumPattern->attr().template cast<hw::EnumFieldAttr>()));
          })
          .Case<CaseDefaultPattern>([&](auto) { ps << "default"; })
          .Default([&](auto) { assert(false && "unhandled case pattern"); });

      ps << ":";
      emitBlockAsStatement(caseInfo.block, emptyOps);
    }
  });

  startStatement();
  ps << "endcase";
  emitLocationInfoAndNewLine(ops);
  return success();
}

LogicalResult StmtEmitter::visitStmt(InstanceOp op) {
  bool doNotPrint = op->hasAttr("doNotPrint");
  if (doNotPrint && !state.options.emitBindComments)
    return success();

  // Emit SV attributes if the op is not emitted as a bind statement.
  if (!doNotPrint)
    emitSVAttributes(op);
  startStatement();
  if (doNotPrint) {
    ps << PP::ibox2
       << "/* This instance is elsewhere emitted as a bind statement."
       << PP::newline;
    if (hasSVAttributes(op))
      op->emitWarning() << "is emitted as a bind statement but has SV "
                           "attributes. The attributes will not be emitted.";
  }

  SmallPtrSet<Operation *, 8> ops;
  ops.insert(op);

  // Use the specified name or the symbol name as appropriate.
  auto *moduleOp = op.getReferencedModule(&state.symbolCache);
  assert(moduleOp && "Invalid IR");
  ps << PPExtString(getVerilogModuleName(moduleOp));

  // If this is a parameterized module, then emit the parameters.
  if (!op.getParameters().empty()) {
    // All the parameters may be defaulted -- don't print out an empty list if
    // so.
    bool printed = false;
    for (auto params :
         llvm::zip(op.getParameters(),
                   moduleOp->getAttrOfType<ArrayAttr>("parameters"))) {
      auto param = std::get<0>(params).cast<ParamDeclAttr>();
      auto modParam = std::get<1>(params).cast<ParamDeclAttr>();
      // Ignore values that line up with their default.
      if (param.getValue() == modParam.getValue())
        continue;

      // Handle # if this is the first parameter we're printing.
      if (!printed) {
        ps << " #(" << PP::bbox2 << PP::newline;
        printed = true;
      } else {
        ps << "," << PP::newline;
      }
      ps << ".";
      ps << PPExtString(
          state.globalNames.getParameterVerilogName(moduleOp, param.getName()));
      ps << "(";
      ps.invokeWithStringOS([&](auto &os) {
        emitter.printParamValue(param.getValue(), os, [&]() {
          return op->emitOpError("invalid instance parameter '")
                 << param.getName().getValue() << "' value";
        });
      });
      ps << ")";
    }
    if (printed) {
      ps << PP::end << PP::newline << ")";
    }
  }

  ps << PP::nbsp << PPExtString(getSymOpName(op)) << " (";

  auto instPortInfo = op.getPortList();
  auto modPortInfo = cast<PortList>(op.getReferencedModule()).getPortList();
  // Get the max port name length so we can align the '('.
  size_t maxNameLength = 0;
  for (auto &elt : modPortInfo) {
    maxNameLength =
        std::max(maxNameLength, getPortVerilogName(moduleOp, elt).size());
  }

  auto getWireForValue = [&](Value result) {
    return result.getUsers().begin()->getOperand(0);
  };

  // Emit the argument and result ports.
  bool isFirst = true; // True until we print a port.
  bool isZeroWidth = false;

  for (size_t portNum = 0, portEnd = instPortInfo.size(); portNum < portEnd;
       ++portNum) {
    auto &modPort = modPortInfo.at(portNum);
    isZeroWidth = isZeroBitType(modPort.type);
    Value portVal = op.getValue(portNum);

    // Decide if we should print a comma.  We can't do this if we're the first
    // port or if all the subsequent ports are zero width.
    if (!isFirst) {
      bool shouldPrintComma = true;
      if (isZeroWidth) {
        shouldPrintComma = false;
        for (size_t i = portNum + 1, e = instPortInfo.size(); i != e; ++i)
          if (!isZeroBitType(modPortInfo.at(i).type)) {
            shouldPrintComma = true;
            break;
          }
      }

      if (shouldPrintComma)
        ps << ",";
    }
    emitLocationInfoAndNewLine(ops);

    // Emit the port's name.
    startStatement();
    if (!isZeroWidth) {
      // If this is a real port we're printing, then it isn't the first one. Any
      // subsequent ones will need a comma.
      isFirst = false;
      ps << "  ";
    } else {
      // We comment out zero width ports, so their presence and initializer
      // expressions are still emitted textually.
      ps << "//";
    }

    ps.scopedBox(isZeroWidth ? PP::neverbox : PP::ibox2, [&]() {
      auto modPortName = getPortVerilogName(moduleOp, modPort);
      ps << "." << PPExtString(modPortName);
      ps.spaces(maxNameLength - modPortName.size() + 1);
      ps << "(";
      ps.scopedBox(PP::ibox0, [&]() {
        // Emit the value as an expression.
        ops.clear();

        // Output ports that are not connected to single use output ports were
        // lowered to wire.
        OutputOp output;
        if (!modPort.isOutput()) {
          if (isZeroWidth &&
              isa_and_nonnull<ConstantOp>(portVal.getDefiningOp()))
            ps << "/* Zero width */";
          else
            emitExpression(portVal, ops, LowestPrecedence);
        } else if (portVal.use_empty()) {
          ps << "/* unused */";
        } else if (portVal.hasOneUse() &&
                   (output = dyn_cast_or_null<OutputOp>(
                        portVal.getUses().begin()->getOwner()))) {
          // If this is directly using the output port of the containing module,
          // just specify that directly so we avoid a temporary wire.
          // Keep this synchronized with countStatements() and
          // visitStmt(OutputOp).
          size_t outputPortNo = portVal.getUses().begin()->getOperandNumber();
          auto containingModule = cast<HWModuleOp>(emitter.currentModuleOp);
          ps << PPExtString(getPortVerilogName(
              containingModule, containingModule.getOutputPort(outputPortNo)));
        } else {
          portVal = getWireForValue(portVal);
          emitExpression(portVal, ops);
        }
        ps << ")";
      });
    });
  }
  if (!isFirst || isZeroWidth) {
    emitLocationInfoAndNewLine(ops);
    ops.clear();
    startStatement();
  }
  ps << ");";
  emitLocationInfoAndNewLine(ops);
  if (doNotPrint) {
    ps << PP::end;
    startStatement();
    ps << "*/";
    setPendingNewline();
  }
  return success();
}

// This may be called in the top-level, not just in an hw.module.  Thus we can't
// use the name map to find expression names for arguments to the instance, nor
// do we need to emit subexpressions.  Prepare pass, which has run for all
// modules prior to this, has ensured that all arguments are bound to wires,
// regs, or ports, with legalized names, so we can lookup up the names through
// the IR.
LogicalResult StmtEmitter::visitSV(BindOp op) {
  emitter.emitBind(op);
  assert(state.pendingNewline);
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceOp op) {
  emitComment(op.getCommentAttr());
  // Emit SV attributes.
  emitSVAttributes(op);
  // TODO: source info!
  startStatement();
  ps << "interface " << PPExtString(getSymOpName(op)) << ";";
  setPendingNewline();
  // FIXME: Don't emit the body of this as general statements, they aren't!
  emitStatementBlock(*op.getBodyBlock());
  startStatement();
  ps << "endinterface" << PP::newline;
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceSignalOp op) {
  // Emit SV attributes.
  emitSVAttributes(op);
  startStatement();
  if (isZeroBitType(op.getType()))
    ps << PP::neverbox << "// ";
  ps.invokeWithStringOS([&](auto &os) {
    emitter.printPackedType(stripUnpackedTypes(op.getType()), os, op->getLoc(),
                            Type(), false);
  });
  ps << PP::nbsp << PPExtString(getSymOpName(op));
  ps.invokeWithStringOS(
      [&](auto &os) { emitter.printUnpackedTypePostfix(op.getType(), os); });
  ps << ";";
  if (isZeroBitType(op.getType()))
    ps << PP::end; // Close never-break group.
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitSV(InterfaceModportOp op) {
  startStatement();
  ps << "modport " << PPExtString(getSymOpName(op)) << "(";

  // TODO: revisit, better breaks/grouping.
  llvm::interleaveComma(op.getPorts(), ps, [&](const Attribute &portAttr) {
    auto port = portAttr.cast<ModportStructAttr>();
    ps << PPExtString(stringifyEnum(port.getDirection().getValue())) << " ";
    auto *signalDecl = state.symbolCache.getDefinition(port.getSignal());
    ps << PPExtString(getSymOpName(signalDecl));
  });

  ps << ");";
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitSV(AssignInterfaceSignalOp op) {
  startStatement();
  SmallPtrSet<Operation *, 8> emitted;
  // TODO: emit like emitAssignLike does, maybe refactor.
  ps << "assign ";
  emitExpression(op.getIface(), emitted);
  ps << "." << PPExtString(op.getSignalName()) << " = ";
  emitExpression(op.getRhs(), emitted);
  ps << ";";
  setPendingNewline();
  return success();
}

LogicalResult StmtEmitter::visitSV(MacroDefOp op) {
  auto decl = op.getReferencedMacro(&state.symbolCache);
  // TODO: source info!
  startStatement();
  ps << "`define " << PPExtString(getSymOpName(decl));
  if (decl.getArgs()) {
    ps << "(";
    llvm::interleaveComma(*decl.getArgs(), ps, [&](const Attribute &name) {
      ps << name.cast<StringAttr>();
    });
    ps << ")";
  }
  if (!op.getFormatString().empty()) {
    ps << " ";
    emitTextWithSubstitutions(ps, op.getFormatString(), op, {},
                              op.getSymbols());
  }
  setPendingNewline();
  return success();
}

void StmtEmitter::emitStatement(Operation *op) {
  // Expressions may either be ignored or emitted as an expression statements.
  if (isVerilogExpression(op))
    return;

  // Ignore LTL expressions as they are emitted as part of verification
  // statements.
  if (isa<ltl::LTLDialect>(op->getDialect()))
    return;

  // Handle HW statements, SV statements.
  if (succeeded(dispatchStmtVisitor(op)) || succeeded(dispatchSVVisitor(op)) ||
      succeeded(dispatchVerifVisitor(op)))
    return;

  emitOpError(op, "emission to Verilog not supported");
  emitPendingNewlineIfNeeded();
  ps << "unknown MLIR operation " << PPExtString(op->getName().getStringRef());
  setPendingNewline();
}

/// Given an operation corresponding to a VerilogExpression, determine whether
/// it is safe to emit inline into a 'localparam' or 'automatic logic' varaible
/// initializer in a procedural region.
///
/// We can't emit exprs inline when they refer to something else that can't be
/// emitted inline, when they're in a general #ifdef region,
static bool
isExpressionEmittedInlineIntoProceduralDeclaration(Operation *op,
                                                   StmtEmitter &stmtEmitter) {
  if (!isVerilogExpression(op))
    return false;

  // If the expression exists in an #ifdef region, then bail.  Emitting it
  // inline would cause it to be executed unconditionally, because the
  // declarations are outside the #ifdef.
  if (isa<IfDefProceduralOp>(op->getParentOp()))
    return false;

  // This expression tree can be emitted into the initializer if all leaf
  // references are safe to refer to from here.  They are only safe if they are
  // defined in an enclosing scope (guaranteed to already be live by now) or if
  // they are defined in this block and already emitted to an inline automatic
  // logic variable.
  SmallVector<Value, 8> exprsToScan(op->getOperands());

  // This loop is guaranteed to terminate because we're only scanning up
  // single-use expressions and other things that 'isExpressionEmittedInline'
  // returns success for.  Cycles won't get in here.
  while (!exprsToScan.empty()) {
    Operation *expr = exprsToScan.pop_back_val().getDefiningOp();
    if (!expr)
      continue; // Ports are always safe to reference.

    // If this is an inout op, check that its inout op has no blocking
    // assignment. A register or logic might be mutated by a blocking assignment
    // so it is not always safe to inline.
    if (auto readInout = dyn_cast<sv::ReadInOutOp>(expr)) {
      auto *defOp = readInout.getOperand().getDefiningOp();

      // If it is a read from an inout port, it's unsafe to inline in general.
      if (!defOp)
        return false;

      // If the operand is a wire, it's OK to inline the read.
      if (isa<sv::WireOp>(defOp))
        continue;

      // Reject struct_field_inout/array_index_inout for now because it's
      // necessary to consider aliasing inout operations.
      if (!isa<RegOp, LogicOp>(defOp))
        return false;

      // It's safe to inline if all users are read op, passign or assign.
      // If the op is a logic op whose single assignment is inlined into
      // declaration, we can inline the read.
      if (isa<LogicOp>(defOp) &&
          stmtEmitter.emitter.expressionsEmittedIntoDecl.count(defOp))
        continue;

      // Check that it's safe for all users to be inlined.
      if (llvm::all_of(defOp->getResult(0).getUsers(), [&](Operation *op) {
            return isa<ReadInOutOp, PAssignOp, AssignOp>(op);
          }))
        continue;
      return false;
    }

    // If this is an internal node in the expression tree, process its operands.
    if (isExpressionEmittedInline(expr, stmtEmitter.state.options)) {
      exprsToScan.append(expr->getOperands().begin(),
                         expr->getOperands().end());
      continue;
    }

    // Otherwise, this isn't an inlinable expression.  If it is defined outside
    // this block, then it is live-in.
    if (expr->getBlock() != op->getBlock())
      continue;

    // Otherwise, if it is defined in this block then it is only ok to reference
    // if it has already been emitted into an automatic logic.
    if (!stmtEmitter.emitter.expressionsEmittedIntoDecl.count(expr))
      return false;
  }

  return true;
}

template <class AssignTy>
static AssignTy getSingleAssignAndCheckUsers(Operation *op) {
  AssignTy singleAssign;
  if (llvm::all_of(op->getUsers(), [&](Operation *user) {
        if (hasSVAttributes(user))
          return false;

        if (auto assign = dyn_cast<AssignTy>(user)) {
          if (singleAssign)
            return false;
          singleAssign = assign;
          return true;
        }

        return isa<ReadInOutOp>(user);
      }))
    return singleAssign;
  return {};
}

/// Return true if `op1` dominates users of `op2`.
static bool checkDominanceOfUsers(Operation *op1, Operation *op2) {
  return llvm::all_of(op2->getUsers(), [&](Operation *user) {
    /// TODO: Use MLIR DominanceInfo.

    // If the op1 and op2 are in different blocks, conservatively return false.
    if (op1->getBlock() != user->getBlock())
      return false;

    if (op1 == user)
      return true;

    return op1->isBeforeInBlock(user);
  });
}

LogicalResult StmtEmitter::emitDeclaration(Operation *op) {
  emitSVAttributes(op);
  auto value = op->getResult(0);
  SmallPtrSet<Operation *, 8> opsForLocation;
  opsForLocation.insert(op);
  startStatement();

  // Emit the leading word, like 'wire', 'reg' or 'logic'.
  auto type = value.getType();
  auto word = getVerilogDeclWord(op, state.options);
  auto isZeroBit = isZeroBitType(type);
  ps.scopedBox(isZeroBit ? PP::neverbox : PP::ibox2, [&]() {
    if (!isZeroBit) {
      if (!word.empty())
        ps << PPExtString(word);
      auto extraIndent = word.empty() ? 0 : 1;
      ps.spaces(maxDeclNameWidth - word.size() + extraIndent);
    } else {
      ps << "// Zero width: " << PPExtString(word) << PP::space;
    }

    SmallString<8> typeString;
    // Convert the port's type to a string and measure it.
    {
      llvm::raw_svector_ostream stringStream(typeString);
      emitter.printPackedType(stripUnpackedTypes(type), stringStream,
                              op->getLoc());
    }
    // Emit the type.
    if (!typeString.empty())
      ps << typeString;
    if (typeString.size() < maxTypeWidth)
      ps.spaces(maxTypeWidth - typeString.size());

    // Emit the name.
    ps << PPExtString(getSymOpName(op));

    // Print out any array subscripts or other post-name stuff.
    ps.invokeWithStringOS(
        [&](auto &os) { emitter.printUnpackedTypePostfix(type, os); });

    // Print debug info.
    if (state.options.printDebugInfo) {
      if (auto innerSymOp = dyn_cast<hw::InnerSymbolOpInterface>(op)) {
        auto innerSym = innerSymOp.getInnerSymAttr();
        if (innerSym && !innerSym.empty()) {
          ps << " /* ";
          ps.invokeWithStringOS([&](auto &os) { os << innerSym; });
          ps << " */";
        }
      }
    }

    if (auto localparam = dyn_cast<LocalParamOp>(op)) {
      ps << PP::space << "=" << PP::space;
      ps.invokeWithStringOS([&](auto &os) {
        emitter.printParamValue(localparam.getValue(), os, [&]() {
          return op->emitOpError("invalid localparam value");
        });
      });
    }

    // Try inlining an assignment into declarations.
    if (isa<sv::WireOp, LogicOp>(op) &&
        !op->getParentOp()->hasTrait<ProceduralRegion>()) {
      // Get a single assignments if any.
      if (auto singleAssign = getSingleAssignAndCheckUsers<AssignOp>(op)) {
        auto *source = singleAssign.getSrc().getDefiningOp();
        // Check that the source value is OK to inline in the current emission
        // point. A port or constant is fine, otherwise check that the assign is
        // next to the operation.
        if (!source || isa<ConstantOp>(source) ||
            op->getNextNode() == singleAssign) {
          ps << PP::space << "=" << PP::space;
          ps.scopedBox(PP::ibox0, [&]() {
            emitExpression(singleAssign.getSrc(), opsForLocation);
          });
          emitter.assignsInlined.insert(singleAssign);
        }
      }
    }

    // Try inlining a blocking assignment to logic op declaration.
    if (isa<LogicOp>(op) && op->getParentOp()->hasTrait<ProceduralRegion>()) {
      // Get a single assignment which might be possible to inline.
      if (auto singleAssign = getSingleAssignAndCheckUsers<BPAssignOp>(op)) {
        // It is necessary for the assignment to dominate users of the op.
        if (checkDominanceOfUsers(singleAssign, op)) {
          auto *source = singleAssign.getSrc().getDefiningOp();
          // A port or constant can be inlined at everywhere. Otherwise, check
          // the validity by
          // `isExpressionEmittedInlineIntoProceduralDeclaration`.
          if (!source || isa<ConstantOp>(source) ||
              isExpressionEmittedInlineIntoProceduralDeclaration(source,
                                                                 *this)) {
            ps << PP::space << "=" << PP::space;
            ps.scopedBox(PP::ibox0, [&]() {
              emitExpression(singleAssign.getSrc(), opsForLocation);
            });
            // Remember that the assignment and logic op are emitted into decl.
            emitter.assignsInlined.insert(singleAssign);
            emitter.expressionsEmittedIntoDecl.insert(op);
          }
        }
      }
    }
    ps << ";";
  });
  emitLocationInfoAndNewLine(opsForLocation);
  return success();
}

void StmtEmitter::collectNamesAndCalculateDeclarationWidths(Block &block) {
  // In the first pass, we fill in the symbol table, calculate the max width
  // of the declaration words and the max type width.
  NameCollector collector(emitter);
  collector.collectNames(block);

  // Record maxDeclNameWidth and maxTypeWidth in the current scope.
  maxDeclNameWidth = collector.getMaxDeclNameWidth();
  maxTypeWidth = collector.getMaxTypeWidth();

  if (maxTypeWidth > 0) // add a space if any type exists
    maxTypeWidth += 1;
}

void StmtEmitter::emitStatementBlock(Block &body) {
  ps.scopedBox(PP::bbox2, [&]() {
    // Ensure decl alignment values are preserved after the block is emitted.
    // These values were computed for and from all declarations in the current
    // block (before/after this nested block), so be sure they're restored
    // and not overwritten by the declaration alignment within the block.
    llvm::SaveAndRestore<size_t> x(maxDeclNameWidth);
    llvm::SaveAndRestore<size_t> x2(maxTypeWidth);

    // Build up the symbol table for all of the values that need names in the
    // module.  #ifdef's in procedural regions are special because local
    // variables are all emitted at the top of their enclosing blocks.
    if (!isa<IfDefProceduralOp>(body.getParentOp()))
      collectNamesAndCalculateDeclarationWidths(body);

    // Emit the body.
    for (auto &op : body) {
      emitStatement(&op);
    }
  });
}
// NOLINTEND(misc-no-recursion)

void ModuleEmitter::emitStatement(Operation *op) {
  StmtEmitter(*this).emitStatement(op);
}

/// Emit SystemVerilog attributes attached to the expression op as dialect
/// attributes.
void ModuleEmitter::emitSVAttributes(Operation *op) {
  // SystemVerilog 2017 Section 5.12.
  auto svAttrs = getSVAttributes(op);
  if (!svAttrs)
    return;

  startStatement(); // For attributes.
  emitSVAttributesImpl(ps, svAttrs, /*mayBreak=*/true);
  setPendingNewline();
}

//===----------------------------------------------------------------------===//
// Module Driver
//===----------------------------------------------------------------------===//

void ModuleEmitter::emitHWExternModule(HWModuleExternOp module) {
  auto verilogName = module.getVerilogModuleNameAttr();
  startStatement();
  ps << "// external module " << PPExtString(verilogName.getValue())
     << PP::newline;
  setPendingNewline();
}

void ModuleEmitter::emitHWGeneratedModule(HWModuleGeneratedOp module) {
  auto verilogName = module.getVerilogModuleNameAttr();
  startStatement();
  ps << "// external generated module " << PPExtString(verilogName.getValue())
     << PP::newline;
  setPendingNewline();
}

// This may be called in the top-level, not just in an hw.module.  Thus we can't
// use the name map to find expression names for arguments to the instance, nor
// do we need to emit subexpressions.  Prepare pass, which has run for all
// modules prior to this, has ensured that all arguments are bound to wires,
// regs, or ports, with legalized names, so we can lookup up the names through
// the IR.
void ModuleEmitter::emitBind(BindOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");
  InstanceOp inst = op.getReferencedInstance(&state.symbolCache);

  HWModuleOp parentMod = inst->getParentOfType<hw::HWModuleOp>();
  auto parentVerilogName = getVerilogModuleNameAttr(parentMod);

  Operation *childMod = inst.getReferencedModule(&state.symbolCache);
  auto childVerilogName = getVerilogModuleNameAttr(childMod);

  startStatement();
  ps << "bind " << PPExtString(parentVerilogName.getValue()) << PP::nbsp
     << PPExtString(childVerilogName.getValue()) << PP::nbsp
     << PPExtString(getSymOpName(inst)) << " (";
  bool isFirst = true; // True until we print a port.
  ps.scopedBox(PP::bbox2, [&]() {
    auto parentPortInfo = parentMod.getPortList();
    auto childPortInfo = cast<PortList>(childMod).getPortList();

    // Get the max port name length so we can align the '('.
    size_t maxNameLength = 0;
    for (auto &elt : childPortInfo) {
      auto portName = getPortVerilogName(childMod, elt);
      elt.name = Builder(inst.getContext()).getStringAttr(portName);
      maxNameLength = std::max(maxNameLength, elt.getName().size());
    }

    // Emit the argument and result ports.
    for (auto [idx, elt] : llvm::enumerate(childPortInfo)) {
      // Figure out which value we are emitting.
      Value portVal = inst.getValue(idx);
      bool isZeroWidth = isZeroBitType(elt.type);

      // Decide if we should print a comma.  We can't do this if we're the
      // first port or if all the subsequent ports are zero width.
      if (!isFirst) {
        bool shouldPrintComma = true;
        if (isZeroWidth) {
          shouldPrintComma = false;
          for (size_t i = idx + 1, e = childPortInfo.size(); i != e; ++i)
            if (!isZeroBitType(childPortInfo.at(i).type)) {
              shouldPrintComma = true;
              break;
            }
        }

        if (shouldPrintComma)
          ps << ",";
      }
      ps << PP::newline;

      // Emit the port's name.
      if (!isZeroWidth) {
        // If this is a real port we're printing, then it isn't the first
        // one. Any subsequent ones will need a comma.
        isFirst = false;
      } else {
        // We comment out zero width ports, so their presence and
        // initializer expressions are still emitted textually.
        ps << PP::neverbox << "//";
      }

      ps << "." << PPExtString(elt.getName());
      ps.nbsp(maxNameLength - elt.getName().size());
      ps << " (";
      llvm::SmallPtrSet<Operation *, 4> ops;
      if (elt.isOutput()) {
        assert((portVal.hasOneUse() || portVal.use_empty()) &&
               "output port must have either single or no use");
        if (portVal.use_empty()) {
          ps << "/* unused */";
        } else if (auto output = dyn_cast_or_null<OutputOp>(
                       portVal.getUses().begin()->getOwner())) {
          // If this is directly using the output port of the containing
          // module, just specify that directly.
          size_t outputPortNo = portVal.getUses().begin()->getOperandNumber();
          ps << PPExtString(getPortVerilogName(
              parentMod, parentMod.getOutputPort(outputPortNo)));
        } else {
          portVal = portVal.getUsers().begin()->getOperand(0);
          ExprEmitter(*this, ops).emitExpression(portVal, LowestPrecedence);
        }
      } else {
        ExprEmitter(*this, ops).emitExpression(portVal, LowestPrecedence);
      }

      ps << ")";

      if (isZeroWidth)
        ps << PP::end; // Close never-break group.
    }
  });
  if (!isFirst)
    ps << PP::newline;
  ps << ");";
  setPendingNewline();
}

void ModuleEmitter::emitBindInterface(BindInterfaceOp op) {
  if (hasSVAttributes(op))
    emitError(op, "SV attributes emission is unimplemented for the op");

  auto instance = op.getReferencedInstance(&state.symbolCache);
  auto instantiator = instance->getParentOfType<HWModuleOp>().getName();
  auto *interface = op->getParentOfType<ModuleOp>().lookupSymbol(
      instance.getInterfaceType().getInterface());
  startStatement();
  ps << "bind " << PPExtString(instantiator) << PP::nbsp
     << PPExtString(cast<InterfaceOp>(*interface).getSymName()) << PP::nbsp
     << PPExtString(getSymOpName(instance)) << " (.*);" << PP::newline;
  setPendingNewline();
}

void ModuleEmitter::emitParameters(Operation *module, ArrayAttr params) {
  if (params.empty())
    return;

  auto printParamType = [&](Type type, Attribute defaultValue,
                            SmallString<8> &result) {
    result.clear();
    llvm::raw_svector_ostream sstream(result);

    // If there is a default value like "32" then just print without type at
    // all.
    if (defaultValue) {
      if (auto intAttr = defaultValue.dyn_cast<IntegerAttr>())
        if (intAttr.getValue().getBitWidth() == 32)
          return;
      if (auto fpAttr = defaultValue.dyn_cast<FloatAttr>())
        if (fpAttr.getType().isF64())
          return;
    }
    if (type.isa<NoneType>())
      return;

    // Classic Verilog parser don't allow a type in the parameter declaration.
    // For compatibility with them, we omit the type when it is implicit based
    // on its initializer value, and print the type commented out when it is
    // a 32-bit "integer" parameter.
    if (auto intType = type_dyn_cast<IntegerType>(type))
      if (intType.getWidth() == 32) {
        sstream << "/*integer*/";
        return;
      }

    printPackedType(type, sstream, module->getLoc(),
                    /*optionalAliasType=*/Type(),
                    /*implicitIntType=*/true,
                    // Print single-bit values as explicit `[0:0]` type.
                    /*singleBitDefaultType=*/false);
  };

  // Determine the max width of the parameter types so things are lined up.
  size_t maxTypeWidth = 0;
  SmallString<8> scratch;
  for (auto param : params) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    // Measure the type length by printing it to a temporary string.
    printParamType(paramAttr.getType(), paramAttr.getValue(), scratch);
    maxTypeWidth = std::max(scratch.size(), maxTypeWidth);
  }

  if (maxTypeWidth > 0) // add a space if any type exists.
    maxTypeWidth += 1;

  ps.scopedBox(PP::bbox2, [&]() {
    ps << PP::newline << "#(";
    ps.scopedBox(PP::cbox0, [&]() {
      llvm::interleave(
          params,
          [&](Attribute param) {
            auto paramAttr = param.cast<ParamDeclAttr>();
            auto defaultValue = paramAttr.getValue(); // may be null if absent.
            ps << "parameter ";
            printParamType(paramAttr.getType(), defaultValue, scratch);
            if (!scratch.empty())
              ps << scratch;
            if (scratch.size() < maxTypeWidth)
              ps.nbsp(maxTypeWidth - scratch.size());

            ps << PPExtString(state.globalNames.getParameterVerilogName(
                module, paramAttr.getName()));

            if (defaultValue) {
              ps << " = ";
              ps.invokeWithStringOS([&](auto &os) {
                printParamValue(defaultValue, os, [&]() {
                  return module->emitError("parameter '")
                         << paramAttr.getName().getValue()
                         << "' has invalid value";
                });
              });
            }
          },
          [&]() { ps << "," << PP::newline; });
      ps << ") ";
    });
  });
}

void ModuleEmitter::emitPortList(Operation *module,
                                 const ModulePortInfo &portInfo) {
  ps << "(";
  if (portInfo.size())
    emitLocationInfo(module->getLoc());

  // Determine the width of the widest type we have to print so everything
  // lines up nicely.
  bool hasOutputs = false, hasZeroWidth = false;
  size_t maxTypeWidth = 0, lastNonZeroPort = -1;
  SmallVector<SmallString<8>, 16> portTypeStrings;

  for (size_t i = 0, e = portInfo.size(); i < e; ++i) {
    auto port = portInfo.at(i);
    hasOutputs |= port.isOutput();
    hasZeroWidth |= isZeroBitType(port.type);
    if (!isZeroBitType(port.type))
      lastNonZeroPort = i;

    // Convert the port's type to a string and measure it.
    portTypeStrings.push_back({});
    {
      llvm::raw_svector_ostream stringStream(portTypeStrings.back());
      printPackedType(stripUnpackedTypes(port.type), stringStream,
                      module->getLoc());
    }

    maxTypeWidth = std::max(portTypeStrings.back().size(), maxTypeWidth);
  }

  if (maxTypeWidth > 0) // add a space if any type exists
    maxTypeWidth += 1;

  // Emit the port list.
  ps.scopedBox(PP::bbox2, [&]() {
    for (size_t portIdx = 0, e = portInfo.size(); portIdx != e;) {
      auto lastPort = e - 1;

      ps << PP::newline;
      auto portType = portInfo.at(portIdx).type;

      // If this is a zero width type, emit the port as a comment and create a
      // neverbox to ensure we don't insert a line break.
      bool isZeroWidth = false;
      if (hasZeroWidth) {
        isZeroWidth = isZeroBitType(portType);
        if (isZeroWidth)
          ps << PP::neverbox;
        ps << (isZeroWidth ? "// " : "   ");
      }

      // Emit the port direction.
      auto thisPortDirection = portInfo.at(portIdx).dir;
      switch (thisPortDirection) {
      case ModulePort::Direction::Output:
        ps << "output ";
        break;
      case ModulePort::Direction::Input:
        ps << (hasOutputs ? "input  " : "input ");
        break;
      case ModulePort::Direction::InOut:
        ps << (hasOutputs ? "inout  " : "inout ");
        break;
      }
      bool emitWireInPorts = state.options.emitWireInPorts;
      if (emitWireInPorts)
        ps << "wire ";

      // Emit the type.
      if (!portTypeStrings[portIdx].empty())
        ps << portTypeStrings[portIdx];
      if (portTypeStrings[portIdx].size() < maxTypeWidth)
        ps.nbsp(maxTypeWidth - portTypeStrings[portIdx].size());

      size_t startOfNamePos =
          (hasOutputs ? 7 : 6) + (emitWireInPorts ? 5 : 0) + maxTypeWidth;

      // Emit the name.
      ps << PPExtString(getPortVerilogName(module, portInfo.at(portIdx)));

      // Emit array dimensions.
      ps.invokeWithStringOS(
          [&](auto &os) { printUnpackedTypePostfix(portType, os); });

      // Emit the symbol.
      auto innerSym = portInfo.at(portIdx).sym;
      if (state.options.printDebugInfo && innerSym && !innerSym.empty()) {
        ps << " /* ";
        ps.invokeWithStringOS([&](auto &os) { os << innerSym; });
        ps << " */";
      }

      // Emit the comma if this is not the last real port.
      if (portIdx != lastNonZeroPort && portIdx != lastPort)
        ps << ",";

      // Emit the location.
      if (auto loc = portInfo.at(portIdx).loc)
        emitLocationInfo(loc);

      if (isZeroWidth)
        ps << PP::end; // Close never-break group.

      ++portIdx;

      // If we have any more ports with the same types and the same
      // direction, emit them in a list one per line. Optionally skip this
      // behavior when requested by user.
      if (!state.options.disallowPortDeclSharing) {
        while (portIdx != e && portInfo.at(portIdx).dir == thisPortDirection &&
               stripUnpackedTypes(portType) ==
                   stripUnpackedTypes(portInfo.at(portIdx).type)) {
          auto port = portInfo.at(portIdx);
          // Append this to the running port decl.
          ps << PP::newline;

          bool isZeroWidth = false;
          if (hasZeroWidth) {
            isZeroWidth = isZeroBitType(portType);
            if (isZeroWidth)
              ps << PP::neverbox;
            ps << (isZeroWidth ? "// " : "   ");
          }

          ps.nbsp(startOfNamePos);

          // Emit the name.
          StringRef name = getPortVerilogName(module, port);
          ps << PPExtString(name);

          // Emit array dimensions.
          ps.invokeWithStringOS(
              [&](auto &os) { printUnpackedTypePostfix(port.type, os); });

          // Emit the symbol.
          if (state.options.printDebugInfo && port.sym && !port.sym.empty())
            ps << " /* inner_sym: "
               << PPExtString(port.sym.getSymName().getValue()) << " */";

          // Emit the comma if this is not the last real port.
          if (portIdx != lastNonZeroPort && portIdx != lastPort)
            ps << ",";

          // Emit the location.
          if (auto loc = port.loc)
            emitLocationInfo(loc);

          if (isZeroWidth)
            ps << PP::end; // Close never-break group.

          ++portIdx;
        }
      }
    }
  });

  if (!portInfo.size()) {
    ps << ");";
    SmallPtrSet<Operation *, 8> moduleOpSet;
    moduleOpSet.insert(module);
    emitLocationInfoAndNewLine(moduleOpSet);
  } else {
    ps << PP::newline;
    ps << ");" << PP::newline;
    setPendingNewline();
  }
}

void ModuleEmitter::emitHWModule(HWModuleOp module) {
  currentModuleOp = module;

  emitComment(module.getCommentAttr());
  emitSVAttributes(module);
  startStatement();
  ps << "module " << PPExtString(getVerilogModuleName(module));

  // If we have any parameters, print them on their own line.
  emitParameters(module, module.getParameters());

  emitPortList(module, module.getPortList());

  assert(state.pendingNewline);

  // Emit the body of the module.
  StmtEmitter(*this).emitStatementBlock(*module.getBodyBlock());
  startStatement();
  ps << "endmodule" << PP::newline;
  setPendingNewline();

  currentModuleOp = nullptr;
}

void ModuleEmitter::emitHWTestModule(HWTestModuleOp module) {
  currentModuleOp = module;

  emitComment(module.getCommentAttr());
  emitSVAttributes(module);
  startStatement();
  ps << "module " << PPExtString(getVerilogModuleName(module));

  // If we have any parameters, print them on their own line.
  emitParameters(module, module.getParameters());

  emitPortList(module, module.getPortList());

  assert(state.pendingNewline);

  // Emit the body of the module.
  StmtEmitter(*this).emitStatementBlock(*module.getBodyBlock());
  startStatement();
  ps << "endmodule" << PP::newline;
  setPendingNewline();

  currentModuleOp = nullptr;
}

//===----------------------------------------------------------------------===//
// Top level "file" emitter logic
//===----------------------------------------------------------------------===//

/// Organize the operations in the root MLIR module into output files to be
/// generated. If `separateModules` is true, a handful of top-level
/// declarations will be split into separate output files even in the absence
/// of an explicit output file attribute.
void SharedEmitterState::gatherFiles(bool separateModules) {

  /// Collect all the inner names from the specified module and add them to the
  /// IRCache.  Declarations (named things) only exist at the top level of the
  /// module.  Also keep track of any modules that contain bind operations.
  /// These are non-hierarchical references which we need to be careful about
  /// during emission.
  auto collectInstanceSymbolsAndBinds = [&](Operation *moduleOp) {
    moduleOp->walk([&](Operation *op) {
      // Populate the symbolCache with all operations that can define a symbol.
      if (auto name = op->getAttrOfType<InnerSymAttr>(
              hw::InnerSymbolTable::getInnerSymbolAttrName()))
        symbolCache.addDefinition(moduleOp->getAttrOfType<StringAttr>(
                                      SymbolTable::getSymbolAttrName()),
                                  name.getSymName(), op);
      if (isa<BindOp>(op))
        modulesContainingBinds.insert(moduleOp);
    });
  };
  /// Collect any port marked as being referenced via symbol.
  auto collectPorts = [&](auto moduleOp) {
    auto portInfo = moduleOp.getPortList();
    for (auto [i, p] : llvm::enumerate(portInfo)) {
      if (!p.attrs || p.attrs.empty())
        continue;
      for (NamedAttribute portAttr : p.attrs) {
        if (auto sym = portAttr.getValue().dyn_cast<InnerSymAttr>()) {
          symbolCache.addDefinition(moduleOp.getNameAttr(), sym.getSymName(),
                                    moduleOp, i);
        }
      }
    }
  };

  SmallString<32> outputPath;
  for (auto &op : *designOp.getBody()) {
    auto info = OpFileInfo{&op, replicatedOps.size()};

    bool hasFileName = false;
    bool emitReplicatedOps = true;
    bool addToFilelist = true;

    outputPath.clear();

    // Check if the operation has an explicit `output_file` attribute set. If
    // it does, extract the information from the attribute.
    auto attr = op.getAttrOfType<hw::OutputFileAttr>("output_file");
    if (attr) {
      LLVM_DEBUG(llvm::dbgs() << "Found output_file attribute " << attr
                              << " on " << op << "\n";);
      if (!attr.isDirectory())
        hasFileName = true;
      appendPossiblyAbsolutePath(outputPath, attr.getFilename().getValue());
      emitReplicatedOps = attr.getIncludeReplicatedOps().getValue();
      addToFilelist = !attr.getExcludeFromFilelist().getValue();
    }

    // Collect extra file lists to output the file to.
    SmallVector<StringAttr> opFileList;
    if (auto fl = op.getAttrOfType<hw::FileListAttr>("output_filelist"))
      opFileList.push_back(fl.getFilename());
    if (auto fla = op.getAttrOfType<ArrayAttr>("output_filelist"))
      for (auto fl : fla)
        opFileList.push_back(fl.cast<hw::FileListAttr>().getFilename());

    auto separateFile = [&](Operation *op, Twine defaultFileName = "") {
      // If we're emitting to a separate file and the output_file attribute
      // didn't specify a filename, take the default one if present or emit an
      // error if not.
      if (!hasFileName) {
        if (!defaultFileName.isTriviallyEmpty()) {
          llvm::sys::path::append(outputPath, defaultFileName);
        } else {
          op->emitError("file name unspecified");
          encounteredError = true;
          llvm::sys::path::append(outputPath, "error.out");
        }
      }

      auto destFile = StringAttr::get(op->getContext(), outputPath);
      auto &file = files[destFile];
      file.ops.push_back(info);
      file.emitReplicatedOps = emitReplicatedOps;
      file.addToFilelist = addToFilelist;
      file.isVerilog = outputPath.endswith(".sv");
      for (auto fl : opFileList)
        fileLists[fl.getValue()].push_back(destFile);
    };

    // Separate the operation into dedicated output file, or emit into the
    // root file, or replicate in all output files.
    TypeSwitch<Operation *>(&op)
        .Case<HWModuleOp>([&](auto mod) {
          // Build the IR cache.
          symbolCache.addDefinition(mod.getNameAttr(), mod);
          collectPorts(mod);
          collectInstanceSymbolsAndBinds(mod);

          // Emit into a separate file named after the module.
          if (attr || separateModules)
            separateFile(mod, getVerilogModuleName(mod) + ".sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<HWTestModuleOp>([&](auto mod) {
          // Build the IR cache.
          symbolCache.addDefinition(mod.getNameAttr(), mod);
          collectPorts(mod);
          collectInstanceSymbolsAndBinds(mod);

          // Emit into a separate file named after the module.
          if (attr || separateModules)
            separateFile(mod, getVerilogModuleName(mod) + ".sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<InterfaceOp>([&](InterfaceOp intf) {
          // Build the IR cache.
          symbolCache.addDefinition(intf.getNameAttr(), intf);
          // Populate the symbolCache with all operations that can define a
          // symbol.
          for (auto &op : *intf.getBodyBlock())
            if (auto symOp = dyn_cast<mlir::SymbolOpInterface>(op))
              if (auto name = symOp.getNameAttr())
                symbolCache.addDefinition(name, symOp);

          // Emit into a separate file named after the interface.
          if (attr || separateModules)
            separateFile(intf, intf.getSymName() + ".sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<HWModuleExternOp>([&](HWModuleExternOp op) {
          // Build the IR cache.
          symbolCache.addDefinition(op.getNameAttr(), op);
          collectPorts(op);
          if (separateModules)
            separateFile(op, "extern_modules.sv");
          else
            rootFile.ops.push_back(info);
        })
        .Case<VerbatimOp, IfDefOp, MacroDefOp>([&](Operation *op) {
          // Emit into a separate file using the specified file name or
          // replicate the operation in each outputfile.
          if (!attr) {
            replicatedOps.push_back(op);
          } else
            separateFile(op, "");
        })
        .Case<HWGeneratorSchemaOp>([&](HWGeneratorSchemaOp schemaOp) {
          symbolCache.addDefinition(schemaOp.getNameAttr(), schemaOp);
        })
        .Case<GlobalRefOp>([&](GlobalRefOp globalRefOp) {
          symbolCache.addDefinition(globalRefOp.getSymNameAttr(), globalRefOp);
        })
        .Case<HierPathOp>([&](HierPathOp hierPathOp) {
          symbolCache.addDefinition(hierPathOp.getSymNameAttr(), hierPathOp);
        })
        .Case<TypeScopeOp>([&](TypeScopeOp op) {
          symbolCache.addDefinition(op.getNameAttr(), op);
          // TODO: How do we want to handle typedefs in a split output?
          if (!attr) {
            replicatedOps.push_back(op);
          } else
            separateFile(op, "");
        })
        .Case<BindOp, BindInterfaceOp>([&](auto op) {
          if (!attr) {
            separateFile(op, "bindfile.sv");
          } else {
            separateFile(op);
          }
        })
        .Case<MacroDeclOp>([&](auto op) {
          symbolCache.addDefinition(op.getSymNameAttr(), op);
        })
        .Case<om::ClassLike>([&](auto op) {
          symbolCache.addDefinition(op.getSymNameAttr(), op);
        })
        .Case<om::ConstantOp>([&](auto op) {
          // Constant ops might reference symbols, skip them.
        })
        .Default([&](auto *) {
          op.emitError("unknown operation (SharedEmitterState::gatherFiles)");
          encounteredError = true;
        });
  }

  // We've built the whole symbol cache.  Freeze it so things can start
  // querying it (potentially concurrently).
  symbolCache.freeze();
}

/// Given a FileInfo, collect all the replicated and designated operations
/// that go into it and append them to "thingsToEmit".
void SharedEmitterState::collectOpsForFile(const FileInfo &file,
                                           EmissionList &thingsToEmit,
                                           bool emitHeader) {
  // Include the version string comment when the file is verilog.
  if (file.isVerilog && !options.omitVersionComment)
    thingsToEmit.emplace_back(circt::getCirctVersionComment());

  // If we're emitting replicated ops, keep track of where we are in the list.
  size_t lastReplicatedOp = 0;

  bool emitHeaderInclude =
      emitHeader && file.emitReplicatedOps && !file.isHeader;

  if (emitHeaderInclude)
    thingsToEmit.emplace_back(circtHeaderInclude);

  size_t numReplicatedOps =
      file.emitReplicatedOps && !emitHeaderInclude ? replicatedOps.size() : 0;

  thingsToEmit.reserve(thingsToEmit.size() + numReplicatedOps +
                       file.ops.size());

  // Emit each operation in the file preceded by the replicated ops not yet
  // printed.
  for (const auto &opInfo : file.ops) {
    // Emit the replicated per-file operations before the main operation's
    // position (if enabled).
    for (; lastReplicatedOp < std::min(opInfo.position, numReplicatedOps);
         ++lastReplicatedOp)
      thingsToEmit.emplace_back(replicatedOps[lastReplicatedOp]);

    // Emit the operation itself.
    thingsToEmit.emplace_back(opInfo.op);
  }

  // Emit the replicated per-file operations after the last operation (if
  // enabled).
  for (; lastReplicatedOp < numReplicatedOps; lastReplicatedOp++)
    thingsToEmit.emplace_back(replicatedOps[lastReplicatedOp]);
}

static void emitOperation(VerilogEmitterState &state, Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<HWModuleOp>([&](auto op) { ModuleEmitter(state).emitHWModule(op); })
      .Case<HWTestModuleOp>(
          [&](auto op) { ModuleEmitter(state).emitHWTestModule(op); })
      .Case<HWModuleExternOp>(
          [&](auto op) { ModuleEmitter(state).emitHWExternModule(op); })
      .Case<HWModuleGeneratedOp>(
          [&](auto op) { ModuleEmitter(state).emitHWGeneratedModule(op); })
      .Case<HWGeneratorSchemaOp, hw::GlobalRefOp>([&](auto op) { /* Empty */ })
      .Case<BindOp>([&](auto op) { ModuleEmitter(state).emitBind(op); })
      .Case<BindInterfaceOp>(
          [&](auto op) { ModuleEmitter(state).emitBindInterface(op); })
      .Case<InterfaceOp, VerbatimOp, IfDefOp>(
          [&](auto op) { ModuleEmitter(state).emitStatement(op); })
      .Case<TypeScopeOp>([&](auto typedecls) {
        ModuleEmitter(state).emitStatement(typedecls);
      })
      .Case<MacroDefOp>(
          [&](auto op) { ModuleEmitter(state).emitStatement(op); })
      .Default([&](auto *op) {
        state.encounteredError = true;
        op->emitError("unknown operation (ExportVerilog::emitOperation)");
      });
}

/// Actually emit the collected list of operations and strings to the
/// specified file.
void SharedEmitterState::emitOps(EmissionList &thingsToEmit, raw_ostream &os,
                                 bool parallelize) {
  MLIRContext *context = designOp->getContext();

  // Disable parallelization overhead if MLIR threading is disabled.
  if (parallelize)
    parallelize &= context->isMultithreadingEnabled();

  // If we aren't parallelizing output, directly output each operation to the
  // specified stream.
  if (!parallelize) {
    VerilogEmitterState state(designOp, *this, options, symbolCache,
                              globalNames, os);
    for (auto &entry : thingsToEmit) {
      if (auto *op = entry.getOperation())
        emitOperation(state, op);
      else
        os << entry.getStringData();
    }

    if (state.encounteredError)
      encounteredError = true;
    return;
  }

  // If we are parallelizing emission, we emit each independent operation to a
  // string buffer in parallel, then concat at the end.
  parallelForEach(context, thingsToEmit, [&](StringOrOpToEmit &stringOrOp) {
    auto *op = stringOrOp.getOperation();
    if (!op)
      return; // Ignore things that are already strings.

    // BindOp emission reaches into the hw.module of the instance, and that
    // body may be being transformed by its own emission.  Defer their
    // emission to the serial phase.  They are speedy to emit anyway.
    if (isa<BindOp>(op) || modulesContainingBinds.count(op))
      return;

    SmallString<256> buffer;
    llvm::raw_svector_ostream tmpStream(buffer);
    VerilogEmitterState state(designOp, *this, options, symbolCache,
                              globalNames, tmpStream);
    emitOperation(state, op);
    stringOrOp.setString(buffer);
  });

  // Finally emit each entry now that we know it is a string.
  for (auto &entry : thingsToEmit) {
    // Almost everything is lowered to a string, just concat the strings onto
    // the output stream.
    auto *op = entry.getOperation();
    if (!op) {
      os << entry.getStringData();
      continue;
    }

    // If this wasn't emitted to a string (e.g. it is a bind) do so now.
    VerilogEmitterState state(designOp, *this, options, symbolCache,
                              globalNames, os);
    emitOperation(state, op);
  }
}

//===----------------------------------------------------------------------===//
// Unified Emitter
//===----------------------------------------------------------------------===//

static LogicalResult exportVerilogImpl(ModuleOp module, llvm::raw_ostream &os) {
  LoweringOptions options(module);
  GlobalNameTable globalNames = legalizeGlobalNames(module, options);

  SharedEmitterState emitter(module, options, std::move(globalNames));
  emitter.gatherFiles(false);

  if (emitter.options.emitReplicatedOpsToHeader)
    module.emitWarning()
        << "`emitReplicatedOpsToHeader` option is enabled but an header is "
           "created only at SplitExportVerilog";

  SharedEmitterState::EmissionList list;

  // Collect the contents of the main file. This is a container for anything
  // not explicitly split out into a separate file.
  emitter.collectOpsForFile(emitter.rootFile, list);

  // Emit the separate files.
  for (const auto &it : emitter.files) {
    list.emplace_back("\n// ----- 8< ----- FILE \"" + it.first.str() +
                      "\" ----- 8< -----\n\n");
    emitter.collectOpsForFile(it.second, list);
  }

  // Emit the filelists.
  for (auto &it : emitter.fileLists) {
    std::string contents("\n// ----- 8< ----- FILE \"" + it.first().str() +
                         "\" ----- 8< -----\n\n");
    for (auto &name : it.second)
      contents += name.str() + "\n";
    list.emplace_back(contents);
  }

  // Finally, emit all the ops we collected.
  emitter.emitOps(list, os, /*parallelize=*/true);
  return failure(emitter.encounteredError);
}

LogicalResult circt::exportVerilog(ModuleOp module, llvm::raw_ostream &os) {
  LoweringOptions options(module);
  SmallVector<HWModuleOp> modulesToPrepare;
  module.walk([&](HWModuleOp op) { modulesToPrepare.push_back(op); });
  if (failed(failableParallelForEach(
          module->getContext(), modulesToPrepare,
          [&](auto op) { return prepareHWModule(op, options); })))
    return failure();
  return exportVerilogImpl(module, os);
}

namespace {

struct ExportVerilogPass : public ExportVerilogBase<ExportVerilogPass> {
  ExportVerilogPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    // Prepare the ops in the module for emission.
    mlir::OpPassManager preparePM("builtin.module");
    preparePM.addPass(createLegalizeAnonEnumsPass());
    auto &modulePM = preparePM.nest<hw::HWModuleOp>();
    modulePM.addPass(createPrepareForEmissionPass());
    if (failed(runPipeline(preparePM, getOperation())))
      return signalPassFailure();

    if (failed(exportVerilogImpl(getOperation(), os)))
      return signalPassFailure();
  }

private:
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::createExportVerilogPass(llvm::raw_ostream &os) {
  return std::make_unique<ExportVerilogPass>(os);
}

std::unique_ptr<mlir::Pass> circt::createExportVerilogPass() {
  return createExportVerilogPass(llvm::outs());
}

//===----------------------------------------------------------------------===//
// Split Emitter
//===----------------------------------------------------------------------===//

static std::unique_ptr<llvm::ToolOutputFile>
createOutputFile(StringRef fileName, StringRef dirname,
                 SharedEmitterState &emitter) {
  // Determine the output path from the output directory and filename.
  SmallString<128> outputFilename(dirname);
  appendPossiblyAbsolutePath(outputFilename, fileName);
  auto outputDir = llvm::sys::path::parent_path(outputFilename);

  // Create the output directory if needed.
  std::error_code error = llvm::sys::fs::create_directories(outputDir);
  if (error) {
    emitter.designOp.emitError("cannot create output directory \"")
        << outputDir << "\": " << error.message();
    emitter.encounteredError = true;
    return {};
  }

  // Open the output file.
  std::string errorMessage;
  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    emitter.designOp.emitError(errorMessage);
    emitter.encounteredError = true;
  }
  return output;
}

static void createSplitOutputFile(StringAttr fileName, FileInfo &file,
                                  StringRef dirname,
                                  SharedEmitterState &emitter) {
  auto output = createOutputFile(fileName, dirname, emitter);
  if (!output)
    return;

  SharedEmitterState::EmissionList list;
  emitter.collectOpsForFile(file, list,
                            emitter.options.emitReplicatedOpsToHeader);

  // Emit the file, copying the global options into the individual module
  // state.  Don't parallelize emission of the ops within this file - we
  // already parallelize per-file emission and we pay a string copy overhead
  // for parallelization.
  emitter.emitOps(list, output->os(), /*parallelize=*/false);
  output->keep();
}

static LogicalResult exportSplitVerilogImpl(ModuleOp module,
                                            StringRef dirname) {
  // Prepare the ops in the module for emission and legalize the names that will
  // end up in the output.
  LoweringOptions options(module);
  GlobalNameTable globalNames = legalizeGlobalNames(module, options);

  SharedEmitterState emitter(module, options, std::move(globalNames));
  emitter.gatherFiles(true);

  if (emitter.options.emitReplicatedOpsToHeader) {
    // Add a header to the file list.
    bool insertSuccess =
        emitter.files
            .insert({StringAttr::get(module.getContext(), circtHeader),
                     FileInfo{/*ops*/ {},
                              /*emitReplicatedOps*/ true,
                              /*addToFilelist*/ true,
                              /*isHeader*/ true}})
            .second;
    if (!insertSuccess) {
      module.emitError() << "tried to emit a heder to " << circtHeader
                         << ", but the file is used as an output too.";
      return failure();
    }
  }

  // Emit each file in parallel if context enables it.
  parallelForEach(module->getContext(), emitter.files.begin(),
                  emitter.files.end(), [&](auto &it) {
                    createSplitOutputFile(it.first, it.second, dirname,
                                          emitter);
                  });

  // Write the file list.
  SmallString<128> filelistPath(dirname);
  llvm::sys::path::append(filelistPath, "filelist.f");

  std::string errorMessage;
  auto output = mlir::openOutputFile(filelistPath, &errorMessage);
  if (!output) {
    module->emitError(errorMessage);
    return failure();
  }

  for (const auto &it : emitter.files) {
    if (it.second.addToFilelist)
      output->os() << it.first.str() << "\n";
  }
  output->keep();

  // Emit the filelists.
  for (auto &it : emitter.fileLists) {
    auto output = createOutputFile(it.first(), dirname, emitter);
    if (!output)
      continue;
    for (auto &name : it.second)
      output->os() << name.str() << "\n";
    output->keep();
  }

  return failure(emitter.encounteredError);
}

LogicalResult circt::exportSplitVerilog(ModuleOp module, StringRef dirname) {
  LoweringOptions options(module);
  SmallVector<HWModuleOp> modulesToPrepare;
  module.walk([&](HWModuleOp op) { modulesToPrepare.push_back(op); });
  if (failed(failableParallelForEach(
          module->getContext(), modulesToPrepare,
          [&](auto op) { return prepareHWModule(op, options); })))
    return failure();

  return exportSplitVerilogImpl(module, dirname);
}

namespace {

struct ExportSplitVerilogPass
    : public ExportSplitVerilogBase<ExportSplitVerilogPass> {
  ExportSplitVerilogPass(StringRef directory) {
    directoryName = directory.str();
  }
  void runOnOperation() override {
    // Prepare the ops in the module for emission.
    mlir::OpPassManager preparePM("builtin.module");
    auto &modulePM = preparePM.nest<hw::HWModuleOp>();
    modulePM.addPass(createPrepareForEmissionPass());
    if (failed(runPipeline(preparePM, getOperation())))
      return signalPassFailure();

    if (failed(exportSplitVerilogImpl(getOperation(), directoryName)))
      return signalPassFailure();
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::createExportSplitVerilogPass(StringRef directory) {
  return std::make_unique<ExportSplitVerilogPass>(directory);
}
