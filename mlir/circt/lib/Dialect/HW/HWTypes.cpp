//===- HWTypes.cpp - HW types code defs -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation logic for HW data types.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::hw;
using namespace circt::hw::detail;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/HW/HWTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type Helpers
//===----------------------------------------------------------------------===/

mlir::Type circt::hw::getCanonicalType(mlir::Type type) {
  Type canonicalType;
  if (auto typeAlias = type.dyn_cast<TypeAliasType>())
    canonicalType = typeAlias.getCanonicalType();
  else
    canonicalType = type;
  return canonicalType;
}

/// Return true if the specified type is a value HW Integer type.  This checks
/// that it is a signless standard dialect type or a hw::IntType.
bool circt::hw::isHWIntegerType(mlir::Type type) {
  Type canonicalType = getCanonicalType(type);

  if (canonicalType.isa<hw::IntType>())
    return true;

  auto intType = canonicalType.dyn_cast<IntegerType>();
  if (!intType || !intType.isSignless())
    return false;

  return true;
}

bool circt::hw::isHWEnumType(mlir::Type type) {
  return getCanonicalType(type).isa<hw::EnumType>();
}

/// Return true if the specified type can be used as an HW value type, that is
/// the set of types that can be composed together to represent synthesized,
/// hardware but not marker types like InOutType.
bool circt::hw::isHWValueType(Type type) {
  // Signless and signed integer types are both valid.
  if (type.isa<IntegerType, IntType, EnumType>())
    return true;

  if (auto array = type.dyn_cast<ArrayType>())
    return isHWValueType(array.getElementType());

  if (auto array = type.dyn_cast<UnpackedArrayType>())
    return isHWValueType(array.getElementType());

  if (auto t = type.dyn_cast<StructType>())
    return llvm::all_of(t.getElements(),
                        [](auto f) { return isHWValueType(f.type); });

  if (auto t = type.dyn_cast<UnionType>())
    return llvm::all_of(t.getElements(),
                        [](auto f) { return isHWValueType(f.type); });

  if (auto t = type.dyn_cast<TypeAliasType>())
    return isHWValueType(t.getCanonicalType());

  return false;
}

/// Return the hardware bit width of a type. Does not reflect any encoding,
/// padding, or storage scheme, just the bit (and wire width) of a
/// statically-size type. Reflects the number of wires needed to transmit a
/// value of this type. Returns -1 if the type is not known or cannot be
/// statically computed.
int64_t circt::hw::getBitWidth(mlir::Type type) {
  return llvm::TypeSwitch<::mlir::Type, size_t>(type)
      .Case<IntegerType>(
          [](IntegerType t) { return t.getIntOrFloatBitWidth(); })
      .Case<ArrayType, UnpackedArrayType>([](auto a) {
        int64_t elementBitWidth = getBitWidth(a.getElementType());
        if (elementBitWidth < 0)
          return elementBitWidth;
        int64_t dimBitWidth = a.getSize();
        if (dimBitWidth < 0)
          return static_cast<int64_t>(-1L);
        return (int64_t)a.getSize() * elementBitWidth;
      })
      .Case<StructType>([](StructType s) {
        int64_t total = 0;
        for (auto field : s.getElements()) {
          int64_t fieldSize = getBitWidth(field.type);
          if (fieldSize < 0)
            return fieldSize;
          total += fieldSize;
        }
        return total;
      })
      .Case<UnionType>([](UnionType u) {
        int64_t maxSize = 0;
        for (auto field : u.getElements()) {
          int64_t fieldSize = getBitWidth(field.type) + field.offset;
          if (fieldSize > maxSize)
            maxSize = fieldSize;
        }
        return maxSize;
      })
      .Case<EnumType>([](EnumType e) { return e.getBitWidth(); })
      .Case<TypeAliasType>(
          [](TypeAliasType t) { return getBitWidth(t.getCanonicalType()); })
      .Default([](Type) { return -1; });
}

/// Return true if the specified type contains known marker types like
/// InOutType.  Unlike isHWValueType, this is not conservative, it only returns
/// false on known InOut types, rather than any unknown types.
bool circt::hw::hasHWInOutType(Type type) {
  if (auto array = type.dyn_cast<ArrayType>())
    return hasHWInOutType(array.getElementType());

  if (auto array = type.dyn_cast<UnpackedArrayType>())
    return hasHWInOutType(array.getElementType());

  if (auto t = type.dyn_cast<StructType>()) {
    return std::any_of(t.getElements().begin(), t.getElements().end(),
                       [](const auto &f) { return hasHWInOutType(f.type); });
  }

  if (auto t = type.dyn_cast<TypeAliasType>())
    return hasHWInOutType(t.getCanonicalType());

  return type.isa<InOutType>();
}

/// Parse and print nested HW types nicely.  These helper methods allow eliding
/// the "hw." prefix on array, inout, and other types when in a context that
/// expects HW subelement types.
static ParseResult parseHWElementType(Type &result, AsmParser &p) {
  // If this is an HW dialect type, then we don't need/want the !hw. prefix
  // redundantly specified.
  auto fullString = static_cast<DialectAsmParser &>(p).getFullSymbolSpec();
  auto *curPtr = p.getCurrentLocation().getPointer();
  auto typeString =
      StringRef(curPtr, fullString.size() - (curPtr - fullString.data()));

  if (typeString.startswith("array<") || typeString.startswith("inout<") ||
      typeString.startswith("uarray<") || typeString.startswith("struct<") ||
      typeString.startswith("typealias<") || typeString.startswith("int<") ||
      typeString.startswith("enum<")) {
    llvm::StringRef mnemonic;
    auto parseResult = generatedTypeParser(p, &mnemonic, result);
    return parseResult.has_value() ? success() : failure();
  }

  return p.parseType(result);
}

static void printHWElementType(Type element, AsmPrinter &p) {
  if (succeeded(generatedTypePrinter(element, p)))
    return;
  p.printType(element);
}

//===----------------------------------------------------------------------===//
// Int Type
//===----------------------------------------------------------------------===//

Type IntType::get(mlir::TypedAttr width) {
  // The width expression must always be a 32-bit wide integer type itself.
  auto widthWidth = width.getType().dyn_cast<IntegerType>();
  assert(widthWidth && widthWidth.getWidth() == 32 &&
         "!hw.int width must be 32-bits");
  (void)widthWidth;

  if (auto cstWidth = width.dyn_cast<IntegerAttr>())
    return IntegerType::get(width.getContext(),
                            cstWidth.getValue().getZExtValue());

  return Base::get(width.getContext(), width);
}

Type IntType::parse(AsmParser &p) {
  // The bitwidth of the parameter size is always 32 bits.
  auto int32Type = p.getBuilder().getIntegerType(32);

  mlir::TypedAttr width;
  if (p.parseLess() || p.parseAttribute(width, int32Type) || p.parseGreater())
    return Type();
  return get(width);
}

void IntType::print(AsmPrinter &p) const {
  p << "<";
  p.printAttributeWithoutType(getWidth());
  p << '>';
}

//===----------------------------------------------------------------------===//
// Struct Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace hw {
namespace detail {
bool operator==(const FieldInfo &a, const FieldInfo &b) {
  return a.name == b.name && a.type == b.type;
}
llvm::hash_code hash_value(const FieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type);
}
} // namespace detail
} // namespace hw
} // namespace circt

/// Parse a list of field names and types within <>. E.g.:
/// <foo: i7, bar: i8>
static ParseResult parseFields(AsmParser &p,
                               SmallVectorImpl<FieldInfo> &parameters) {
  return p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
        StringRef name;
        Type type;
        if (p.parseKeyword(&name) || p.parseColon() || p.parseType(type))
          return failure();
        parameters.push_back(
            FieldInfo{StringAttr::get(p.getContext(), name), type});
        return success();
      });
}

/// Print out a list of named fields surrounded by <>.
static void printFields(AsmPrinter &p, ArrayRef<FieldInfo> fields) {
  p << '<';
  llvm::interleaveComma(fields, p, [&](const FieldInfo &field) {
    p << field.name.getValue() << ": " << field.type;
  });
  p << ">";
}

Type StructType::parse(AsmParser &p) {
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(p, parameters))
    return Type();
  return get(p.getContext(), parameters);
}

void StructType::print(AsmPrinter &p) const { printFields(p, getElements()); }

Type StructType::getFieldType(mlir::StringRef fieldName) {
  for (const auto &field : getElements())
    if (field.name == fieldName)
      return field.type;
  return Type();
}

std::optional<unsigned> StructType::getFieldIndex(mlir::StringRef fieldName) {
  ArrayRef<hw::StructType::FieldInfo> elems = getElements();
  for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
    if (elems[idx].name == fieldName)
      return idx;
  return {};
}

std::optional<unsigned> StructType::getFieldIndex(mlir::StringAttr fieldName) {
  ArrayRef<hw::StructType::FieldInfo> elems = getElements();
  for (size_t idx = 0, numElems = elems.size(); idx < numElems; ++idx)
    if (elems[idx].name == fieldName)
      return idx;
  return {};
}

void StructType::getInnerTypes(SmallVectorImpl<Type> &types) {
  for (const auto &field : getElements())
    types.push_back(field.type);
}

//===----------------------------------------------------------------------===//
// Union Type
//===----------------------------------------------------------------------===//

namespace circt {
namespace hw {
namespace detail {
bool operator==(const OffsetFieldInfo &a, const OffsetFieldInfo &b) {
  return a.name == b.name && a.type == b.type && a.offset == b.offset;
}
// NOLINTNEXTLINE
llvm::hash_code hash_value(const OffsetFieldInfo &fi) {
  return llvm::hash_combine(fi.name, fi.type, fi.offset);
}
} // namespace detail
} // namespace hw
} // namespace circt

Type UnionType::parse(AsmParser &p) {
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (p.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::LessGreater, [&]() -> ParseResult {
            StringRef name;
            Type type;
            if (p.parseKeyword(&name) || p.parseColon() || p.parseType(type))
              return failure();
            size_t offset = 0;
            if (succeeded(p.parseOptionalKeyword("offset")))
              if (p.parseInteger(offset))
                return failure();
            parameters.push_back(UnionType::FieldInfo{
                StringAttr::get(p.getContext(), name), type, offset});
            return success();
          }))
    return Type();
  return get(p.getContext(), parameters);
}

void UnionType::print(AsmPrinter &odsPrinter) const {
  odsPrinter << '<';
  llvm::interleaveComma(
      getElements(), odsPrinter, [&](const UnionType::FieldInfo &field) {
        odsPrinter << field.name.getValue() << ": " << field.type;
        if (field.offset)
          odsPrinter << " offset " << field.offset;
      });
  odsPrinter << ">";
}

UnionType::FieldInfo UnionType::getFieldInfo(::mlir::StringRef fieldName) {
  for (const auto &field : getElements())
    if (field.name == fieldName)
      return field;
  return FieldInfo();
}

Type UnionType::getFieldType(mlir::StringRef fieldName) {
  return getFieldInfo(fieldName).type;
}

//===----------------------------------------------------------------------===//
// Enum Type
//===----------------------------------------------------------------------===//

Type EnumType::parse(AsmParser &p) {
  llvm::SmallVector<Attribute> fields;

  if (p.parseCommaSeparatedList(AsmParser::Delimiter::LessGreater, [&]() {
        StringRef name;
        if (p.parseKeyword(&name))
          return failure();
        fields.push_back(StringAttr::get(p.getContext(), name));
        return success();
      }))
    return Type();

  return get(p.getContext(), ArrayAttr::get(p.getContext(), fields));
}

void EnumType::print(AsmPrinter &p) const {
  p << '<';
  llvm::interleaveComma(getFields(), p, [&](Attribute enumerator) {
    p << enumerator.cast<StringAttr>().getValue();
  });
  p << ">";
}

bool EnumType::contains(mlir::StringRef field) {
  return indexOf(field).has_value();
}

std::optional<size_t> EnumType::indexOf(mlir::StringRef field) {
  for (auto it : llvm::enumerate(getFields()))
    if (it.value().cast<StringAttr>().getValue() == field)
      return it.index();
  return {};
}

size_t EnumType::getBitWidth() {
  auto w = getFields().size();
  if (w > 1)
    return llvm::Log2_64_Ceil(getFields().size());
  return 1;
}

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

static LogicalResult parseArray(AsmParser &p, Attribute &dim, Type &inner) {
  if (p.parseLess())
    return failure();

  uint64_t dimLiteral;
  auto int64Type = p.getBuilder().getIntegerType(64);

  if (auto res = p.parseOptionalInteger(dimLiteral); res.has_value())
    dim = p.getBuilder().getI64IntegerAttr(dimLiteral);
  else if (!p.parseOptionalAttribute(dim, int64Type).has_value())
    return failure();

  if (!dim.isa<IntegerAttr, ParamExprAttr, ParamDeclRefAttr>()) {
    p.emitError(p.getNameLoc(), "unsupported dimension kind in hw.array");
    return failure();
  }

  if (p.parseXInDimensionList() || parseHWElementType(inner, p) ||
      p.parseGreater())
    return failure();

  return success();
}

Type ArrayType::parse(AsmParser &p) {
  Attribute dim;
  Type inner;

  if (failed(parseArray(p, dim, inner)))
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner, dim)))
    return Type();

  return get(inner.getContext(), inner, dim);
}

void ArrayType::print(AsmPrinter &p) const {
  p << "<";
  p.printAttributeWithoutType(getSizeAttr());
  p << "x";
  printHWElementType(getElementType(), p);
  p << '>';
}

size_t ArrayType::getSize() const {
  if (auto intAttr = getSizeAttr().dyn_cast<IntegerAttr>())
    return intAttr.getInt();
  return -1;
}

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type innerType, Attribute size) {
  if (hasHWInOutType(innerType))
    return emitError() << "hw.array cannot contain InOut types";
  return success();
}

//===----------------------------------------------------------------------===//
// UnpackedArrayType
//===----------------------------------------------------------------------===//

Type UnpackedArrayType::parse(AsmParser &p) {
  Attribute dim;
  Type inner;

  if (failed(parseArray(p, dim, inner)))
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner, dim)))
    return Type();

  return get(inner.getContext(), inner, dim);
}

void UnpackedArrayType::print(AsmPrinter &p) const {
  p << "<";
  p.printAttributeWithoutType(getSizeAttr());
  p << "x";
  printHWElementType(getElementType(), p);
  p << '>';
}

LogicalResult
UnpackedArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                          Type innerType, Attribute size) {
  if (!isHWValueType(innerType))
    return emitError() << "invalid element for uarray type";
  return success();
}

size_t UnpackedArrayType::getSize() const {
  return getSizeAttr().cast<IntegerAttr>().getInt();
}

//===----------------------------------------------------------------------===//
// InOutType
//===----------------------------------------------------------------------===//

Type InOutType::parse(AsmParser &p) {
  Type inner;
  if (p.parseLess() || parseHWElementType(inner, p) || p.parseGreater())
    return Type();

  auto loc = p.getEncodedSourceLoc(p.getCurrentLocation());
  if (failed(verify(mlir::detail::getDefaultDiagnosticEmitFn(loc), inner)))
    return Type();

  return get(p.getContext(), inner);
}

void InOutType::print(AsmPrinter &p) const {
  p << "<";
  printHWElementType(getElementType(), p);
  p << '>';
}

LogicalResult InOutType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type innerType) {
  if (!isHWValueType(innerType))
    return emitError() << "invalid element for hw.inout type " << innerType;
  return success();
}

//===----------------------------------------------------------------------===//
// TypeAliasType
//===----------------------------------------------------------------------===//

static Type computeCanonicalType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case([](TypeAliasType t) {
        return computeCanonicalType(t.getCanonicalType());
      })
      .Case([](ArrayType t) {
        return ArrayType::get(computeCanonicalType(t.getElementType()),
                              t.getSize());
      })
      .Case([](UnpackedArrayType t) {
        return UnpackedArrayType::get(computeCanonicalType(t.getElementType()),
                                      t.getSize());
      })
      .Case([](StructType t) {
        SmallVector<StructType::FieldInfo> fieldInfo;
        for (auto field : t.getElements())
          fieldInfo.push_back(StructType::FieldInfo{
              field.name, computeCanonicalType(field.type)});
        return StructType::get(t.getContext(), fieldInfo);
      })
      .Default([](Type t) { return t; });
}

TypeAliasType TypeAliasType::get(SymbolRefAttr ref, Type innerType) {
  return get(ref.getContext(), ref, innerType, computeCanonicalType(innerType));
}

Type TypeAliasType::parse(AsmParser &p) {
  SymbolRefAttr ref;
  Type type;
  if (p.parseLess() || p.parseAttribute(ref) || p.parseComma() ||
      p.parseType(type) || p.parseGreater())
    return Type();

  return get(ref, type);
}

void TypeAliasType::print(AsmPrinter &p) const {
  p << "<" << getRef() << ", " << getInnerType() << ">";
}

/// Return the Typedecl referenced by this TypeAlias, given the module to look
/// in.  This returns null when the IR is malformed.
TypedeclOp TypeAliasType::getTypeDecl(const HWSymbolCache &cache) {
  SymbolRefAttr ref = getRef();
  auto typeScope = ::dyn_cast_or_null<TypeScopeOp>(
      cache.getDefinition(ref.getRootReference()));
  if (!typeScope)
    return {};

  return typeScope.lookupSymbol<TypedeclOp>(ref.getLeafReference());
}

////////////////////////////////////////////////////////////////////////////////
// ModuleType
////////////////////////////////////////////////////////////////////////////////

LogicalResult ModuleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<ModulePort> ports) {
  if (llvm::any_of(ports, [](const ModulePort &port) {
        return hasHWInOutType(port.type);
      }))
    return emitError() << "Ports cannot be inout types";
  return success();
}

size_t ModuleType::getNumInputs() { return getInputTypes().size(); }

size_t ModuleType::getNumOutputs() { return getOutputTypes().size(); }

SmallVector<Type> ModuleType::getInputTypes() {
  SmallVector<Type> retval;
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Input ||
        p.dir == ModulePort::Direction::InOut) {
      retval.push_back(p.type);
    }
  }
  return retval;
}

SmallVector<Type> ModuleType::getOutputTypes() {
  SmallVector<Type> retval;
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Output) {
      retval.push_back(p.type);
    }
  }
  return retval;
}

Type ModuleType::getInputType(size_t idx) {
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Input ||
        p.dir == ModulePort::Direction::InOut) {
      if (!idx)
        return p.type;
      --idx;
    }
  }
  // Tolerate Malformed IR for debug printing
  return {};
}

Type ModuleType::getOutputType(size_t idx) {
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Output) {
      if (!idx)
        return p.type;
      --idx;
    }
  }
  // Tolerate Malformed IR for debug printing
  return {};
}

SmallVector<StringAttr> ModuleType::getInputNames() {
  SmallVector<StringAttr> retval;
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Input ||
        p.dir == ModulePort::Direction::InOut) {
      retval.push_back(p.name);
    }
  }
  return retval;
}

SmallVector<StringAttr> ModuleType::getOutputNames() {
  SmallVector<StringAttr> retval;
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Output) {
      retval.push_back(p.name);
    }
  }
  return retval;
}

StringAttr ModuleType::getNameAttr(size_t idx) { return getPorts()[idx].name; }

StringRef ModuleType::getName(size_t idx) {
  auto sa = getNameAttr(idx);
  if (sa)
    return sa.getValue();
  return {};
}

StringAttr ModuleType::getInputNameAttr(size_t idx) {
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Input ||
        p.dir == ModulePort::Direction::InOut) {
      if (!idx)
        return p.name;
      --idx;
    }
  }
  // Tolerate Malformed IR for debug printing
  return {};
}

StringRef ModuleType::getInputName(size_t idx) {
  auto sa = getInputNameAttr(idx);
  if (sa)
    return sa.getValue();
  return {};
}

StringAttr ModuleType::getOutputNameAttr(size_t idx) {
  for (auto &p : getPorts()) {
    if (p.dir == ModulePort::Direction::Output) {
      if (!idx)
        return p.name;
      --idx;
    }
  }
  // Tolerate Malformed IR for debug printing
  return {};
}

StringRef ModuleType::getOutputName(size_t idx) {
  auto sa = getOutputNameAttr(idx);
  if (sa)
    return sa.getValue();
  return {};
}

FunctionType ModuleType::getFuncType() {
  return FunctionType::get(getContext(), getInputTypes(), getOutputTypes());
}

namespace mlir {
template <>
struct FieldParser<circt::hw::ModulePort> {
  static FailureOr<circt::hw::ModulePort> parse(AsmParser &parser) {
    StringRef dir, name;
    Type type;
    if (parser.parseKeyword(&dir) || parser.parseKeyword(&name) ||
        parser.parseColon() || parser.parseType(type))
      return failure();
    circt::hw::ModulePort::Direction d;
    if (dir == "input")
      d = circt::hw::ModulePort::Input;
    else if (dir == "output")
      d = circt::hw::ModulePort::Output;
    else if (dir == "inout")
      d = circt::hw::ModulePort::InOut;
    else
      return failure();
    return circt::hw::ModulePort{parser.getBuilder().getStringAttr(name), type,
                                 d};
  }
};
} // namespace mlir

namespace circt {
namespace hw {

static raw_ostream &operator<<(raw_ostream &printer, ModulePort port) {
  StringRef dirstr;
  switch (port.dir) {
  case ModulePort::Direction::Input:
    dirstr = "input";
    break;
  case ModulePort::Direction::Output:
    dirstr = "output";
    break;
  case ModulePort::Direction::InOut:
    dirstr = "inout";
    break;
  default:
    assert(0 && "unknown direction");
    dirstr = "unknown";
    break;
  }
  printer << dirstr << " " << port.name << " : " << port.type;
  return printer;
}
static bool operator==(const ModulePort &a, const ModulePort &b) {
  return a.dir == b.dir && a.name == b.name && a.type == b.type;
}
static llvm::hash_code hash_value(const ModulePort &port) {
  return llvm::hash_combine(port.dir, port.name, port.type);
}
} // namespace hw
} // namespace circt

ModuleType circt::hw::detail::fnToMod(Operation *op, ArrayAttr inputNames,
                                      ArrayAttr outputNames) {
  return fnToMod(
      cast<FunctionType>(cast<mlir::FunctionOpInterface>(op).getFunctionType()),
      inputNames, outputNames);
}

ModuleType circt::hw::detail::fnToMod(FunctionType fnty, ArrayAttr inputNames,
                                      ArrayAttr outputNames) {
  SmallVector<ModulePort> ports;
  for (auto [t, n] : llvm::zip(fnty.getInputs(), inputNames))
    if (auto iot = dyn_cast<hw::InOutType>(t))
      ports.push_back({cast<StringAttr>(n), iot.getElementType(),
                       ModulePort::Direction::InOut});
    else
      ports.push_back({cast<StringAttr>(n), t, ModulePort::Direction::Input});
  for (auto [t, n] : llvm::zip(fnty.getResults(), outputNames))
    ports.push_back({cast<StringAttr>(n), t, ModulePort::Direction::Output});
  return ModuleType::get(fnty.getContext(), ports);
}

////////////////////////////////////////////////////////////////////////////////
// BoilerPlate
////////////////////////////////////////////////////////////////////////////////

void HWDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/HW/HWTypes.cpp.inc"
      >();
}
