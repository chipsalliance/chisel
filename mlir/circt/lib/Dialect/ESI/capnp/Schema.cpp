//===- Schema.cpp - ESI Cap'nProto schema utilities -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "ESICapnp.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Support/IndentedOstream.h"

// NOLINTNEXTLINE(clang-diagnostic-error)
#include "capnp/schema-parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"

#include <initializer_list>
#include <string>

using namespace circt::esi::capnp::detail;
using namespace circt;
using circt::comb::ICmpPredicate;

namespace {
struct GasketComponent;
struct Slice;
} // anonymous namespace

//===----------------------------------------------------------------------===//
// CapnpTypeSchema class implementation.
//===----------------------------------------------------------------------===//

namespace circt {
namespace esi {
namespace capnp {

namespace detail {
/// Actual implementation of `CapnpTypeSchema` to keep all the details out of
/// the header.
struct CapnpTypeSchemaImpl {
public:
  CapnpTypeSchemaImpl(CapnpTypeSchema &base);
  CapnpTypeSchemaImpl(const CapnpTypeSchemaImpl &) = delete;
  LogicalResult write(llvm::raw_ostream &os) const;

  bool operator==(const CapnpTypeSchemaImpl &) const;

  // Compute the expected size of the capnp message in bits.
  size_t size() const;

  /// Build an HW/SV dialect capnp encoder for this type.
  hw::HWModuleOp buildEncoder(Value clk, Value valid, Value);
  /// Build an HW/SV dialect capnp decoder for this type.
  hw::HWModuleOp buildDecoder(Value clk, Value valid, Value);

private:
  ::capnp::ParsedSchema getSchema() const;
  ::capnp::StructSchema getCapnpTypeSchema() const;

  CapnpTypeSchema &base;

  ::capnp::SchemaParser parser;
  mutable ::capnp::ParsedSchema rootSchema;
  mutable ::capnp::StructSchema typeSchema;
};
} // namespace detail
} // namespace capnp
} // namespace esi
} // namespace circt

/// Return the encoding value for the size of this type (from the encoding
/// spec): 0 = 0 bits, 1 = 1 bit, 2 = 1 byte, 3 = 2 bytes, 4 = 4 bytes, 5 = 8
/// bytes (non-pointer), 6 = 8 bytes (pointer).
static size_t bitsEncoding(::capnp::schema::Type::Reader type) {
  using ty = ::capnp::schema::Type;
  switch (type.which()) {
  case ty::VOID:
    return 0;
  case ty::BOOL:
    return 1;
  case ty::UINT8:
  case ty::INT8:
    return 2;
  case ty::UINT16:
  case ty::INT16:
    return 3;
  case ty::UINT32:
  case ty::INT32:
    return 4;
  case ty::UINT64:
  case ty::INT64:
    return 5;
  case ty::ANY_POINTER:
  case ty::DATA:
  case ty::INTERFACE:
  case ty::LIST:
  case ty::STRUCT:
  case ty::TEXT:
    return 6;
  default:
    llvm_unreachable("Type not yet supported");
  }
}

/// Return the number of bits used by a Capnp type.
static size_t bits(::capnp::schema::Type::Reader type) {
  size_t enc = bitsEncoding(type);
  if (enc <= 1)
    return enc;
  if (enc == 6)
    return 64;
  return 1 << (enc + 1);
}

/// Return true if 'type' is capnp pointer.
static bool isPointerType(::capnp::schema::Type::Reader type) {
  using ty = ::capnp::schema::Type;
  switch (type.which()) {
  case ty::ANY_POINTER:
  case ty::DATA:
  case ty::INTERFACE:
  case ty::LIST:
  case ty::STRUCT:
  case ty::TEXT:
    return true;
  default:
    return false;
  }
}

CapnpTypeSchemaImpl::CapnpTypeSchemaImpl(CapnpTypeSchema &base) : base(base) {}

/// Write a valid capnp schema to memory, then parse it out of memory using the
/// capnp library. Writing and parsing text within a single process is ugly, but
/// this is by far the easiest way to do this. This isn't the use case for which
/// Cap'nProto was designed.
::capnp::ParsedSchema CapnpTypeSchemaImpl::getSchema() const {
  if (rootSchema != ::capnp::ParsedSchema())
    return rootSchema;

  // Write the schema to `schemaText`.
  std::string schemaText;
  llvm::raw_string_ostream os(schemaText);
  emitCapnpID(os, 0xFFFFFFFFFFFFFFFF) << ";\n";
  auto rc = write(os);
  assert(succeeded(rc) && "Failed schema text output.");
  (void)rc;
  os.str();

  // Write `schemaText` to an in-memory filesystem then parse it. Yes, this is
  // the only way to do this.
  kj::Own<kj::Filesystem> fs = kj::newDiskFilesystem();
  kj::Own<kj::Directory> dir = kj::newInMemoryDirectory(kj::nullClock());
  kj::Path fakePath = kj::Path::parse("schema.capnp");
  { // Ensure that 'fakeFile' has flushed.
    auto fakeFile = dir->openFile(fakePath, kj::WriteMode::CREATE);
    fakeFile->writeAll(schemaText);
  }
  rootSchema = parser.parseFromDirectory(*dir, std::move(fakePath), nullptr);
  return rootSchema;
}

/// Find the schema corresponding to `type` and return it.
::capnp::StructSchema CapnpTypeSchemaImpl::getCapnpTypeSchema() const {
  if (typeSchema != ::capnp::StructSchema())
    return typeSchema;
  uint64_t id = base.typeID();
  for (auto schemaNode : getSchema().getAllNested()) {
    if (schemaNode.getProto().getId() == id) {
      typeSchema = schemaNode.asStruct();
      return typeSchema;
    }
  }
  llvm_unreachable("A node with a matching ID should always be found.");
}

/// Returns the expected size of an array (capnp list) in 64-bit words.
static int64_t size(hw::ArrayType mType, capnp::schema::Field::Reader cField) {
  assert(cField.isSlot());
  auto cType = cField.getSlot().getType();
  assert(cType.isList());
  size_t elementBits = bits(cType.getList().getElementType());
  int64_t listBits = mType.getSize() * elementBits;
  return llvm::divideCeil(listBits, 64);
}

/// Compute the size of a capnp struct, in 64-bit words.
static int64_t size(capnp::schema::Node::Struct::Reader cStruct,
                    ArrayRef<hw::StructType::FieldInfo> mFields) {
  using namespace capnp::schema;
  int64_t size = (1 + // Header
                  cStruct.getDataWordCount() + cStruct.getPointerCount());
  auto cFields = cStruct.getFields();
  for (Field::Reader cField : cFields) {
    assert(!cField.isGroup() && "Capnp groups are not supported");
    // Capnp code order is the index in the MLIR fields array.
    assert(cField.getCodeOrder() < mFields.size());

    // The size of the thing to which the pointer is pointing, not the size of
    // the pointer itself.
    int64_t pointedToSize =
        TypeSwitch<mlir::Type, int64_t>(mFields[cField.getCodeOrder()].type)
            .Case([](IntegerType) { return 0; })
            .Case([cField](hw::ArrayType mType) {
              return ::size(mType, cField);
            });
    size += pointedToSize;
  }
  return size; // Convert from 64-bit words to bits.
}

// Compute the expected size of the capnp message in bits.
size_t CapnpTypeSchemaImpl::size() const {
  auto schema = getCapnpTypeSchema();
  auto structProto = schema.getProto().getStruct();
  return ::size(structProto, base.getFields()) * 64;
}

/// Write a valid Capnp type.
static void emitCapnpType(Type type, llvm::raw_ostream &os) {
  llvm::TypeSwitch<Type>(type)
      .Case([&os](IntegerType intTy) {
        auto w = intTy.getWidth();
        if (w == 0) {
          os << "Void";
        } else if (w == 1) {
          os << "Bool";
        } else {
          if (intTy.isSigned())
            os << "Int";
          else
            os << "UInt";

          // Round up.
          if (w <= 8)
            os << "8";
          else if (w <= 16)
            os << "16";
          else if (w <= 32)
            os << "32";
          else if (w <= 64)
            os << "64";
          else
            assert(false && "Type not supported. Integer too wide. Please "
                            "check support first with isSupported()");
        }
      })
      .Case([&os](hw::ArrayType arrTy) {
        os << "List(";
        emitCapnpType(arrTy.getElementType(), os);
        os << ')';
      })
      .Case([](hw::StructType structTy) {
        assert(false && "Struct containing structs not supported");
      })
      .Default([](Type) {
        assert(false && "Type not supported. Please check support first with "
                        "isSupported()");
      });
}

/// This function is essentially a placeholder which only supports ints. It'll
/// need to be re-worked when we start supporting structs, arrays, unions,
/// enums, etc.
LogicalResult CapnpTypeSchemaImpl::write(llvm::raw_ostream &rawOS) const {
  mlir::raw_indented_ostream os(rawOS);

  // Since capnp requires messages to be structs, emit a wrapper struct.
  os << "struct ";
  base.writeMetadata(rawOS);
  os << " {\n";
  os.indent();

  size_t counter = 0;
  size_t maxNameLength = 0;
  for (auto field : base.getFields())
    maxNameLength = std::max(maxNameLength, field.name.size());

  for (auto field : base.getFields()) {
    // Specify the actual type, followed by the capnp field.
    os << field.name.getValue();
    std::string padding = std::string(maxNameLength - field.name.size(), ' ');
    os << padding << " @" << counter++ << " :";
    emitCapnpType(field.type, os.getOStream());
    os << ";  # Actual type is " << field.type << ".\n";
  }

  os.unindent();
  os << "}\n\n";
  return success();
}

//===----------------------------------------------------------------------===//
// Helper classes for common operations in the encode / decoders
//===----------------------------------------------------------------------===//

namespace {
/// Provides easy methods to build common operations.
struct GasketBuilder {
public:
  GasketBuilder() {} // To satisfy containers.
  GasketBuilder(OpBuilder &b, Location loc) : builder(&b), location(loc) {}

  /// Get a zero constant of 'width' bit width.
  GasketComponent zero(uint64_t width) const;
  /// Get a constant 'value' of a certain bit width.
  GasketComponent constant(uint64_t width, uint64_t value) const;

  /// Get 'p' bits of i1 padding.
  Slice padding(uint64_t p) const;

  Location loc() const { return *location; }
  void setLoc(Location loc) { location = loc; }
  OpBuilder &b() const { return *builder; }
  MLIRContext *ctxt() const { return builder->getContext(); }

protected:
  OpBuilder *builder;
  std::optional<Location> location;
};
} // anonymous namespace

namespace {
/// Contains helper methods to assist with naming and casting.
struct GasketComponent : GasketBuilder {
public:
  GasketComponent() {} // To satisfy containers.
  GasketComponent(OpBuilder &b, Value init)
      : GasketBuilder(b, init.getLoc()), s(init) {}
  GasketComponent(std::initializer_list<GasketComponent> values) {
    *this = GasketComponent::concat(values);
  }

  /// Set the "name" attribute of a value's op.
  template <typename T = GasketComponent>
  T &name(const Twine &name) {
    std::string nameStr = name.str();
    if (nameStr.empty())
      return *(T *)this;
    auto nameAttr = StringAttr::get(ctxt(), nameStr);
    s.getDefiningOp()->setAttr("name", nameAttr);
    return *(T *)this;
  }
  template <typename T = GasketComponent>
  T &name(capnp::Text::Reader fieldName, const Twine &nameSuffix) {
    return name<T>(StringRef(fieldName.cStr()) + nameSuffix);
  }

  StringRef getName() const {
    auto nameAttr = s.getDefiningOp()->getAttrOfType<StringAttr>("name");
    if (nameAttr)
      return nameAttr.getValue();
    return StringRef();
  }

  /// Construct a bitcast.
  GasketComponent cast(Type t) const {
    auto dst = builder->create<hw::BitcastOp>(loc(), t, s);
    auto gc = GasketComponent(*builder, dst);
    StringRef name = getName();
    if (name.empty())
      return gc;
    return gc.name(name + "_casted");
    ;
  }

  /// Construct a bitcast.
  Slice castBitArray() const;

  /// Downcast an int, accounting for signedness.
  GasketComponent downcast(IntegerType t) const {
    // Since the HW dialect operators only operate on signless integers, we
    // have to cast to signless first, then cast the sign back.
    assert(s.getType().isa<IntegerType>());
    Value signlessVal = s;
    if (!signlessVal.getType().isSignlessInteger())
      signlessVal = builder->create<hw::BitcastOp>(
          loc(), builder->getIntegerType(s.getType().getIntOrFloatBitWidth()),
          s);

    if (!t.isSigned()) {
      auto extracted =
          builder->create<comb::ExtractOp>(loc(), t, signlessVal, 0);
      return GasketComponent(*builder, extracted).cast(t);
    }
    auto magnitude = builder->create<comb::ExtractOp>(
        loc(), builder->getIntegerType(t.getWidth() - 1), signlessVal, 0);
    auto sign = builder->create<comb::ExtractOp>(
        loc(), builder->getIntegerType(1), signlessVal, t.getWidth() - 1);
    auto result = builder->create<comb::ConcatOp>(loc(), sign, magnitude);

    // We still have to cast to handle signedness.
    return GasketComponent(*builder, result).cast(t);
  }

  /// Pad this value with zeros up to `finalBits`.
  GasketComponent padTo(uint64_t finalBits) const;

  /// Returns the bit width of this value.
  uint64_t size() const { return hw::getBitWidth(s.getType()); }

  /// Build a component by concatenating some values.
  static GasketComponent concat(ArrayRef<GasketComponent> concatValues);

  bool operator==(const GasketComponent &that) { return this->s == that.s; }
  bool operator!=(const GasketComponent &that) { return this->s != that.s; }
  Operation *operator->() const { return s.getDefiningOp(); }
  Value getValue() const { return s; }
  Type getType() const { return s.getType(); }
  operator Value() { return s; }

protected:
  Value s;
};
} // anonymous namespace

namespace {
/// Holds a 'slice' of an array and is able to construct more slice ops, then
/// cast to a type. A sub-slice holds a pointer to the slice which created it,
/// so it forms a hierarchy. This is so we can easily track offsets from the
/// root message for pointer resolution.
///
/// Requirement: any slice which has sub-slices must not be free'd before its
/// children slices.
struct Slice : public GasketComponent {
private:
  Slice(const Slice *parent, std::optional<int64_t> offset, Value val)
      : GasketComponent(*parent->builder, val), parent(parent),
        offsetIntoParent(offset) {
    type = val.getType().dyn_cast<hw::ArrayType>();
    assert(type && "Value must be array type");
  }

public:
  Slice(OpBuilder &b, Value val)
      : GasketComponent(b, val), parent(nullptr), offsetIntoParent(0) {
    type = val.getType().dyn_cast<hw::ArrayType>();
    assert(type && "Value must be array type");
  }
  Slice(GasketComponent gc)
      : GasketComponent(gc), parent(nullptr), offsetIntoParent(0) {
    type = gc.getValue().getType().dyn_cast<hw::ArrayType>();
    assert(type && "Value must be array type");
  }

  /// Create an op to slice the array from lsb to lsb + size. Return a new slice
  /// with that op.
  Slice slice(int64_t lsb, int64_t size) const {
    hw::ArrayType dstTy = hw::ArrayType::get(type.getElementType(), size);
    IntegerType idxTy =
        builder->getIntegerType(llvm::Log2_64_Ceil(type.getSize()));
    Value lsbConst = builder->create<hw::ConstantOp>(loc(), idxTy, lsb);
    Value newSlice =
        builder->create<hw::ArraySliceOp>(loc(), dstTy, s, lsbConst);
    return Slice(this, lsb, newSlice);
  }

  /// Create an op to slice the array from lsb to lsb + size. Return a new slice
  /// with that op. If lsb is greater width thn necessary, lop off the high
  /// bits.
  Slice slice(Value lsb, int64_t size) const {
    assert(lsb.getType().isa<IntegerType>());

    unsigned expIdxWidth = llvm::Log2_64_Ceil(type.getSize());
    int64_t lsbWidth = lsb.getType().getIntOrFloatBitWidth();
    if (lsbWidth > expIdxWidth)
      lsb = builder->create<comb::ExtractOp>(
          loc(), builder->getIntegerType(expIdxWidth), lsb, 0);
    else if (lsbWidth < expIdxWidth)
      assert(false && "LSB Value must not be smaller than expected.");
    auto dstTy = hw::ArrayType::get(type.getElementType(), size);
    Value newSlice = builder->create<hw::ArraySliceOp>(loc(), dstTy, s, lsb);
    return Slice(this, std::nullopt, newSlice);
  }
  Slice &name(const Twine &name) { return GasketComponent::name<Slice>(name); }
  Slice &name(capnp::Text::Reader fieldName, const Twine &nameSuffix) {
    return GasketComponent::name<Slice>(fieldName.cStr(), nameSuffix);
  }
  Slice castToSlice(Type elemTy, size_t size, StringRef name = StringRef(),
                    Twine nameSuffix = Twine()) const {
    auto arrTy = hw::ArrayType::get(elemTy, size);
    GasketComponent rawCast =
        GasketComponent::cast(arrTy).name(name + nameSuffix);
    return Slice(*builder, rawCast);
  }

  GasketComponent operator[](Value idx) const {
    return GasketComponent(*builder,
                           builder->create<hw::ArrayGetOp>(loc(), s, idx));
  }

  GasketComponent operator[](size_t idx) const {
    IntegerType idxTy =
        builder->getIntegerType(llvm::Log2_32_Ceil(type.getSize()));
    auto idxVal = builder->create<hw::ConstantOp>(loc(), idxTy, idx);
    return GasketComponent(*builder,
                           builder->create<hw::ArrayGetOp>(loc(), s, idxVal));
  }

  /// Return the root of this slice hierarchy.
  const Slice &getRootSlice() const {
    if (parent == nullptr)
      return *this;
    return parent->getRootSlice();
  }

  std::optional<int64_t> getOffsetFromRoot() const {
    if (parent == nullptr)
      return 0;
    auto parentOffset = parent->getOffsetFromRoot();
    if (!offsetIntoParent || !parentOffset)
      return std::nullopt;
    return *offsetIntoParent + *parentOffset;
  }

  uint64_t size() const { return type.getSize(); }

private:
  hw::ArrayType type;
  const Slice *parent;
  std::optional<int64_t> offsetIntoParent;
};
} // anonymous namespace

// The following methods have to be defined out-of-line because they use types
// which aren't yet defined when they are declared.

GasketComponent GasketBuilder::zero(uint64_t width) const {
  return GasketComponent(*builder,
                         builder->create<hw::ConstantOp>(
                             loc(), builder->getIntegerType(width), 0));
}
GasketComponent GasketBuilder::constant(uint64_t width, uint64_t value) const {
  return GasketComponent(*builder,
                         builder->create<hw::ConstantOp>(
                             loc(), builder->getIntegerType(width), value));
}

Slice GasketBuilder::padding(uint64_t p) const {
  auto zero = GasketBuilder::zero(p);
  return zero.castBitArray();
}

Slice GasketComponent::castBitArray() const {
  auto dstTy =
      hw::ArrayType::get(builder->getI1Type(), hw::getBitWidth(s.getType()));
  if (s.getType() == dstTy)
    return Slice(*builder, s);
  auto dst = builder->create<hw::BitcastOp>(loc(), dstTy, s);
  return Slice(*builder, dst);
}

GasketComponent
GasketComponent::concat(ArrayRef<GasketComponent> concatValues) {
  assert(concatValues.size() > 0);
  auto builder = concatValues[0].builder;
  auto loc = concatValues[0].loc();
  SmallVector<Value, 8> values;
  for (auto gc : concatValues) {
    values.push_back(gc.castBitArray());
  }
  // Since the "endianness" of `values` is the reverse of ArrayConcat, we must
  // reverse ourselves.
  std::reverse(values.begin(), values.end());
  return GasketComponent(*builder,
                         builder->create<hw::ArrayConcatOp>(loc, values));
}
namespace {
/// Utility class for building sv::AssertOps. Since SV assertions need to be in
/// an `always` block (so the simulator knows when to check the assertion), we
/// build them all in a region intended for assertions.
class AssertBuilder : public OpBuilder {
public:
  AssertBuilder(Location loc, Region &r) : OpBuilder(r), loc(loc) {}

  void assertPred(GasketComponent veg, ICmpPredicate pred, int64_t expected) {
    if (veg.getValue().getType().isa<IntegerType>()) {
      assertPred(veg.getValue(), pred, expected);
      return;
    }

    auto valTy = veg.getValue().getType().dyn_cast<hw::ArrayType>();
    assert(valTy && valTy.getElementType() == veg.b().getIntegerType(1) &&
           "Can only compare ints and bit arrays");
    assertPred(veg.cast(veg.b().getIntegerType(valTy.getSize())).getValue(),
               pred, expected);
  }

  void assertEqual(GasketComponent s, int64_t expected) {
    assertPred(s, ICmpPredicate::eq, expected);
  }

private:
  void assertPred(Value val, ICmpPredicate pred, int64_t expected) {
    auto expectedVal = create<hw::ConstantOp>(loc, val.getType(), expected);
    create<sv::AssertOp>(
        loc,
        create<comb::ICmpOp>(loc, getI1Type(), pred, val, expectedVal, false),
        sv::DeferAssertAttr::get(loc.getContext(), sv::DeferAssert::Immediate));
  }
  Location loc;
};
} // anonymous namespace

//===----------------------------------------------------------------------===//
// Capnp encode "gasket" HW builders.
//
// These have the potential to get large and complex as we add more types. The
// encoding spec is here: https://capnproto.org/encoding.html
//===----------------------------------------------------------------------===//

namespace {
/// Helps build capnp message DAGs, which are stored in 'segments'. To better
/// reason about something which is more memory-like than wire-like, this class
/// contains a data structure to efficiently model memory and map it to Values
/// (wires).
class CapnpSegmentBuilder : public GasketBuilder {
public:
  CapnpSegmentBuilder(OpBuilder &b, Location loc, uint64_t expectedSize)
      : GasketBuilder(b, loc), segmentValues(allocator), messageSize(0),
        expectedSize(expectedSize) {}
  CapnpSegmentBuilder(const CapnpSegmentBuilder &) = delete;
  ~CapnpSegmentBuilder() {}

  GasketComponent build(::capnp::schema::Node::Struct::Reader cStruct,
                        ArrayRef<GasketComponent> mFieldValues);

private:
  /// Allocate and build a struct. Return the address of the data section as an
  /// offset into the 'memory' map.
  GasketComponent encodeStructAt(uint64_t ptrLoc,
                                 ::capnp::schema::Node::Struct::Reader cStruct,
                                 ArrayRef<GasketComponent> mFieldValues);
  /// Build a value from the 'memory' map. Concatenates all the values in the
  /// 'memory' map, filling in the blank addresses with padding.
  GasketComponent compile() const;

  /// Encode 'val' and place the value at the specified 'memory' offset.
  void encodeFieldAt(uint64_t offset, GasketComponent val,
                     ::capnp::schema::Type::Reader type);
  /// Allocate and build a list, returning the address which was allocated.
  uint64_t buildList(Slice val, ::capnp::schema::Type::Reader type);

  /// Insert 'val' into the 'memory' map.
  void insert(uint64_t offset, GasketComponent val) {
    uint64_t valSize = val.size();
    assert(!segmentValues.overlaps(offset, offset + valSize - 1));
    assert(offset + valSize - 1 < expectedSize &&
           "Tried to insert above the max expected size of the message.");
    segmentValues.insert(offset, offset + valSize - 1, val);
  }

  /// This is where the magic lives. An IntervalMap allows us to efficiently
  /// model segment 'memory' and to place Values at any address. We can then
  /// manage 'memory allocations' (figuring out where to place pointed to
  /// objects) separately from the data contained in those values, some of which
  /// are pointers themselves.
  llvm::IntervalMap<uint64_t, GasketComponent>::Allocator allocator;
  llvm::IntervalMap<uint64_t, GasketComponent> segmentValues;

  /// Track the allocated message size. Increase to 'alloc' more.
  uint64_t messageSize;
  uint64_t alloc(size_t bits) {
    uint64_t ptr = messageSize;
    messageSize += bits;
    return ptr;
  }

  /// The expected maximum size of the message.
  uint64_t expectedSize;
};
} // anonymous namespace

void CapnpSegmentBuilder::encodeFieldAt(uint64_t offset, GasketComponent val,
                                        ::capnp::schema::Type::Reader type) {
  TypeSwitch<Type>(val.getValue().getType())
      .Case([&](IntegerType it) { insert(offset, val); })
      .Case([&](hw::ArrayType arrTy) {
        uint64_t listOffset = buildList(Slice(val), type);
        int32_t relativeOffset = (listOffset - offset - 64) / 64;
        insert(offset,
               GasketComponent::concat(
                   {constant(2, 1), constant(30, relativeOffset),
                    constant(3, bitsEncoding(type.getList().getElementType())),
                    constant(29, arrTy.getSize())}));
      });
}

uint64_t CapnpSegmentBuilder::buildList(Slice val,
                                        ::capnp::schema::Type::Reader type) {
  hw::ArrayType arrTy = val.getValue().getType().cast<hw::ArrayType>();
  auto elemType = type.getList().getElementType();
  size_t elemWidth = bits(elemType);
  uint64_t listSize = elemWidth * arrTy.getSize();
  uint64_t m;
  if ((m = listSize % 64) != 0)
    listSize += (64 - m);
  uint64_t listOffset = alloc(listSize);

  for (size_t i = 0, e = arrTy.getSize(); i < e; ++i) {
    size_t elemNum = e - i - 1;
    encodeFieldAt(listOffset + (elemNum * elemWidth), val[i], elemType);
  }
  return listOffset;
}

GasketComponent CapnpSegmentBuilder::encodeStructAt(
    uint64_t ptrLoc, ::capnp::schema::Node::Struct::Reader cStruct,
    ArrayRef<GasketComponent> mFieldValues) {

  assert(ptrLoc % 64 == 0);
  size_t structSize =
      (cStruct.getDataWordCount() + cStruct.getPointerCount()) * 64;
  uint64_t structDataSectionOffset = alloc(structSize);
  uint64_t structPointerSectionOffset =
      structDataSectionOffset + (cStruct.getDataWordCount() * 64);
  assert(structDataSectionOffset % 64 == 0);
  int64_t relativeStructDataOffsetWords =
      ((structDataSectionOffset - ptrLoc) / 64) -
      /*offset from end of pointer.*/ 1;
  GasketComponent structPtr = {constant(2, 0),
                               constant(30, relativeStructDataOffsetWords),
                               constant(16, cStruct.getDataWordCount()),
                               constant(16, cStruct.getPointerCount())};

  // Loop through data fields.
  for (auto field : cStruct.getFields()) {
    uint16_t idx = field.getCodeOrder();
    assert(idx < mFieldValues.size() &&
           "Capnp struct longer than fieldValues.");
    auto cFieldType = field.getSlot().getType();
    uint64_t fieldOffset =
        (isPointerType(cFieldType) ? structPointerSectionOffset
                                   : structDataSectionOffset) +
        field.getSlot().getOffset() * bits(cFieldType);
    encodeFieldAt(fieldOffset, mFieldValues[idx], cFieldType);
  }

  return structPtr;
}

GasketComponent CapnpSegmentBuilder::compile() const {
  // Fill in missing bits.
  SmallVector<GasketComponent, 16> segmentValuesPlusPadding;
  uint64_t lastStop = 0;
  for (auto it = segmentValues.begin(), e = segmentValues.end(); it != e;
       ++it) {
    auto value = it.value();
    int64_t padBits = it.start() - lastStop;
    assert(padBits >= 0 && "Overlap not allowed");
    if (padBits)
      segmentValuesPlusPadding.push_back(padding(padBits));
    segmentValuesPlusPadding.push_back(value.castBitArray());
    // IntervalMap has inclusive ranges, but we want to reason about [,) regions
    // to make the math work.
    lastStop = it.stop() + 1;
  }
  assert(expectedSize >= lastStop);
  if (lastStop != expectedSize)
    segmentValuesPlusPadding.push_back(padding(expectedSize - lastStop));

  return GasketComponent::concat(segmentValuesPlusPadding);
}

GasketComponent
CapnpSegmentBuilder::build(::capnp::schema::Node::Struct::Reader cStruct,
                           ArrayRef<GasketComponent> mFieldValues) {
  uint64_t rootPtrLoc = alloc(64);
  assert(rootPtrLoc == 0);
  auto rootPtr = encodeStructAt(rootPtrLoc, cStruct, mFieldValues);
  insert(rootPtrLoc, rootPtr);
  return compile();
}

/// Build an HW/SV dialect capnp encoder module for this type. Inputs need to
/// be packed and unpadded.
hw::HWModuleOp CapnpTypeSchemaImpl::buildEncoder(Value clk, Value valid,
                                                 Value operandVal) {
  Location loc = operandVal.getDefiningOp()->getLoc();
  ModuleOp topMod = operandVal.getDefiningOp()->getParentOfType<ModuleOp>();
  OpBuilder b = OpBuilder::atBlockEnd(topMod.getBody());

  SmallString<64> modName;
  modName.append("encode");
  modName.append(base.name());
  SmallVector<hw::PortInfo, 4> ports;
  ports.push_back(hw::PortInfo{
      {b.getStringAttr("clk"), clk.getType(), hw::ModulePort::Direction::Input},
      0});
  ports.push_back(hw::PortInfo{{b.getStringAttr("valid"), valid.getType(),
                                hw::ModulePort::Direction::Input},
                               1});
  ports.push_back(
      hw::PortInfo{{b.getStringAttr("unencodedInput"), operandVal.getType(),
                    hw::ModulePort::Direction::Input},
                   2});
  hw::ArrayType modOutputType = hw::ArrayType::get(b.getI1Type(), size());
  ports.push_back(hw::PortInfo{{b.getStringAttr("encoded"), modOutputType,
                                hw::ModulePort::Direction::Output},
                               0});
  hw::HWModuleOp retMod = b.create<hw::HWModuleOp>(
      operandVal.getLoc(), b.getStringAttr(modName), ports);

  Block *innerBlock = retMod.getBodyBlock();
  b.setInsertionPointToStart(innerBlock);
  clk = innerBlock->getArgument(0);
  valid = innerBlock->getArgument(1);
  GasketComponent operand(b, innerBlock->getArgument(2));
  operand.setLoc(loc);

  ::capnp::schema::Node::Reader rootProto = getCapnpTypeSchema().getProto();
  auto st = rootProto.getStruct();
  CapnpSegmentBuilder seg(b, loc, size());

  // The values in the struct we are encoding.
  SmallVector<GasketComponent, 16> fieldValues;
  assert(operand.getValue().getType() == base.getType());
  if (auto structTy = base.getType().dyn_cast<hw::StructType>()) {
    for (auto field : structTy.getElements()) {
      fieldValues.push_back(GasketComponent(
          b, b.create<hw::StructExtractOp>(loc, operand, field)));
    }
  } else {
    fieldValues.push_back(GasketComponent(b, operand));
  }
  GasketComponent ret = seg.build(st, fieldValues);

  innerBlock->getTerminator()->erase();
  b.setInsertionPointToEnd(innerBlock);
  b.create<hw::OutputOp>(loc, ValueRange{ret});
  return retMod;
}

//===----------------------------------------------------------------------===//
// Capnp decode "gasket" HW builders.
//
// These have the potential to get large and complex as we add more types. The
// encoding spec is here: https://capnproto.org/encoding.html
//===----------------------------------------------------------------------===//

/// Construct the proper operations to decode a capnp list. This only works for
/// arrays of ints or bools. Will need to be updated for structs and lists of
/// lists.
static GasketComponent decodeList(hw::ArrayType type,
                                  capnp::schema::Field::Reader field,
                                  Slice ptrSection, AssertBuilder &asserts) {
  capnp::schema::Type::Reader capnpType = field.getSlot().getType();
  assert(capnpType.isList());
  assert(capnpType.getList().hasElementType());

  auto loc = ptrSection.loc();
  OpBuilder &b = ptrSection.b();
  GasketBuilder gb(b, loc);

  // Get the list pointer and break out its parts.
  auto ptr = ptrSection.slice(field.getSlot().getOffset() * 64, 64)
                 .name(field.getName(), "_ptr");
  auto ptrType = ptr.slice(0, 2).name(field.getName(), "_ptrType");
  auto offset = ptr.slice(2, 30)
                    .cast(b.getIntegerType(30))
                    .name(field.getName(), "_offset");
  auto elemSize = ptr.slice(32, 3).name(field.getName(), "_elemSize");
  auto length = ptr.slice(35, 29).name(field.getName(), "_listLength");

  // Assert that ptr type == list type;
  asserts.assertEqual(ptrType, 1);

  // Assert that the element size in the message matches our expectation.
  auto expectedElemSizeBits = bits(capnpType.getList().getElementType());
  unsigned expectedElemSizeField;
  switch (expectedElemSizeBits) {
  case 0:
    expectedElemSizeField = 0;
    break;
  case 1:
    expectedElemSizeField = 1;
    break;
  case 8:
    expectedElemSizeField = 2;
    break;
  case 16:
    expectedElemSizeField = 3;
    break;
  case 32:
    expectedElemSizeField = 4;
    break;
  case 64:
    expectedElemSizeField = 5;
    break;
  default:
    llvm_unreachable("bits() returned unexpected value");
  }
  asserts.assertEqual(elemSize, expectedElemSizeField);

  // Assert that the length of the list (array) is at most the length of the
  // array.
  asserts.assertPred(length, ICmpPredicate::ule, type.getSize());

  // Get the entire message slice, compute the offset into the list, then get
  // the list data in an ArrayType.
  auto msg = ptr.getRootSlice();
  auto ptrOffset = ptr.getOffsetFromRoot();
  assert(ptrOffset);
  GasketComponent offsetInBits(
      b, b.create<comb::ConcatOp>(loc, offset, gb.zero(6)));
  GasketComponent listOffset(
      b, b.create<comb::AddOp>(loc, offsetInBits,
                               gb.constant(36, *ptrOffset + 64), false));
  listOffset.name(field.getName(), "_listOffset");
  auto listSlice =
      msg.slice(listOffset, type.getSize() * expectedElemSizeBits).name("list");

  // Cast to an array of capnp int elements.
  assert(type.getElementType().isa<IntegerType>() &&
         "DecodeList() only works on arrays of ints currently");
  Type capnpElemTy =
      b.getIntegerType(expectedElemSizeBits, IntegerType::Signless);
  auto arrayOfElements = listSlice.castToSlice(capnpElemTy, type.getSize());
  if (arrayOfElements.getValue().getType() == type)
    return arrayOfElements;

  // Collect the reduced elements.
  SmallVector<Value, 64> arrayValues;
  for (size_t i = 0, e = type.getSize(); i < e; ++i) {
    auto capnpElem = arrayOfElements[i].name(field.getName(), "_capnp_elem");
    auto esiElem = capnpElem.downcast(type.getElementType().cast<IntegerType>())
                       .name(field.getName(), "_elem");
    arrayValues.push_back(esiElem);
  }
  auto array = b.create<hw::ArrayCreateOp>(loc, arrayValues);
  return GasketComponent(b, array);
}

/// Construct the proper operations to convert a capnp field to 'type'.
static GasketComponent decodeField(Type type,
                                   capnp::schema::Field::Reader field,
                                   Slice dataSection, Slice ptrSection,
                                   AssertBuilder &asserts) {
  GasketComponent esiValue =
      TypeSwitch<Type, GasketComponent>(type)
          .Case([&](IntegerType it) {
            auto slice = dataSection.slice(field.getSlot().getOffset() *
                                               bits(field.getSlot().getType()),
                                           it.getWidth());
            return slice.name(field.getName(), "_bits").cast(type);
          })
          .Case([&](hw::ArrayType at) {
            return decodeList(at, field, ptrSection, asserts);
          });
  esiValue.name(field.getName().cStr(), "Value");
  return esiValue;
}

/// Build an HW/SV dialect capnp decoder module for this type. Outputs packed
/// and unpadded data.
hw::HWModuleOp CapnpTypeSchemaImpl::buildDecoder(Value clk, Value valid,
                                                 Value operandVal) {
  auto loc = operandVal.getDefiningOp()->getLoc();
  auto topMod = operandVal.getDefiningOp()->getParentOfType<ModuleOp>();
  OpBuilder b = OpBuilder::atBlockEnd(topMod.getBody());

  SmallString<64> modName;
  modName.append("decode");
  modName.append(base.name());
  SmallVector<hw::PortInfo, 4> ports;
  ports.push_back(hw::PortInfo{
      {b.getStringAttr("clk"), clk.getType(), hw::ModulePort::Direction::Input},
      0});
  ports.push_back(hw::PortInfo{{b.getStringAttr("valid"), valid.getType(),
                                hw::ModulePort::Direction::Input},
                               1});
  ports.push_back(
      hw::PortInfo{{b.getStringAttr("encodedInput"), operandVal.getType(),
                    hw::ModulePort::Direction::Input},
                   2});
  ports.push_back(hw::PortInfo{{b.getStringAttr("decoded"), base.getType(),
                                hw::ModulePort::Direction::Output},
                               0});
  hw::HWModuleOp retMod = b.create<hw::HWModuleOp>(
      operandVal.getLoc(), b.getStringAttr(modName), ports);

  Block *innerBlock = retMod.getBodyBlock();
  b.setInsertionPointToStart(innerBlock);
  clk = innerBlock->getArgument(0);
  valid = innerBlock->getArgument(1);
  operandVal = innerBlock->getArgument(2);

  // Various useful integer types.
  auto i16 = b.getIntegerType(16);

  size_t size = this->size();
  hw::ArrayType operandType = operandVal.getType().dyn_cast<hw::ArrayType>();
  assert(operandType && operandType.getSize() == size &&
         "Operand type and length must match the type's capnp size.");
  (void)size;
  (void)operandType;

  Slice operand(b, operandVal);
  operand.setLoc(loc);

  auto alwaysAt = b.create<sv::AlwaysOp>(loc, sv::EventControl::AtPosEdge, clk);
  auto ifValid =
      OpBuilder(alwaysAt.getBodyRegion()).create<sv::IfOp>(loc, valid);
  AssertBuilder asserts(loc, ifValid.getBodyRegion());

  // The next 64-bits of a capnp message is the root struct pointer.
  ::capnp::schema::Node::Reader rootProto = getCapnpTypeSchema().getProto();
  auto ptr = operand.slice(0, 64).name("rootPointer");

  // Since this is the root, we _expect_ the offset to be zero but that's only
  // guaranteed to be the case with canonically-encoded messages.
  // TODO: support cases where the pointer offset is non-zero.
  Slice assertPtr(ptr);
  auto typeAndOffset = assertPtr.slice(0, 32).name("typeAndOffset");
  if (base.getType().isInteger(0)) {
    asserts.assertEqual(typeAndOffset.slice(0, 2), 0);
    asserts.assertEqual(typeAndOffset.slice(2, 30), 0x3FFFFFFF);
  } else {
    asserts.assertEqual(typeAndOffset, 0);
  }

  // We expect the data section to be equal to the computed data section size.
  auto dataSectionSize =
      assertPtr.slice(32, 16).cast(i16).name("dataSectionSize");
  asserts.assertEqual(dataSectionSize,
                      rootProto.getStruct().getDataWordCount());

  // We expect the pointer section to be equal to the computed pointer section
  // size.
  auto ptrSectionSize =
      assertPtr.slice(48, 16).cast(i16).name("ptrSectionSize");
  asserts.assertEqual(ptrSectionSize, rootProto.getStruct().getPointerCount());

  // Get pointers to the data and pointer sections.
  auto st = rootProto.getStruct();
  auto dataSection =
      operand.slice(64, st.getDataWordCount() * 64).name("dataSection");
  auto ptrSection = operand
                        .slice(64 + (st.getDataWordCount() * 64),
                               rootProto.getStruct().getPointerCount() * 64)
                        .name("ptrSection");

  // Loop through fields.
  SmallVector<GasketComponent, 64> fieldValues;
  for (auto field : st.getFields()) {
    uint16_t idx = field.getCodeOrder();
    assert(idx < base.getFields().size() &&
           "Capnp struct longer than fieldTypes.");
    fieldValues.push_back(decodeField(base.getFields()[idx].type, field,
                                      dataSection, ptrSection, asserts));
  }

  // What to return depends on the type. (e.g. structs have to be constructed
  // from the field values.)
  GasketComponent ret =
      TypeSwitch<Type, GasketComponent>(base.getType())
          .Case([&fieldValues](IntegerType) { return fieldValues[0]; })
          .Case([&fieldValues](hw::ArrayType) { return fieldValues[0]; })
          .Case([&](hw::StructType) {
            SmallVector<Value, 8> rawValues(llvm::map_range(
                fieldValues, [](GasketComponent c) { return c.getValue(); }));
            return GasketComponent(b, b.create<hw::StructCreateOp>(
                                          loc, base.getType(), rawValues));
          });
  ret.name(base.name());

  innerBlock->getTerminator()->erase();
  b.setInsertionPointToEnd(innerBlock);
  auto outputOp = b.create<hw::OutputOp>(loc, ValueRange{ret.getValue()});
  alwaysAt->moveBefore(outputOp);
  return retMod;
}

//===----------------------------------------------------------------------===//
// CapnpTypeSchema wrapper.
//===----------------------------------------------------------------------===//

llvm::SmallDenseMap<Type, hw::HWModuleOp>
    circt::esi::capnp::CapnpTypeSchema::decImplMods;
llvm::SmallDenseMap<Type, hw::HWModuleOp>
    circt::esi::capnp::CapnpTypeSchema::encImplMods;

size_t circt::esi::capnp::CapnpTypeSchema::size() const { return s->size(); }

circt::esi::capnp::CapnpTypeSchema::CapnpTypeSchema(Type outerType)
    : circt::esi::ESIAPIType(outerType) {
  s = std::make_shared<detail::CapnpTypeSchemaImpl>(*this);
}

LogicalResult
circt::esi::capnp::CapnpTypeSchema::write(llvm::raw_ostream &os) const {
  return s->write(os);
}

void circt::esi::capnp::CapnpTypeSchema::writeMetadata(
    llvm::raw_ostream &os) const {
  os << name() << " ";
  emitCapnpID(os, typeID());
}

Value circt::esi::capnp::CapnpTypeSchema::buildEncoder(OpBuilder &builder,
                                                       Value clk, Value valid,
                                                       Value rawData) const {
  hw::HWModuleOp encImplMod;
  auto encImplIT = encImplMods.find(getType());
  if (encImplIT == encImplMods.end()) {
    encImplMod = s->buildEncoder(clk, valid, rawData);
    encImplMods[getType()] = encImplMod;
  } else {
    encImplMod = encImplIT->second;
  }

  SmallString<64> instName;
  instName.append("encode");
  instName.append(name());
  instName.append("Inst");
  auto encodeInst =
      builder.create<hw::InstanceOp>(rawData.getLoc(), encImplMod, instName,
                                     ArrayRef<Value>{clk, valid, rawData});
  return encodeInst.getResult(0);
}

Value circt::esi::capnp::CapnpTypeSchema::buildDecoder(OpBuilder &builder,
                                                       Value clk, Value valid,
                                                       Value capnpData) const {
  hw::HWModuleOp decImplMod;
  auto decImplIT = decImplMods.find(getType());
  if (decImplIT == decImplMods.end()) {
    decImplMod = s->buildDecoder(clk, valid, capnpData);
    decImplMods[getType()] = decImplMod;
  } else {
    decImplMod = decImplIT->second;
  }

  SmallString<64> instName;
  instName.append("decode");
  instName.append(name());
  instName.append("Inst");
  auto decodeInst =
      builder.create<hw::InstanceOp>(capnpData.getLoc(), decImplMod, instName,
                                     ArrayRef<Value>{clk, valid, capnpData});
  return decodeInst.getResult(0);
}
