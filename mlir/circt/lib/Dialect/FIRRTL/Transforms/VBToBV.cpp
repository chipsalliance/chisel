//===- VBToBV.cpp - "Vector of Bundle" to "Bundle of Vector" ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the VBToBV pass, which takes any bundle embedded in
// a vector, and converts it to a vector embedded within a bundle.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Visitor
//===----------------------------------------------------------------------===//

namespace {
class Visitor : public FIRRTLVisitor<Visitor, LogicalResult> {
public:
  explicit Visitor(MLIRContext *);

  LogicalResult visit(FModuleOp);

  using FIRRTLVisitor<Visitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<Visitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<Visitor, LogicalResult>::visitStmt;

  LogicalResult visitUnhandledOp(Operation *);

  LogicalResult visitInvalidOp(Operation *op) {
    return op->emitError("invalid operation operation");
  }

  template <typename Op>
  void emitExplodedConnect(ImplicitLocOpBuilder &, Type, ArrayRef<Value>,
                           ArrayRef<Value>);
  Value emitBundleCreate(ImplicitLocOpBuilder &, Type, ArrayRef<Value>);
  template <typename Op>
  void handleConnect(Op);

  LogicalResult visitStmt(ConnectOp);
  LogicalResult visitStmt(StrictConnectOp);

  LogicalResult visitExpr(AggregateConstantOp);
  LogicalResult visitExpr(VectorCreateOp);
  LogicalResult visitExpr(SubfieldOp);
  LogicalResult visitExpr(SubindexOp);
  LogicalResult visitExpr(SubaccessOp);
  LogicalResult visitExpr(RefSubOp);
  LogicalResult visitExpr(RefResolveOp);

  Type convertType(Type);
  FIRRTLType convertType(FIRRTLType);
  RefType convertType(RefType);
  FIRRTLBaseType convertType(FIRRTLBaseType);
  FIRRTLBaseType convertType(FIRRTLBaseType, SmallVector<unsigned> &);

  Attribute convertConstant(Type, Attribute);
  Attribute convertVectorConstant(FVectorType, ArrayAttr);
  Attribute convertBundleConstant(BundleType, ArrayAttr);
  Attribute convertBundleInVectorConstant(BundleType, ArrayRef<Attribute>);

  /// Blow out an annotation target to it's leaf targets.
  void explodeFieldID(Type, uint64_t,
                      SmallVectorImpl<std::pair<Type, uint64_t>> &);
  void fixAnnotation(Type, Type, DictionaryAttr, SmallVectorImpl<Attribute> &);
  ArrayAttr fixAnnotations(Type, Type, ArrayAttr);

  /// Blow out a value into it's leaf-values.
  void explode(Value, SmallVectorImpl<Value> &);
  SmallVector<Value> explode(Value);
  /// Blow out a ref into it's leaf-refs.
  void explodeRef(Value, SmallVectorImpl<Value> &);
  /// Fix an operand.
  std::pair<SmallVector<Value>, bool> fixOperand(Value);
  /// Fix a read-only/rhs operand. This operand may be rematerialized as a
  /// bundle using an intermediate bundle-create op, which means it is not
  /// possible to write to the converted value.
  Value fixROperand(Value);
  /// Fix a ref-typed operand.
  std::pair<SmallVector<Value>, bool> fixRefOperand(Value);

  /// Bundle/Vector Create Ops
  Value sinkVecDimIntoOperands(ImplicitLocOpBuilder &, FIRRTLBaseType,
                               const SmallVectorImpl<Value> &);

  /// pull an access op from the cache if available, create the op if needed.
  Value getSubfield(Value, unsigned);
  Value getSubindex(Value, unsigned);
  Value getSubaccess(Operation *, Value, Value);
  Value getRefSub(Value, unsigned);

  MLIRContext *context;
  SmallVector<Operation *> toDelete;

  /// A mapping from old values to their fixed up values. If a value is
  /// unchanged, it will be mapped to itself. If a value is present in the map,
  /// and does not map to itself, then  it must be deleted.
  DenseMap<Value, Value> valueMap;

  /// A cache mapping unconverted types to their bv-converted equivalents.
  DenseMap<FIRRTLType, FIRRTLType> typeMap;

  /// A cache of generated subfield/index/access operations.
  DenseMap<std::tuple<Value, unsigned>, Value> subfieldCache;
  DenseMap<std::tuple<Value, unsigned>, Value> subindexCache;
  DenseMap<std::tuple<Operation *, Value, Value>, Value> subaccessCache;
  DenseMap<std::tuple<Value, unsigned>, Value> refSubCache;
};
} // end anonymous namespace

Visitor::Visitor(MLIRContext *context) : context(context) {}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
FIRRTLBaseType Visitor::convertType(FIRRTLBaseType type,
                                    SmallVector<unsigned> &dimensions) {
  if (auto vectorType = type_dyn_cast<FVectorType>(type); vectorType) {
    dimensions.push_back(vectorType.getNumElements());
    auto converted = convertType(vectorType.getElementType(), dimensions);
    dimensions.pop_back();
    return converted;
  }
  if (auto bundleType = type_dyn_cast<BundleType>(type); bundleType) {
    SmallVector<BundleType::BundleElement> elements;
    for (auto element : bundleType.getElements()) {
      elements.push_back(BundleType::BundleElement(
          element.name, element.isFlip, convertType(element.type, dimensions)));
    }
    return BundleType::get(context, elements);
  }
  for (auto size : llvm::reverse(dimensions))
    type = FVectorType::get(type, size);
  return type;
}

FIRRTLBaseType Visitor::convertType(FIRRTLBaseType type) {
  auto cached = typeMap.lookup(type);
  if (cached)
    return type_cast<FIRRTLBaseType>(cached);

  SmallVector<unsigned> dimensions;
  auto converted = convertType(type, dimensions);

  typeMap.insert({type, converted});
  return converted;
}

RefType Visitor::convertType(RefType type) {
  auto cached = typeMap.lookup(type);
  if (cached)
    return type_cast<RefType>(cached);
  auto converted = RefType::get(convertType(type.getType()));
  typeMap.insert({type, converted});
  return converted;
}

FIRRTLType Visitor::convertType(FIRRTLType type) {
  auto cached = typeMap.lookup(type);
  if (cached)
    return type_cast<FIRRTLType>(cached);
  if (auto baseType = type_dyn_cast<FIRRTLBaseType>(type))
    return convertType(baseType);
  if (auto refType = type_dyn_cast<RefType>(type))
    return convertType(refType);
  return type;
}

Type Visitor::convertType(Type type) {
  if (auto firrtlType = type_dyn_cast<FIRRTLType>(type))
    return convertType(firrtlType);
  return type;
}

//===----------------------------------------------------------------------===//
// Annotations
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
void Visitor::explodeFieldID(
    Type type, uint64_t fieldID,
    SmallVectorImpl<std::pair<Type, uint64_t>> &fields) {
  if (auto bundleType = type_dyn_cast<BundleType>(type)) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
      auto eltType = bundleType.getElementType(i);
      auto eltID = fieldID + bundleType.getFieldID(i);
      explodeFieldID(eltType, eltID, fields);
    }
    return;
  }
  fields.emplace_back(type, fieldID);
}

void Visitor::fixAnnotation(Type oldType, Type newType, DictionaryAttr annoAttr,
                            SmallVectorImpl<Attribute> &newAnnos) {
  Annotation anno(annoAttr);
  auto fieldID = anno.getFieldID();

  // If the field ID targets the entire structure, we don't need to make a
  // change.
  if (fieldID == 0) {
    newAnnos.push_back(anno.getAttr());
    return;
  }

  SmallVector<uint32_t> bundleAccesses;
  SmallVector<uint32_t> vectorAccesses;
  while (fieldID != 0) {
    if (auto bundleType = type_dyn_cast<BundleType>(oldType)) {
      auto [index, subID] = bundleType.getIndexAndSubfieldID(fieldID);
      bundleAccesses.push_back(index);
      oldType = bundleType.getElementType(index);
      fieldID = subID;
      continue;
    }
    if (auto vectorType = type_dyn_cast<FVectorType>(oldType)) {
      auto [index, subID] = vectorType.getIndexAndSubfieldID(fieldID);
      vectorAccesses.push_back(index);
      oldType = vectorType.getElementType();
      fieldID = subID;
      continue;
    }
    llvm_unreachable("non-zero field ID can only be used on aggregate types");
  }

  uint64_t newID = 0;
  for (auto index : bundleAccesses) {
    auto bundleType = type_cast<BundleType>(newType);
    newID += bundleType.getFieldID(index);
    newType = bundleType.getElementType(index);
  }

  SmallVector<std::pair<Type, uint64_t>> fields;
  if (type_isa<BundleType>(newType) && !vectorAccesses.empty()) {
    explodeFieldID(newType, newID, fields);
  } else {
    fields.emplace_back(newType, newID);
  }

  auto i64Type = IntegerType::get(context, 64);
  for (auto [type, fieldID] : fields) {
    for (auto index : vectorAccesses) {
      auto vectorType = type_cast<FVectorType>(type);
      type = vectorType.getElementType();
      fieldID += vectorType.getFieldID(index);
    }
    anno.setMember("circt.fieldID", IntegerAttr::get(i64Type, fieldID));
    newAnnos.push_back(anno.getAttr());
  }
}

ArrayAttr Visitor::fixAnnotations(Type oldType, Type newType, ArrayAttr annos) {
  SmallVector<Attribute> newAnnos;
  for (auto anno : cast<ArrayAttr>(annos).getAsRange<DictionaryAttr>())
    fixAnnotation(oldType, newType, anno, newAnnos);
  return ArrayAttr::get(context, newAnnos);
}

//===----------------------------------------------------------------------===//
// Path Building and Caching
//===----------------------------------------------------------------------===//

Value Visitor::getSubfield(Value input, unsigned index) {
  Value &result = subfieldCache[{input, index}];
  if (result)
    return result;

  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(input);
  result = builder.create<SubfieldOp>(input.getLoc(), input, index);
  return result;
}

Value Visitor::getSubindex(Value input, unsigned index) {
  auto &result = subindexCache[{input, index}];
  if (result)
    return result;

  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(input);
  result = builder.create<SubindexOp>(input.getLoc(), input, index);
  return result;
}

Value Visitor::getSubaccess(Operation *place, Value input, Value index) {
  auto &result = subaccessCache[{place, input, index}];
  if (result)
    return result;
  OpBuilder builder(place);
  result = builder.create<SubaccessOp>(input.getLoc(), input, index);
  return result;
}

Value Visitor::getRefSub(Value input, unsigned index) {
  auto &result = refSubCache[{input, index}];
  if (result)
    return result;
  OpBuilder builder(context);
  builder.setInsertionPointAfterValue(input);
  result = builder.create<RefSubOp>(input.getLoc(), input, index);
  return result;
}

//===----------------------------------------------------------------------===//
// Ref Operand Fixup
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
void Visitor::explodeRef(Value value, SmallVectorImpl<Value> &output) {
  auto underlyingType = type_cast<RefType>(value.getType()).getType();
  if (auto bundleType = type_dyn_cast<BundleType>(underlyingType)) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
      OpBuilder builder(context);
      builder.setInsertionPointAfterValue(value);
      auto field = builder.create<RefSubOp>(value.getLoc(), value, i);
      explodeRef(field, output);
    }
    return;
  }
  output.push_back(value);
}

std::pair<SmallVector<Value>, bool> Visitor::fixRefOperand(Value value) {
  SmallVector<unsigned> bundleAccesses;
  SmallVector<unsigned> vectorAccesses;

  // Sort subref operations into either bundle or vector accesses.
  while (value) {
    Operation *op = value.getDefiningOp();
    if (!op)
      break;
    if (auto refSubOp = dyn_cast<RefSubOp>(op)) {
      value = refSubOp.getInput();
      auto type = type_cast<RefType>(value.getType()).getType();
      if (type_isa<BundleType>(type))
        bundleAccesses.push_back(refSubOp.getIndex());
      else if (type_isa<FVectorType>(type))
        vectorAccesses.push_back(refSubOp.getIndex());
      else
        refSubOp->emitError("unknown aggregate type");
      continue;
    }
    break;
  }

  // value now points at canonical storage location of the ref operand.
  // get the corresponding converted object.
  value = valueMap[value];
  assert(value);

  // replay the bundle accesses first.
  for (auto index : llvm::reverse(bundleAccesses)) {
    value = getRefSub(value, index);
  }

  // If the current value is a bundle type, but we need to replay vector access,
  // that indicates an AOS->SOA conversion occurred, and a vector was sunk into
  // a bundle. Explode the bundle object to it's leaves, all of which must be
  // an arm of the sunken vector.
  SmallVector<Value> values;
  bool exploded = false;
  if (type_isa<BundleType>(type_cast<RefType>(value.getType()).getType()) &&
      !vectorAccesses.empty()) {
    explodeRef(value, values);
    exploded = true;
  } else {
    values.push_back(value);
    exploded = false;
  }

  // Finally, replay any vector access operations on each of the output values.
  for (auto &value : values) {
    for (auto index : llvm::reverse(vectorAccesses)) {
      value = getRefSub(value, index);
    }
  }

  return {values, exploded};
}

//===----------------------------------------------------------------------===//
// Operand Fixup
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
void Visitor::explode(Value value, SmallVectorImpl<Value> &output) {
  auto type = value.getType();
  if (auto bundleType = type_dyn_cast<BundleType>(type)) {
    for (size_t i = 0, e = bundleType.getNumElements(); i < e; ++i) {
      auto field = getSubfield(value, i);
      explode(field, output);
    }
    return;
  }
  output.push_back(value);
}

SmallVector<Value> Visitor::explode(Value value) {
  auto output = SmallVector<Value>();
  explode(value, output);
  return output;
}

// NOLINTNEXTLINE(misc-no-recursion)
std::pair<SmallVector<Value>, bool> Visitor::fixOperand(Value value) {
  auto type = value.getType();

  // If the operand is a ref-type, we have a different mechanism for repairing
  // the path.
  if (type_isa<RefType>(type))
    return fixRefOperand(value);

  SmallVector<SubfieldOp> bundleAccesses;
  SmallVector<Operation *> vectorAccesses;

  // Walk back through the subaccess ops to the canonical storage location.
  // Collect the path according to the type of access, splitting bundle
  // accesses ops from vector accesses ops.
  while (value) {
    Operation *op = value.getDefiningOp();
    if (!op)
      break;
    if (auto subfieldOp = dyn_cast<SubfieldOp>(op)) {
      value = subfieldOp.getInput();
      bundleAccesses.push_back(subfieldOp);
      continue;
    }
    if (auto subindexOp = dyn_cast<SubindexOp>(op)) {
      value = subindexOp.getInput();
      vectorAccesses.push_back(subindexOp);
      continue;
    }
    if (auto subaccessOp = dyn_cast<SubaccessOp>(op)) {
      value = subaccessOp.getInput();
      vectorAccesses.push_back(subaccessOp);
      continue;
    }
    break;
  }

  // Value now points at the original canonical storage location.
  // Get the converted equivalent object.
  value = valueMap[value];
  assert(value && "canonical storage location must have been converted");

  // Replay the subaccess operations, in the converted order;
  // bundle accesses first, then vector accesses.
  for (auto subfieldOp : llvm::reverse(bundleAccesses))
    value = getSubfield(value, subfieldOp.getFieldIndex());

  // If the current value is a bundle, but we have vector accesses to replay,
  // we must explode the bundle and apply the vector accesses to each leaf of
  // the bundle. The leaves will be vectors corresponding to the sunken vector.
  SmallVector<Value> values;
  bool exploded = false;

  if (type_isa<BundleType>(value.getType()) && !vectorAccesses.empty()) {
    explode(value, values);
    exploded = true;
  } else {
    values.push_back(value);
    exploded = false;
  }

  // Finally, replay the vector access operations.
  for (auto &value : values) {
    for (auto *op : llvm::reverse(vectorAccesses)) {
      if (auto subindexOp = dyn_cast<SubindexOp>(op)) {
        value = getSubindex(value, subindexOp.getIndex());
        continue;
      }
      if (auto subaccessOp = dyn_cast<SubaccessOp>(op)) {
        auto index = fixROperand(subaccessOp.getIndex());
        value = getSubaccess(subaccessOp, value, index);
        continue;
      }
    }
  }

  return {values, exploded};
}

//===----------------------------------------------------------------------===//
// Read-Only / RHS Operand Fixup
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
Value Visitor::fixROperand(Value operand) {
  auto [values, exploded] = fixOperand(operand);
  if (!exploded)
    return values.front();

  // The operand must be materialized into a single read-only bundle.
  auto newType = convertType(operand.getType());
  ImplicitLocOpBuilder builder(operand.getLoc(), context);
  builder.setInsertionPointAfterValue(operand);
  return emitBundleCreate(builder, newType, values);
}

//===----------------------------------------------------------------------===//
// Base Case -- Any Regular Operation
//===----------------------------------------------------------------------===//

LogicalResult Visitor::visitUnhandledOp(Operation *op) {
  ImplicitLocOpBuilder builder(op->getLoc(), op);
  bool changed = false;

  // Typical operations read from passive operands, only.
  // We can materialize any passive operand into a single value, potentially
  // with fresh intermediate bundle create ops in between.
  SmallVector<Value> newOperands;
  for (auto oldOperand : op->getOperands()) {
    auto newOperand = fixROperand(oldOperand);
    changed |= (oldOperand != newOperand);
    newOperands.push_back(newOperand);
  }

  // We can rewrite the type of any result, but if any result type changes,
  // then the operation will be cloned.
  SmallVector<Type> newTypes;
  for (auto oldResult : op->getResults()) {
    auto oldType = oldResult.getType();
    auto newType = convertType(oldType);
    changed |= oldType != newType;
    newTypes.push_back(newType);
  }

  if (changed) {
    auto *newOp = builder.clone(*op);
    newOp->setOperands(newOperands);
    for (size_t i = 0, e = op->getNumResults(); i < e; ++i) {
      auto newResult = newOp->getResult(i);
      newResult.setType(newTypes[i]);
      valueMap[op->getResult(i)] = newResult;
    }

    // Annotation updates.
    if (auto portAnnos = op->getAttrOfType<ArrayAttr>("portAnnotations")) {
      // Update port annotations. We make a hard assumption that there is one
      // operation result per set of port annotations.
      SmallVector<Attribute> newPortAnnos;
      for (unsigned i = 0, e = portAnnos.size(); i < e; ++i) {
        auto oldType = op->getResult(i).getType();
        auto newType = newTypes[i];
        newPortAnnos.push_back(
            fixAnnotations(oldType, newType, cast<ArrayAttr>(portAnnos[i])));
      }
      newOp->setAttr("portAnnotations", ArrayAttr::get(context, newPortAnnos));
    } else if (newOp->getNumResults() == 1) {
      // Update annotations. If the operation does not have exactly 1 result,
      // then we have no type change with which to understand how to transform
      // the annotations. We do not update the regular annotations if the
      // operation had port annotations.
      if (auto annos = newOp->getAttrOfType<ArrayAttr>("annotations")) {
        auto oldType = op->getResult(0).getType();
        auto newType = newTypes[0];
        auto newAnnos = fixAnnotations(oldType, newType, annos);
        AnnotationSet(newAnnos, context).applyToOperation(newOp);
      }
    }

    toDelete.push_back(op);
    op = newOp;

  } else {
    // As a safety precaution, all unchanged "canonical storage locations"
    // must be mapped to themselves.
    for (auto result : op->getResults())
      valueMap[result] = result;
  }

  for (auto &region : op->getRegions())
    for (auto &block : region.getBlocks())
      for (auto &op : block)
        if (failed(dispatchVisitor(&op)))
          return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

template <typename Op>
void Visitor::emitExplodedConnect(ImplicitLocOpBuilder &builder, Type type,
                                  ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  assert(lhs.size() == rhs.size() &&
         "Something went wrong exploding the elements");
  const auto *lhsIt = lhs.begin();
  const auto *rhsIt = rhs.begin();

  auto explodeConnect = [&](auto self, Type type, bool flip = false) -> void {
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      for (auto &element : bundleType) {
        self(self, element.type, flip ^ element.isFlip);
      }
      return;
    }
    auto lhs = *lhsIt++;
    auto rhs = *rhsIt++;
    if (flip)
      std::swap(lhs, rhs);
    builder.create<Op>(lhs, rhs);
  };
  explodeConnect(explodeConnect, type);
}

Value Visitor::emitBundleCreate(ImplicitLocOpBuilder &builder, Type type,
                                ArrayRef<Value> values) {
  auto *it = values.begin();
  auto convert = [&](auto self, Type type) -> Value {
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      SmallVector<Value> fields;
      for (auto element : bundleType.getElements()) {
        fields.push_back(self(self, element.type));
      }
      return builder.create<BundleCreateOp>(type, fields);
    }
    return *(it++);
  };
  return convert(convert, type);
}

template <typename Op>
void Visitor::handleConnect(Op op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto oldLhs = op.getDest();
  auto oldRhs = op.getSrc();

  auto oldType = type_cast<FIRRTLType>(oldLhs.getType());
  auto type = convertType(oldType);

  auto [lhs, lhsExploded] = fixOperand(oldLhs);
  auto [rhs, rhsExploded] = fixOperand(oldRhs);

  if (!lhsExploded && !rhsExploded && oldLhs == lhs[0] && oldRhs == rhs[0])
    return;

  if (lhsExploded) {
    if (rhsExploded) {
      emitExplodedConnect<Op>(builder, type, lhs, rhs);
    } else {
      emitExplodedConnect<Op>(builder, type, lhs, explode(rhs[0]));
    }
  } else {
    if (rhsExploded) {
      if (auto baseType = type_dyn_cast<FIRRTLBaseType>(type);
          baseType && baseType.isPassive()) {
        builder.create<Op>(lhs[0], emitBundleCreate(builder, type, rhs));
      } else {
        emitExplodedConnect<Op>(builder, type, explode(lhs[0]), rhs);
      }
    } else {
      builder.create<Op>(lhs[0], rhs[0]);
    }
  }

  toDelete.push_back(op);
}

LogicalResult Visitor::visitStmt(ConnectOp op) {
  handleConnect(op);
  return success();
}

LogicalResult Visitor::visitStmt(StrictConnectOp op) {
  handleConnect(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Conversion
//===----------------------------------------------------------------------===//

Attribute Visitor::convertBundleInVectorConstant(BundleType type,
                                                 ArrayRef<Attribute> fields) {
  auto numBundleFields = type.getNumElements();
  SmallVector<SmallVector<Attribute>> newBundleFields;
  newBundleFields.resize(numBundleFields);
  for (auto bundle : fields) {
    auto subfields = cast<ArrayAttr>(bundle);
    for (size_t i = 0; i < numBundleFields; ++i) {
      newBundleFields[i].push_back(subfields[i]);
    }
  }

  SmallVector<Attribute> newFieldAttrs;
  for (auto &newBundleField : newBundleFields) {
    newFieldAttrs.push_back(ArrayAttr::get(context, newBundleField));
  }
  return ArrayAttr::get(context, newFieldAttrs);
}

// NOLINTNEXTLINE(misc-no-recursion)
Attribute Visitor::convertVectorConstant(FVectorType oldType,
                                         ArrayAttr oldElements) {
  auto oldElementType = oldType.getElementType();
  auto newElementType = convertType(oldElementType);

  if (oldElementType == newElementType)
    if (auto bundleElementType = type_dyn_cast<BundleType>(oldElementType))
      return convertBundleInVectorConstant(bundleElementType,
                                           oldElements.getValue());

  SmallVector<Attribute> newElements;
  for (auto oldElement : oldElements) {
    newElements.push_back(convertConstant(oldElementType, oldElement));
  }

  auto bundleType = type_cast<BundleType>(newElementType);
  return convertBundleInVectorConstant(bundleType, newElements);
}

// NOLINTNEXTLINE(misc-no-recursion)
Attribute Visitor::convertBundleConstant(BundleType type, ArrayAttr fields) {
  SmallVector<Attribute> converted;
  auto elements = type.getElements();
  for (size_t i = 0, e = elements.size(); i < e; ++i) {
    converted.push_back(convertConstant(elements[i].type, fields[i]));
  }
  return ArrayAttr::get(context, converted);
}

// NOLINTNEXTLINE(misc-no-recursion)
Attribute Visitor::convertConstant(Type type, Attribute value) {
  if (auto bundleType = type_dyn_cast<BundleType>(type))
    return convertBundleConstant(bundleType, cast<ArrayAttr>(value));

  if (auto vectorType = type_dyn_cast<FVectorType>(type))
    return convertVectorConstant(vectorType, cast<ArrayAttr>(value));

  return value;
}

LogicalResult Visitor::visitExpr(AggregateConstantOp op) {
  auto oldValue = op.getResult();

  auto oldType = oldValue.getType();
  auto newType = convertType(oldType);
  if (oldType == newType) {
    valueMap[oldValue] = oldValue;
    return success();
  }

  auto fields = cast<ArrayAttr>(convertConstant(oldType, op.getFields()));

  OpBuilder builder(op);
  auto newOp =
      builder.create<AggregateConstantOp>(op.getLoc(), newType, fields);

  valueMap[oldValue] = newOp.getResult();
  toDelete.push_back(op);

  return success();
}

//===----------------------------------------------------------------------===//
// Aggregate Create Ops
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
Value Visitor::sinkVecDimIntoOperands(ImplicitLocOpBuilder &builder,
                                      FIRRTLBaseType type,
                                      const SmallVectorImpl<Value> &values) {
  auto length = values.size();
  if (auto bundleType = type_dyn_cast<BundleType>(type)) {
    SmallVector<Value> newFields;
    SmallVector<BundleType::BundleElement> newElements;
    for (auto [i, elt] : llvm::enumerate(bundleType)) {
      SmallVector<Value> subValues;
      for (auto v : values)
        subValues.push_back(getSubfield(v, i));
      auto newField = sinkVecDimIntoOperands(builder, elt.type, subValues);
      newFields.push_back(newField);
      newElements.emplace_back(elt.name, /*isFlip=*/false,
                               type_cast<FIRRTLBaseType>(newField.getType()));
    }
    auto newType = BundleType::get(builder.getContext(), newElements);
    auto newBundle = builder.create<BundleCreateOp>(newType, newFields);
    return newBundle;
  }
  auto newType = FVectorType::get(type, length);
  return builder.create<VectorCreateOp>(newType, values);
}

LogicalResult Visitor::visitExpr(VectorCreateOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);

  auto oldType = op.getType();
  auto newType = convertType(oldType);

  if (oldType == newType) {
    auto changed = false;
    SmallVector<Value> newFields;
    for (auto oldField : op.getFields()) {
      auto newField = fixROperand(oldField);
      if (oldField != newField)
        changed = true;
      newFields.push_back(newField);
    }

    if (!changed) {
      auto result = op.getResult();
      valueMap[result] = result;
      return success();
    }

    auto newOp =
        builder.create<VectorCreateOp>(op.getLoc(), newType, newFields);
    valueMap[op.getResult()] = newOp.getResult();
    toDelete.push_back(op);
    return success();
  }

  // OK, We are in for some pain!
  SmallVector<Value> convertedOldFields;
  for (auto oldField : op.getFields()) {
    auto convertedField = fixROperand(oldField);
    convertedOldFields.push_back(convertedField);
  }

  auto value = sinkVecDimIntoOperands(
      builder, convertType(oldType.get().getElementType()), convertedOldFields);
  valueMap[op.getResult()] = value;
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pathing Ops
//===----------------------------------------------------------------------===//

LogicalResult Visitor::visitExpr(SubfieldOp op) {
  toDelete.push_back(op);
  return success();
}

LogicalResult Visitor::visitExpr(SubindexOp op) {
  toDelete.push_back(op);
  return success();
}

LogicalResult Visitor::visitExpr(SubaccessOp op) {
  toDelete.push_back(op);
  return success();
}

LogicalResult Visitor::visitExpr(RefSubOp op) {
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Ref Ops
//===----------------------------------------------------------------------===//

LogicalResult Visitor::visitExpr(RefResolveOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto [refs, exploded] = fixRefOperand(op.getRef());
  if (!exploded) {
    auto ref = refs[0];
    if (ref == op.getRef()) {
      valueMap[op.getResult()] = op.getResult();
      return success();
    }
    auto value = builder.create<RefResolveOp>(convertType(op.getType()), ref)
                     .getResult();
    valueMap[op.getResult()] = value;
    toDelete.push_back(op);
    return success();
  }

  auto type = convertType(op.getType());
  SmallVector<Value> values;
  for (auto ref : refs) {
    values.push_back(builder
                         .create<RefResolveOp>(
                             type_cast<RefType>(ref.getType()).getType(), ref)
                         .getResult());
  }
  auto value = emitBundleCreate(builder, type, values);
  valueMap[op.getResult()] = value;
  toDelete.push_back(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor Entrypoint
//===----------------------------------------------------------------------===//

LogicalResult Visitor::visit(FModuleOp op) {
  BitVector portsToErase(op.getNumPorts() * 2);
  {
    SmallVector<std::pair<unsigned, PortInfo>> newPorts;
    auto ports = op.getPorts();
    auto count = 0;
    for (auto [index, port] : llvm::enumerate(ports)) {
      auto oldType = port.type;
      auto newType = convertType(oldType);
      if (newType == oldType)
        continue;
      auto newPort = port;
      newPort.type = newType;
      newPort.annotations = AnnotationSet(
          fixAnnotations(oldType, newType, port.annotations.getArrayAttr()));
      portsToErase[count + index] = true;
      newPorts.push_back({index + 1, newPort});

      ++count;
    }
    op.insertPorts(newPorts);
  }

  auto *body = op.getBodyBlock();
  for (unsigned i = 0, e = body->getNumArguments(); i < e; ++i) {
    if (portsToErase[i]) {
      auto oldArg = body->getArgument(i);
      auto newArg = body->getArgument(i + 1);
      valueMap[oldArg] = newArg;
    } else {
      auto oldArg = body->getArgument(i);
      valueMap[oldArg] = oldArg;
    }
  }

  for (auto &op : *body) {
    if (failed(dispatchVisitor(&op)))
      return failure();
  }

  while (!toDelete.empty())
    toDelete.pop_back_val()->erase();
  op.erasePorts(portsToErase);

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class VBToBVPass : public VBToBVBase<VBToBVPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void VBToBVPass::runOnOperation() {
  std::vector<FModuleOp> modules;
  llvm::append_range(modules, getOperation().getBody().getOps<FModuleOp>());
  auto result =
      failableParallelForEach(&getContext(), modules, [&](FModuleOp module) {
        Visitor visitor(&getContext());
        return visitor.visit(module);
      });

  if (result.failed())
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createVBToBVPass() {
  return std::make_unique<VBToBVPass>();
}
