//===- LowerTypes.cpp - Lower Aggregate Types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerTypes pass.  This pass replaces aggregate types
// with expanded values.
//
// This pass walks the operations in reverse order. This lets it visit users
// before defs. Users can usually be expanded out to multiple operations (think
// mux of a bundle to muxes of each field) with a temporary subWhatever op
// inserted. When processing an aggregate producer, we blow out the op as
// appropriate, then walk the users, often those are subWhatever ops which can
// be bypassed and deleted. Function arguments are logically last on the
// operation visit order and walked left to right, being peeled one layer at a
// time with replacements inserted to the right of the original argument.
//
// Each processing of an op peels one layer of aggregate type off.  Because new
// ops are inserted immediately above the current up, the walk will visit them
// next, effectively recusing on the aggregate types, without recusing.  These
// potentially temporary ops(if the aggregate is complex) effectively serve as
// the worklist.  Often aggregates are shallow, so the new ops are the final
// ones.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "firrtl-lower-types"

using namespace circt;
using namespace firrtl;

// TODO: check all argument types
namespace {
/// This represents a flattened bundle field element.
struct FlatBundleFieldEntry {
  /// This is the underlying ground type of the field.
  FIRRTLBaseType type;
  /// The index in the parent type
  size_t index;
  /// The fieldID
  unsigned fieldID;
  /// This is a suffix to add to the field name to make it unique.
  SmallString<16> suffix;
  /// This indicates whether the field was flipped to be an output.
  bool isOutput;

  FlatBundleFieldEntry(const FIRRTLBaseType &type, size_t index,
                       unsigned fieldID, StringRef suffix, bool isOutput)
      : type(type), index(index), fieldID(fieldID), suffix(suffix),
        isOutput(isOutput) {}

  void dump() const {
    llvm::errs() << "FBFE{" << type << " index<" << index << "> fieldID<"
                 << fieldID << "> suffix<" << suffix << "> isOutput<"
                 << isOutput << ">}\n";
  }
};
} // end anonymous namespace

/// Return true if the type has more than zero bitwidth.
static bool hasZeroBitWidth(FIRRTLType type) {
  return FIRRTLTypeSwitch<FIRRTLType, bool>(type)
      .Case<BundleType>([&](auto bundle) {
        for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
          auto elt = bundle.getElement(i);
          if (hasZeroBitWidth(elt.type))
            return true;
        }
        return bundle.getNumElements() == 0;
      })
      .Case<FVectorType>([&](auto vector) {
        if (vector.getNumElements() == 0)
          return true;
        return hasZeroBitWidth(vector.getElementType());
      })
      .Case<FIRRTLBaseType>([](auto groundType) {
        return firrtl::getBitWidth(groundType).value_or(0) == 0;
      })
      .Case<RefType>([](auto ref) { return hasZeroBitWidth(ref.getType()); })
      .Default([](auto) { return false; });
}

/// Return true if the type is a 1d vector type or ground type.
static bool isOneDimVectorType(FIRRTLType type) {
  return FIRRTLTypeSwitch<FIRRTLType, bool>(type)
      .Case<BundleType>([&](auto bundle) { return false; })
      .Case<FVectorType>([&](FVectorType vector) {
        // When the size is 1, lower the vector into a scalar.
        return vector.getElementType().isGround() &&
               vector.getNumElements() > 1;
      })
      .Default([](auto groundType) { return true; });
}

/// Return true if the type has a bundle type as subtype.
static bool containsBundleType(FIRRTLType type) {
  return FIRRTLTypeSwitch<FIRRTLType, bool>(type)
      .Case<BundleType>([&](auto bundle) { return true; })
      .Case<FVectorType>([&](FVectorType vector) {
        return containsBundleType(vector.getElementType());
      })
      .Default([](auto groundType) { return false; });
}

/// Return true if we can preserve the type.
static bool isPreservableAggregateType(Type type,
                                       PreserveAggregate::PreserveMode mode) {
  if (auto refType = type_dyn_cast<RefType>(type)) {
    // Always preserve rwprobe's.
    if (refType.getForceable())
      return true;
    // FIXME: Don't preserve read-only RefType for now. This is workaround for
    // MemTap which causes type mismatches (issue 4479).
    return false;
  }

  // Return false if no aggregate value is preserved.
  if (mode == PreserveAggregate::None)
    return false;

  auto firrtlType = type_dyn_cast<FIRRTLBaseType>(type);
  if (!firrtlType)
    return false;

  // We can a preserve the type iff (i) the type is not passive, (ii) the type
  // doesn't contain analog and (iii) type don't contain zero bitwidth.
  if (!firrtlType.isPassive() || firrtlType.containsAnalog() ||
      hasZeroBitWidth(firrtlType))
    return false;

  switch (mode) {
  case PreserveAggregate::All:
    return true;
  case PreserveAggregate::OneDimVec:
    return isOneDimVectorType(firrtlType);
  case PreserveAggregate::Vec:
    return !containsBundleType(firrtlType);
  default:
    llvm_unreachable("unexpected mode");
  }
}

/// Peel one layer of an aggregate type into its components.  Type may be
/// complex, but empty, in which case fields is empty, but the return is true.
static bool peelType(Type type, SmallVectorImpl<FlatBundleFieldEntry> &fields,
                     PreserveAggregate::PreserveMode mode) {
  // If the aggregate preservation is enabled and the type is preservable,
  // then just return.
  if (isPreservableAggregateType(type, mode))
    return false;

  if (auto refType = type_dyn_cast<RefType>(type))
    type = refType.getType();
  return FIRRTLTypeSwitch<Type, bool>(type)
      .Case<BundleType>([&](auto bundle) {
        SmallString<16> tmpSuffix;
        // Otherwise, we have a bundle type.  Break it down.
        for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
          auto elt = bundle.getElement(i);
          // Construct the suffix to pass down.
          tmpSuffix.resize(0);
          tmpSuffix.push_back('_');
          tmpSuffix.append(elt.name.getValue());
          fields.emplace_back(elt.type, i, bundle.getFieldID(i), tmpSuffix,
                              elt.isFlip);
        }
        return true;
      })
      .Case<FVectorType>([&](auto vector) {
        // Increment the field ID to point to the first element.
        for (size_t i = 0, e = vector.getNumElements(); i != e; ++i) {
          fields.emplace_back(vector.getElementType(), i, vector.getFieldID(i),
                              "_" + std::to_string(i), false);
        }
        return true;
      })
      .Default([](auto op) { return false; });
}

/// Return if something is not a normal subaccess.  Non-normal includes
/// zero-length vectors and constant indexes (which are really subindexes).
static bool isNotSubAccess(Operation *op) {
  SubaccessOp sao = llvm::dyn_cast<SubaccessOp>(op);
  if (!sao)
    return true;
  ConstantOp arg =
      llvm::dyn_cast_or_null<ConstantOp>(sao.getIndex().getDefiningOp());
  return arg && sao.getInput().getType().get().getNumElements() != 0;
}

/// Look through and collect subfields leading to a subaccess.
static SmallVector<Operation *> getSAWritePath(Operation *op) {
  SmallVector<Operation *> retval;
  auto defOp = op->getOperand(0).getDefiningOp();
  while (isa_and_nonnull<SubfieldOp, SubindexOp, SubaccessOp>(defOp)) {
    retval.push_back(defOp);
    defOp = defOp->getOperand(0).getDefiningOp();
  }
  // Trim to the subaccess
  while (!retval.empty() && isNotSubAccess(retval.back()))
    retval.pop_back();
  return retval;
}

/// Clone memory for the specified field.  Returns null op on error.
static MemOp cloneMemWithNewType(ImplicitLocOpBuilder *b, MemOp op,
                                 FlatBundleFieldEntry field) {
  SmallVector<Type, 8> ports;
  SmallVector<Attribute, 8> portNames;
  SmallVector<Attribute, 8> portLocations;

  auto oldPorts = op.getPorts();
  for (size_t portIdx = 0, e = oldPorts.size(); portIdx < e; ++portIdx) {
    auto port = oldPorts[portIdx];
    ports.push_back(
        MemOp::getTypeForPort(op.getDepth(), field.type, port.second));
    portNames.push_back(port.first);
  }

  // It's easier to duplicate the old annotations, then fix and filter them.
  auto newMem = b->create<MemOp>(
      ports, op.getReadLatency(), op.getWriteLatency(), op.getDepth(),
      op.getRuw(), portNames, (op.getName() + field.suffix).str(),
      op.getNameKind(), op.getAnnotations().getValue(),
      op.getPortAnnotations().getValue(), op.getInnerSymAttr());

  if (op.getInnerSym()) {
    op.emitError("cannot split memory with symbol present");
    return {};
  }

  SmallVector<Attribute> newAnnotations;
  for (size_t portIdx = 0, e = newMem.getNumResults(); portIdx < e; ++portIdx) {
    auto portType = type_cast<BundleType>(newMem.getResult(portIdx).getType());
    auto oldPortType = type_cast<BundleType>(op.getResult(portIdx).getType());
    SmallVector<Attribute> portAnno;
    for (auto attr : newMem.getPortAnnotation(portIdx)) {
      Annotation anno(attr);
      if (auto annoFieldID = anno.getFieldID()) {
        auto targetIndex = oldPortType.getIndexForFieldID(annoFieldID);

        // Apply annotations to all elements if the target is the whole
        // sub-field.
        if (annoFieldID == oldPortType.getFieldID(targetIndex)) {
          anno.setMember(
              "circt.fieldID",
              b->getI32IntegerAttr(portType.getFieldID(targetIndex)));
          portAnno.push_back(anno.getDict());
          continue;
        }

        // Handle aggregate sub-fields, including `(r/w)data` and `(w)mask`.
        if (type_isa<BundleType>(oldPortType.getElement(targetIndex).type)) {
          // Check whether the annotation falls into the range of the current
          // field. Note that the `field` here is peeled from the `data`
          // sub-field of the memory port, thus we need to add the fieldID of
          // `data` or `mask` sub-field to get the "real" fieldID.
          auto fieldID = field.fieldID + oldPortType.getFieldID(targetIndex);
          if (annoFieldID >= fieldID &&
              annoFieldID <= fieldID + field.type.getMaxFieldID()) {
            // Set the field ID of the new annotation.
            auto newFieldID =
                annoFieldID - fieldID + portType.getFieldID(targetIndex);
            anno.setMember("circt.fieldID", b->getI32IntegerAttr(newFieldID));
            portAnno.push_back(anno.getDict());
          }
        }
      } else
        portAnno.push_back(attr);
    }
    newAnnotations.push_back(b->getArrayAttr(portAnno));
  }
  newMem.setAllPortAnnotations(newAnnotations);
  return newMem;
}

//===----------------------------------------------------------------------===//
// Module Type Lowering
//===----------------------------------------------------------------------===//
namespace {

struct AttrCache {
  AttrCache(MLIRContext *context) {
    i64ty = IntegerType::get(context, 64);
    innerSymAttr = StringAttr::get(context, "inner_sym");
    nameAttr = StringAttr::get(context, "name");
    nameKindAttr = StringAttr::get(context, "nameKind");
    sPortDirections = StringAttr::get(context, "portDirections");
    sPortNames = StringAttr::get(context, "portNames");
    sPortTypes = StringAttr::get(context, "portTypes");
    sPortSyms = StringAttr::get(context, "portSyms");
    sPortLocations = StringAttr::get(context, "portLocations");
    sPortAnnotations = StringAttr::get(context, "portAnnotations");
    sEmpty = StringAttr::get(context, "");
  }
  AttrCache(const AttrCache &) = default;

  Type i64ty;
  StringAttr innerSymAttr, nameAttr, nameKindAttr, sPortDirections, sPortNames,
      sPortTypes, sPortSyms, sPortLocations, sPortAnnotations, sEmpty;
};

// The visitors all return true if the operation should be deleted, false if
// not.
struct TypeLoweringVisitor : public FIRRTLVisitor<TypeLoweringVisitor, bool> {

  TypeLoweringVisitor(
      MLIRContext *context, PreserveAggregate::PreserveMode preserveAggregate,
      PreserveAggregate::PreserveMode memoryPreservationMode,
      SymbolTable &symTbl, const AttrCache &cache,
      const llvm::DenseMap<FModuleLike, Convention> &conventionTable)
      : context(context), aggregatePreservationMode(preserveAggregate),
        memoryPreservationMode(memoryPreservationMode), symTbl(symTbl),
        cache(cache), conventionTable(conventionTable) {}
  using FIRRTLVisitor<TypeLoweringVisitor, bool>::visitDecl;
  using FIRRTLVisitor<TypeLoweringVisitor, bool>::visitExpr;
  using FIRRTLVisitor<TypeLoweringVisitor, bool>::visitStmt;

  /// If the referenced operation is a FModuleOp or an FExtModuleOp, perform
  /// type lowering on all operations.
  void lowerModule(FModuleLike op);

  bool lowerArg(FModuleLike module, size_t argIndex, size_t argsRemoved,
                SmallVectorImpl<PortInfo> &newArgs,
                SmallVectorImpl<Value> &lowering);
  std::pair<Value, PortInfo> addArg(Operation *module, unsigned insertPt,
                                    unsigned insertPtOffset, FIRRTLType srcType,
                                    FlatBundleFieldEntry field,
                                    PortInfo &oldArg, hw::InnerSymAttr newSym);

  // Helpers to manage state.
  bool visitDecl(FExtModuleOp op);
  bool visitDecl(FModuleOp op);
  bool visitDecl(InstanceOp op);
  bool visitDecl(MemOp op);
  bool visitDecl(NodeOp op);
  bool visitDecl(RegOp op);
  bool visitDecl(WireOp op);
  bool visitDecl(RegResetOp op);
  bool visitExpr(InvalidValueOp op);
  bool visitExpr(SubaccessOp op);
  bool visitExpr(VectorCreateOp op);
  bool visitExpr(BundleCreateOp op);
  bool visitExpr(ElementwiseAndPrimOp op);
  bool visitExpr(ElementwiseOrPrimOp op);
  bool visitExpr(ElementwiseXorPrimOp op);
  bool visitExpr(MultibitMuxOp op);
  bool visitExpr(MuxPrimOp op);
  bool visitExpr(Mux2CellIntrinsicOp op);
  bool visitExpr(Mux4CellIntrinsicOp op);
  bool visitExpr(mlir::UnrealizedConversionCastOp op);
  bool visitExpr(BitCastOp op);
  bool visitExpr(RefSendOp op);
  bool visitExpr(RefResolveOp op);
  bool visitExpr(RefCastOp op);
  bool visitStmt(ConnectOp op);
  bool visitStmt(StrictConnectOp op);
  bool visitStmt(RefDefineOp op);
  bool visitStmt(WhenOp op);
  bool visitStmt(GroupOp op);

  bool isFailed() const { return encounteredError; }

private:
  void processUsers(Value val, ArrayRef<Value> mapping);
  bool processSAPath(Operation *);
  void lowerBlock(Block *);
  void lowerSAWritePath(Operation *, ArrayRef<Operation *> writePath);

  /// Lower a "producer" operation one layer based on policy.
  /// Use the provided \p clone function to generate individual ops for
  /// the expanded subelements/fields.  The type used to determine if lowering
  /// is needed is either \p srcType if provided or from the assumed-to-exist
  /// first result of the operation.  When lowering, the clone callback will be
  /// invoked with each subelement/field of this type.
  bool lowerProducer(
      Operation *op,
      llvm::function_ref<Value(const FlatBundleFieldEntry &, ArrayAttr)> clone,
      Type srcType = {});

  /// Filter out and return \p annotations that target includes \field,
  /// modifying as needed to adjust fieldID's relative to to \field.
  ArrayAttr filterAnnotations(MLIRContext *ctxt, ArrayAttr annotations,
                              FIRRTLType srcType, FlatBundleFieldEntry field);

  /// Partition inner symbols on given type.  Fails if any symbols
  /// cannot be assigned to a field, such as inner symbol on root.
  LogicalResult partitionSymbols(hw::InnerSymAttr sym, FIRRTLType parentType,
                                 SmallVectorImpl<hw::InnerSymAttr> &newSyms,
                                 Location errorLoc);

  PreserveAggregate::PreserveMode
  getPreservationModeForModule(FModuleLike moduleLike);
  Value getSubWhatever(Value val, size_t index);

  size_t uniqueIdx = 0;
  std::string uniqueName() {
    auto myID = uniqueIdx++;
    return (Twine("__GEN_") + Twine(myID)).str();
  }

  MLIRContext *context;

  /// Aggregate preservation mode.
  PreserveAggregate::PreserveMode aggregatePreservationMode;
  PreserveAggregate::PreserveMode memoryPreservationMode;

  /// The builder is set and maintained in the main loop.
  ImplicitLocOpBuilder *builder;

  // Keep a symbol table around for resolving symbols
  SymbolTable &symTbl;

  // Cache some attributes
  const AttrCache &cache;

  const llvm::DenseMap<FModuleLike, Convention> &conventionTable;

  // Set true if the lowering failed.
  bool encounteredError = false;
};
} // namespace

/// Return aggregate preservation mode for the module. If the module has a
/// scalarized linkage, then we may not preserve it's aggregate ports.
PreserveAggregate::PreserveMode
TypeLoweringVisitor::getPreservationModeForModule(FModuleLike module) {
  auto lookup = conventionTable.find(module);
  if (lookup == conventionTable.end())
    return aggregatePreservationMode;
  switch (lookup->second) {
  case Convention::Scalarized:
    return PreserveAggregate::None;
  case Convention::Internal:
    return aggregatePreservationMode;
  }
  llvm_unreachable("Unknown convention");
  return aggregatePreservationMode;
}

Value TypeLoweringVisitor::getSubWhatever(Value val, size_t index) {
  if (type_isa<BundleType>(val.getType()))
    return builder->create<SubfieldOp>(val, index);
  if (type_isa<FVectorType>(val.getType()))
    return builder->create<SubindexOp>(val, index);
  if (type_isa<RefType>(val.getType()))
    return builder->create<RefSubOp>(val, index);
  llvm_unreachable("Unknown aggregate type");
  return nullptr;
}

/// Conditionally expand a subaccessop write path
bool TypeLoweringVisitor::processSAPath(Operation *op) {
  // Does this LHS have a subaccessop?
  SmallVector<Operation *> writePath = getSAWritePath(op);
  if (writePath.empty())
    return false;

  lowerSAWritePath(op, writePath);
  // Unhook the writePath from the connect.  This isn't the right type, but we
  // are deleting the op anyway.
  op->eraseOperands(0, 2);
  // See how far up the tree we can delete things.
  for (size_t i = 0; i < writePath.size(); ++i) {
    if (writePath[i]->use_empty()) {
      writePath[i]->erase();
    } else {
      break;
    }
  }
  return true;
}

void TypeLoweringVisitor::lowerBlock(Block *block) {
  // Lower the operations bottom up.
  for (auto it = block->rbegin(), e = block->rend(); it != e;) {
    auto &iop = *it;
    builder->setInsertionPoint(&iop);
    builder->setLoc(iop.getLoc());
    bool removeOp = dispatchVisitor(&iop);
    ++it;
    // Erase old ops eagerly so we don't have dangling uses we've already
    // lowered.
    if (removeOp)
      iop.erase();
  }
}

ArrayAttr TypeLoweringVisitor::filterAnnotations(MLIRContext *ctxt,
                                                 ArrayAttr annotations,
                                                 FIRRTLType srcType,
                                                 FlatBundleFieldEntry field) {
  SmallVector<Attribute> retval;
  if (!annotations || annotations.empty())
    return ArrayAttr::get(ctxt, retval);
  for (auto opAttr : annotations) {
    Annotation anno(opAttr);
    auto fieldID = anno.getFieldID();
    anno.removeMember("circt.fieldID");

    // If no fieldID set, or points to root, forward the annotation without the
    // fieldID field (which was removed above).
    if (fieldID == 0) {
      retval.push_back(anno.getAttr());
      continue;
    }
    // Check whether the annotation falls into the range of the current field.

    if (fieldID < field.fieldID ||
        fieldID > field.fieldID + field.type.getMaxFieldID())
      continue;

    // Add fieldID back if non-zero relative to this field.
    if (auto newFieldID = fieldID - field.fieldID) {
      // If the target is a subfield/subindex of the current field, create a
      // new annotation with the correct circt.fieldID.
      anno.setMember("circt.fieldID", builder->getI32IntegerAttr(newFieldID));
    }

    retval.push_back(anno.getAttr());
  }
  return ArrayAttr::get(ctxt, retval);
}

LogicalResult TypeLoweringVisitor::partitionSymbols(
    hw::InnerSymAttr sym, FIRRTLType parentType,
    SmallVectorImpl<hw::InnerSymAttr> &newSyms, Location errorLoc) {

  // No symbol, nothing to partition.
  if (!sym || sym.empty())
    return success();

  auto *context = sym.getContext();

  auto baseType = getBaseType(parentType);
  if (!baseType)
    return mlir::emitError(errorLoc,
                           "unable to partition symbol on unsupported type ")
           << parentType;

  return TypeSwitch<FIRRTLType, LogicalResult>(baseType)
      .Case<BundleType, FVectorType>([&](auto aggType) -> LogicalResult {
        struct BinningInfo {
          uint64_t index;
          uint64_t relFieldID;
          hw::InnerSymPropertiesAttr prop;
        };

        // Walk each inner symbol, compute binning information/assignment.
        SmallVector<BinningInfo> binning;
        for (auto prop : sym) {
          auto fieldID = prop.getFieldID();
          // Special-case fieldID == 0, helper methods require non-zero fieldID.
          if (fieldID == 0)
            return mlir::emitError(errorLoc, "unable to lower due to symbol ")
                   << prop.getName()
                   << " with target not preserved by lowering";
          auto [index, relFieldID] = aggType.getIndexAndSubfieldID(fieldID);
          binning.push_back({index, relFieldID, prop});
        }

        // Sort by index, fieldID.
        llvm::stable_sort(binning, [&](auto &lhs, auto &rhs) {
          return std::tuple(lhs.index, lhs.relFieldID) <
                 std::tuple(rhs.index, rhs.relFieldID);
        });
        assert(!binning.empty());

        // Populate newSyms, group all symbols on same index.
        newSyms.resize(aggType.getNumElements());
        for (auto binIt = binning.begin(), binEnd = binning.end();
             binIt != binEnd;) {
          auto curIndex = binIt->index;
          SmallVector<hw::InnerSymPropertiesAttr> propsForIndex;
          // Gather all adjacent symbols for this index.
          while (binIt != binEnd && binIt->index == curIndex) {
            propsForIndex.push_back(hw::InnerSymPropertiesAttr::get(
                context, binIt->prop.getName(), binIt->relFieldID,
                binIt->prop.getSymVisibility()));
            ++binIt;
          }

          assert(!newSyms[curIndex]);
          newSyms[curIndex] = hw::InnerSymAttr::get(context, propsForIndex);
        }
        return success();
      })
      .Default([&](auto ty) {
        return mlir::emitError(
                   errorLoc, "unable to partition symbol on unsupported type ")
               << ty;
      });
}

bool TypeLoweringVisitor::lowerProducer(
    Operation *op,
    llvm::function_ref<Value(const FlatBundleFieldEntry &, ArrayAttr)> clone,
    Type srcType) {

  if (!srcType)
    srcType = op->getResult(0).getType();
  auto srcFType = type_dyn_cast<FIRRTLType>(srcType);
  if (!srcFType)
    return false;
  SmallVector<FlatBundleFieldEntry, 8> fieldTypes;

  if (!peelType(srcFType, fieldTypes, aggregatePreservationMode))
    return false;

  SmallVector<Value> lowered;
  // Loop over the leaf aggregates.
  SmallString<16> loweredName;
  auto nameKindAttr = op->getAttrOfType<NameKindEnumAttr>(cache.nameKindAttr);

  if (auto nameAttr = op->getAttrOfType<StringAttr>(cache.nameAttr))
    loweredName = nameAttr.getValue();
  auto baseNameLen = loweredName.size();
  auto oldAnno = op->getAttr("annotations").dyn_cast_or_null<ArrayAttr>();

  SmallVector<hw::InnerSymAttr> fieldSyms(fieldTypes.size());
  if (auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(op)) {
    if (failed(partitionSymbols(symOp.getInnerSymAttr(), srcFType, fieldSyms,
                                symOp.getLoc()))) {
      encounteredError = true;
      return false;
    }
  }

  for (const auto &[field, sym] : llvm::zip_equal(fieldTypes, fieldSyms)) {
    if (!loweredName.empty()) {
      loweredName.resize(baseNameLen);
      loweredName += field.suffix;
    }

    // For all annotations on the parent op, filter them based on the target
    // attribute.
    ArrayAttr loweredAttrs =
        filterAnnotations(context, oldAnno, srcFType, field);
    auto newVal = clone(field, loweredAttrs);

    // If inner symbols on this field, add to new op.
    if (sym) {
      // Splitting up something with symbols on it should lower to ops
      // that also can have symbols on them.
      auto newSymOp = newVal.getDefiningOp<hw::InnerSymbolOpInterface>();
      assert(
          newSymOp &&
          "op with inner symbol lowered to op that cannot take inner symbol");
      newSymOp.setInnerSymbolAttr(sym);
    }

    // Carry over the name, if present.
    if (auto *newOp = newVal.getDefiningOp()) {
      if (!loweredName.empty())
        newOp->setAttr(cache.nameAttr, StringAttr::get(context, loweredName));
      if (nameKindAttr)
        newOp->setAttr(cache.nameKindAttr, nameKindAttr);
    }
    lowered.push_back(newVal);
  }

  processUsers(op->getResult(0), lowered);
  return true;
}

void TypeLoweringVisitor::processUsers(Value val, ArrayRef<Value> mapping) {
  for (auto *user : llvm::make_early_inc_range(val.getUsers())) {
    TypeSwitch<Operation *, void>(user)
        .Case<SubindexOp>([mapping](SubindexOp sio) {
          Value repl = mapping[sio.getIndex()];
          sio.replaceAllUsesWith(repl);
          sio.erase();
        })
        .Case<SubfieldOp>([mapping](SubfieldOp sfo) {
          // Get the input bundle type.
          Value repl = mapping[sfo.getFieldIndex()];
          sfo.replaceAllUsesWith(repl);
          sfo.erase();
        })
        .Case<RefSubOp>([mapping](RefSubOp refSub) {
          Value repl = mapping[refSub.getIndex()];
          refSub.replaceAllUsesWith(repl);
          refSub.erase();
        })
        .Default([&](auto op) {
          // This means we have already processed the user, and it didn't lower
          // its inputs. This is an opaque user, which will continue to have
          // aggregate type as input, even after LowerTypes. So, construct the
          // vector/bundle back from the lowered elements to ensure a valid
          // input into the opaque op. This only supports Bundles and Vectors.

          // This builder ensures that the aggregate construction happens at the
          // user location, and the LowerTypes algorithm will not touch them any
          // more, because LowerTypes was reverse iterating on the block and the
          // user has already been processed.
          ImplicitLocOpBuilder b(user->getLoc(), user);

          // This shouldn't happen (non-FIRRTLBaseType's in lowered types, or
          // refs), check explicitly here for clarity/early detection.
          assert(llvm::none_of(mapping, [](auto v) {
            auto fbasetype = type_dyn_cast<FIRRTLBaseType>(v.getType());
            return !fbasetype || fbasetype.containsReference();
          }));

          Value input =
              TypeSwitch<Type, Value>(val.getType())
                  .template Case<FVectorType>([&](auto vecType) {
                    return b.createOrFold<VectorCreateOp>(vecType, mapping);
                  })
                  .template Case<BundleType>([&](auto bundleType) {
                    return b.createOrFold<BundleCreateOp>(bundleType, mapping);
                  })
                  .Default([&](auto _) -> Value { return {}; });
          if (!input) {
            user->emitError("unable to reconstruct source of type ")
                << val.getType();
            encounteredError = true;
            return;
          }
          user->replaceUsesOfWith(val, input);
        });
  }
}

void TypeLoweringVisitor::lowerModule(FModuleLike op) {
  if (auto module = llvm::dyn_cast<FModuleOp>(*op))
    visitDecl(module);
  else if (auto extModule = llvm::dyn_cast<FExtModuleOp>(*op))
    visitDecl(extModule);
}

// Creates and returns a new block argument of the specified type to the
// module. This also maintains the name attribute for the new argument,
// possibly with a new suffix appended.
std::pair<Value, PortInfo>
TypeLoweringVisitor::addArg(Operation *module, unsigned insertPt,
                            unsigned insertPtOffset, FIRRTLType srcType,
                            FlatBundleFieldEntry field, PortInfo &oldArg,
                            hw::InnerSymAttr newSym) {
  Value newValue;
  FIRRTLType fieldType = mapBaseType(srcType, [&](auto) { return field.type; });
  if (auto mod = llvm::dyn_cast<FModuleOp>(module)) {
    Block *body = mod.getBodyBlock();
    // Append the new argument.
    newValue = body->insertArgument(insertPt, fieldType, oldArg.loc);
  }

  // Save the name attribute for the new argument.
  auto name = builder->getStringAttr(oldArg.name.getValue() + field.suffix);

  // Populate the new arg attributes.
  auto newAnnotations = filterAnnotations(
      context, oldArg.annotations.getArrayAttr(), srcType, field);
  // Flip the direction if the field is an output.
  auto direction = (Direction)((unsigned)oldArg.direction ^ field.isOutput);

  return std::make_pair(newValue,
                        PortInfo{name, fieldType, direction, newSym, oldArg.loc,
                                 AnnotationSet(newAnnotations)});
}

// Lower arguments with bundle type by flattening them.
bool TypeLoweringVisitor::lowerArg(FModuleLike module, size_t argIndex,
                                   size_t argsRemoved,
                                   SmallVectorImpl<PortInfo> &newArgs,
                                   SmallVectorImpl<Value> &lowering) {

  // Flatten any bundle types.
  SmallVector<FlatBundleFieldEntry> fieldTypes;
  auto srcType = type_cast<FIRRTLType>(newArgs[argIndex].type);
  if (!peelType(srcType, fieldTypes, getPreservationModeForModule(module)))
    return false;

  SmallVector<hw::InnerSymAttr> fieldSyms(fieldTypes.size());
  if (failed(partitionSymbols(newArgs[argIndex].sym, srcType, fieldSyms,
                              newArgs[argIndex].loc))) {
    encounteredError = true;
    return false;
  }

  for (const auto &[idx, field, fieldSym] :
       llvm::enumerate(fieldTypes, fieldSyms)) {
    auto newValue = addArg(module, 1 + argIndex + idx, argsRemoved, srcType,
                           field, newArgs[argIndex], fieldSym);
    newArgs.insert(newArgs.begin() + 1 + argIndex + idx, newValue.second);
    // Lower any other arguments by copying them to keep the relative order.
    lowering.push_back(newValue.first);
  }
  return true;
}

static Value cloneAccess(ImplicitLocOpBuilder *builder, Operation *op,
                         Value rhs) {
  if (auto rop = llvm::dyn_cast<SubfieldOp>(op))
    return builder->create<SubfieldOp>(rhs, rop.getFieldIndex());
  if (auto rop = llvm::dyn_cast<SubindexOp>(op))
    return builder->create<SubindexOp>(rhs, rop.getIndex());
  if (auto rop = llvm::dyn_cast<SubaccessOp>(op))
    return builder->create<SubaccessOp>(rhs, rop.getIndex());
  op->emitError("Unknown accessor");
  return nullptr;
}

void TypeLoweringVisitor::lowerSAWritePath(Operation *op,
                                           ArrayRef<Operation *> writePath) {
  SubaccessOp sao = cast<SubaccessOp>(writePath.back());
  FVectorType saoType = sao.getInput().getType();
  auto selectWidth = llvm::Log2_64_Ceil(saoType.getNumElements());

  for (size_t index = 0, e = saoType.getNumElements(); index < e; ++index) {
    auto cond = builder->create<EQPrimOp>(
        sao.getIndex(),
        builder->createOrFold<ConstantOp>(UIntType::get(context, selectWidth),
                                          APInt(selectWidth, index)));
    builder->create<WhenOp>(cond, false, [&]() {
      // Recreate the write Path
      Value leaf = builder->create<SubindexOp>(sao.getInput(), index);
      for (int i = writePath.size() - 2; i >= 0; --i) {
        if (auto access = cloneAccess(builder, writePath[i], leaf))
          leaf = access;
        else {
          encounteredError = true;
          return;
        }
      }

      emitConnect(*builder, leaf, op->getOperand(1));
    });
  }
}

// Expand connects of aggregates
bool TypeLoweringVisitor::visitStmt(ConnectOp op) {
  if (processSAPath(op))
    return true;

  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;

  // We have to expand connections even if the aggregate preservation is true.
  if (!peelType(op.getDest().getType(), fields, PreserveAggregate::None))
    return false;

  // Loop over the leaf aggregates.
  for (const auto &field : llvm::enumerate(fields)) {
    Value src = getSubWhatever(op.getSrc(), field.index());
    Value dest = getSubWhatever(op.getDest(), field.index());
    if (field.value().isOutput)
      std::swap(src, dest);
    emitConnect(*builder, dest, src);
  }
  return true;
}

// Expand connects of aggregates
bool TypeLoweringVisitor::visitStmt(StrictConnectOp op) {
  if (processSAPath(op))
    return true;

  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;

  // We have to expand connections even if the aggregate preservation is true.
  if (!peelType(op.getDest().getType(), fields, PreserveAggregate::None))
    return false;

  // Loop over the leaf aggregates.
  for (const auto &field : llvm::enumerate(fields)) {
    Value src = getSubWhatever(op.getSrc(), field.index());
    Value dest = getSubWhatever(op.getDest(), field.index());
    if (field.value().isOutput)
      std::swap(src, dest);
    builder->create<StrictConnectOp>(dest, src);
  }
  return true;
}

// Expand connects of references-of-aggregates
bool TypeLoweringVisitor::visitStmt(RefDefineOp op) {
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;

  if (!peelType(op.getDest().getType(), fields, aggregatePreservationMode))
    return false;

  // Loop over the leaf aggregates.
  for (const auto &field : llvm::enumerate(fields)) {
    Value src = getSubWhatever(op.getSrc(), field.index());
    Value dest = getSubWhatever(op.getDest(), field.index());
    assert(!field.value().isOutput && "unexpected flip in reftype destination");
    builder->create<RefDefineOp>(dest, src);
  }
  return true;
}

bool TypeLoweringVisitor::visitStmt(WhenOp op) {
  // The WhenOp itself does not require any lowering, the only value it uses
  // is a one-bit predicate.  Recursively visit all regions so internal
  // operations are lowered.

  // Visit operations in the then block.
  lowerBlock(&op.getThenBlock());

  // Visit operations in the else block.
  if (op.hasElseRegion())
    lowerBlock(&op.getElseBlock());
  return false; // don't delete the when!
}

/// Lower any types declared in the group definition.
bool TypeLoweringVisitor::visitStmt(GroupOp op) {
  lowerBlock(op.getBody());
  return false;
}

/// Lower memory operations. A new memory is created for every leaf
/// element in a memory's data type.
bool TypeLoweringVisitor::visitDecl(MemOp op) {
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;

  // MemOp should have ground types so we can't preserve aggregates.
  if (!peelType(op.getDataType(), fields, memoryPreservationMode))
    return false;

  if (op.getInnerSym()) {
    op->emitError() << "has a symbol, but no symbols may exist on aggregates "
                       "passed through LowerTypes";
    encounteredError = true;
    return false;
  }

  SmallVector<MemOp> newMemories;
  SmallVector<WireOp> oldPorts;

  // Wires for old ports
  for (unsigned int index = 0, end = op.getNumResults(); index < end; ++index) {
    auto result = op.getResult(index);
    if (op.getPortKind(index) == MemOp::PortKind::Debug) {
      op.emitOpError("cannot lower memory with debug port");
      encounteredError = true;
      return false;
    }
    auto wire = builder->create<WireOp>(
        result.getType(),
        (op.getName() + "_" + op.getPortName(index).getValue()).str());
    oldPorts.push_back(wire);
    result.replaceAllUsesWith(wire.getResult());
  }
  // If annotations targeting fields of an aggregate are present, we cannot
  // flatten the memory. It must be split into one memory per aggregate field.
  // Do not overwrite the pass flag!

  // Memory for each field
  for (const auto &field : fields) {
    auto newMemForField = cloneMemWithNewType(builder, op, field);
    if (!newMemForField) {
      op.emitError("failed cloning memory for field");
      encounteredError = true;
      return false;
    }
    newMemories.push_back(newMemForField);
  }
  // Hook up the new memories to the wires the old memory was replaced with.
  for (size_t index = 0, rend = op.getNumResults(); index < rend; ++index) {
    auto result = oldPorts[index].getResult();
    auto rType = type_cast<BundleType>(result.getType());
    for (size_t fieldIndex = 0, fend = rType.getNumElements();
         fieldIndex != fend; ++fieldIndex) {
      auto name = rType.getElement(fieldIndex).name.getValue();
      auto oldField = builder->create<SubfieldOp>(result, fieldIndex);
      // data and mask depend on the memory type which was split.  They can also
      // go both directions, depending on the port direction.
      if (name == "data" || name == "mask" || name == "wdata" ||
          name == "wmask" || name == "rdata") {
        for (const auto &field : fields) {
          auto realOldField = getSubWhatever(oldField, field.index);
          auto newField = getSubWhatever(
              newMemories[field.index].getResult(index), fieldIndex);
          if (rType.getElement(fieldIndex).isFlip)
            std::swap(realOldField, newField);
          emitConnect(*builder, newField, realOldField);
        }
      } else {
        for (auto mem : newMemories) {
          auto newField =
              builder->create<SubfieldOp>(mem.getResult(index), fieldIndex);
          emitConnect(*builder, newField, oldField);
        }
      }
    }
  }
  return true;
}

bool TypeLoweringVisitor::visitDecl(FExtModuleOp extModule) {
  ImplicitLocOpBuilder theBuilder(extModule.getLoc(), context);
  builder = &theBuilder;

  // Top level builder
  OpBuilder builder(context);

  // Lower the module block arguments.
  SmallVector<unsigned> argsToRemove;
  auto newArgs = extModule.getPorts();
  for (size_t argIndex = 0, argsRemoved = 0; argIndex < newArgs.size();
       ++argIndex) {
    SmallVector<Value> lowering;
    if (lowerArg(extModule, argIndex, argsRemoved, newArgs, lowering)) {
      argsToRemove.push_back(argIndex);
      ++argsRemoved;
    }
    // lowerArg might have invalidated any reference to newArgs, be careful
  }

  // Remove block args that have been lowered
  for (auto ii = argsToRemove.rbegin(), ee = argsToRemove.rend(); ii != ee;
       ++ii)
    newArgs.erase(newArgs.begin() + *ii);

  SmallVector<NamedAttribute, 8> newModuleAttrs;

  // Copy over any attributes that weren't original argument attributes.
  for (auto attr : extModule->getAttrDictionary())
    // Drop old "portNames", directions, and argument attributes.  These are
    // handled differently below.
    if (attr.getName() != "portDirections" && attr.getName() != "portNames" &&
        attr.getName() != "portTypes" && attr.getName() != "portAnnotations" &&
        attr.getName() != "portSyms" && attr.getName() != "portLocations")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute> newArgNames;
  SmallVector<Attribute, 8> newArgTypes;
  SmallVector<Attribute, 8> newArgSyms;
  SmallVector<Attribute, 8> newArgLocations;
  SmallVector<Attribute, 8> newArgAnnotations;

  for (auto &port : newArgs) {
    newArgDirections.push_back(port.direction);
    newArgNames.push_back(port.name);
    newArgTypes.push_back(TypeAttr::get(port.type));
    newArgSyms.push_back(port.sym);
    newArgLocations.push_back(port.loc);
    newArgAnnotations.push_back(port.annotations.getArrayAttr());
  }

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortDirections,
                     direction::packAttribute(context, newArgDirections)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortNames, builder.getArrayAttr(newArgNames)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortTypes, builder.getArrayAttr(newArgTypes)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortLocations, builder.getArrayAttr(newArgLocations)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortAnnotations, builder.getArrayAttr(newArgAnnotations)));

  // Update the module's attributes.
  extModule->setAttrs(newModuleAttrs);
  extModule.setPortSymbols(newArgSyms);
  return false;
}

bool TypeLoweringVisitor::visitDecl(FModuleOp module) {
  auto *body = module.getBodyBlock();

  ImplicitLocOpBuilder theBuilder(module.getLoc(), context);
  builder = &theBuilder;

  // Lower the operations.
  lowerBlock(body);

  // Lower the module block arguments.
  llvm::BitVector argsToRemove;
  auto newArgs = module.getPorts();
  for (size_t argIndex = 0, argsRemoved = 0; argIndex < newArgs.size();
       ++argIndex) {
    SmallVector<Value> lowerings;
    if (lowerArg(module, argIndex, argsRemoved, newArgs, lowerings)) {
      auto arg = module.getArgument(argIndex);
      processUsers(arg, lowerings);
      argsToRemove.push_back(true);
      ++argsRemoved;
    } else
      argsToRemove.push_back(false);
    // lowerArg might have invalidated any reference to newArgs, be careful
  }

  // Remove block args that have been lowered.
  body->eraseArguments(argsToRemove);
  for (auto deadArg = argsToRemove.find_last(); deadArg != -1;
       deadArg = argsToRemove.find_prev(deadArg))
    newArgs.erase(newArgs.begin() + deadArg);

  SmallVector<NamedAttribute, 8> newModuleAttrs;

  // Copy over any attributes that weren't original argument attributes.
  for (auto attr : module->getAttrDictionary())
    // Drop old "portNames", directions, and argument attributes.  These are
    // handled differently below.
    if (attr.getName() != "portNames" && attr.getName() != "portDirections" &&
        attr.getName() != "portTypes" && attr.getName() != "portAnnotations" &&
        attr.getName() != "portSyms" && attr.getName() != "portLocations")
      newModuleAttrs.push_back(attr);

  SmallVector<Direction> newArgDirections;
  SmallVector<Attribute> newArgNames;
  SmallVector<Attribute> newArgTypes;
  SmallVector<Attribute> newArgSyms;
  SmallVector<Attribute> newArgLocations;
  SmallVector<Attribute, 8> newArgAnnotations;
  for (auto &port : newArgs) {
    newArgDirections.push_back(port.direction);
    newArgNames.push_back(port.name);
    newArgTypes.push_back(TypeAttr::get(port.type));
    newArgSyms.push_back(port.sym);
    newArgLocations.push_back(port.loc);
    newArgAnnotations.push_back(port.annotations.getArrayAttr());
  }

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortDirections,
                     direction::packAttribute(context, newArgDirections)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortNames, builder->getArrayAttr(newArgNames)));

  newModuleAttrs.push_back(
      NamedAttribute(cache.sPortTypes, builder->getArrayAttr(newArgTypes)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortLocations, builder->getArrayAttr(newArgLocations)));

  newModuleAttrs.push_back(NamedAttribute(
      cache.sPortAnnotations, builder->getArrayAttr(newArgAnnotations)));

  // Update the module's attributes.
  module->setAttrs(newModuleAttrs);
  module.setPortSymbols(newArgSyms);
  return false;
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
bool TypeLoweringVisitor::visitDecl(WireOp op) {
  if (op.isForceable())
    return false;

  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    return builder
        ->create<WireOp>(field.type, "", NameKindEnum::DroppableName, attrs,
                         StringAttr{})
        .getResult();
  };
  return lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
bool TypeLoweringVisitor::visitDecl(RegOp op) {
  if (op.isForceable())
    return false;

  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    return builder
        ->create<RegOp>(field.type, op.getClockVal(), "",
                        NameKindEnum::DroppableName, attrs, StringAttr{})
        .getResult();
  };
  return lowerProducer(op, clone);
}

/// Lower a reg op with a bundle to multiple non-bundled regs.
bool TypeLoweringVisitor::visitDecl(RegResetOp op) {
  if (op.isForceable())
    return false;

  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto resetVal = getSubWhatever(op.getResetValue(), field.index);
    return builder
        ->create<RegResetOp>(field.type, op.getClockVal(), op.getResetSignal(),
                             resetVal, "", NameKindEnum::DroppableName, attrs,
                             StringAttr{})
        .getResult();
  };
  return lowerProducer(op, clone);
}

/// Lower a wire op with a bundle to multiple non-bundled wires.
bool TypeLoweringVisitor::visitDecl(NodeOp op) {
  if (op.isForceable())
    return false;

  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto input = getSubWhatever(op.getInput(), field.index);
    return builder
        ->create<NodeOp>(input, "", NameKindEnum::DroppableName, attrs)
        .getResult();
  };
  return lowerProducer(op, clone);
}

/// Lower an InvalidValue op with a bundle to multiple non-bundled InvalidOps.
bool TypeLoweringVisitor::visitExpr(InvalidValueOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    return builder->create<InvalidValueOp>(field.type);
  };
  return lowerProducer(op, clone);
}

// Expand muxes of aggregates
bool TypeLoweringVisitor::visitExpr(MuxPrimOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto high = getSubWhatever(op.getHigh(), field.index);
    auto low = getSubWhatever(op.getLow(), field.index);
    return builder->create<MuxPrimOp>(op.getSel(), high, low);
  };
  return lowerProducer(op, clone);
}

// Expand muxes of aggregates
bool TypeLoweringVisitor::visitExpr(Mux2CellIntrinsicOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto high = getSubWhatever(op.getHigh(), field.index);
    auto low = getSubWhatever(op.getLow(), field.index);
    return builder->create<Mux2CellIntrinsicOp>(op.getSel(), high, low);
  };
  return lowerProducer(op, clone);
}

// Expand muxes of aggregates
bool TypeLoweringVisitor::visitExpr(Mux4CellIntrinsicOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto v3 = getSubWhatever(op.getV3(), field.index);
    auto v2 = getSubWhatever(op.getV2(), field.index);
    auto v1 = getSubWhatever(op.getV1(), field.index);
    auto v0 = getSubWhatever(op.getV0(), field.index);
    return builder->create<Mux4CellIntrinsicOp>(op.getSel(), v3, v2, v1, v0);
  };
  return lowerProducer(op, clone);
}

// Expand UnrealizedConversionCastOp of aggregates
bool TypeLoweringVisitor::visitExpr(mlir::UnrealizedConversionCastOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto input = getSubWhatever(op.getOperand(0), field.index);
    return builder->create<mlir::UnrealizedConversionCastOp>(field.type, input)
        .getResult(0);
  };
  return lowerProducer(op, clone);
}

// Expand BitCastOp of aggregates
bool TypeLoweringVisitor::visitExpr(BitCastOp op) {
  Value srcLoweredVal = op.getInput();
  // If the input is of aggregate type, then cat all the leaf fields to form a
  // UInt type result. That is, first bitcast the aggregate type to a UInt.
  // Attempt to get the bundle types.
  SmallVector<FlatBundleFieldEntry> fields;
  if (peelType(op.getInput().getType(), fields, PreserveAggregate::None)) {
    size_t uptoBits = 0;
    // Loop over the leaf aggregates and concat each of them to get a UInt.
    // Bitcast the fields to handle nested aggregate types.
    for (const auto &field : llvm::enumerate(fields)) {
      auto fieldBitwidth = *getBitWidth(field.value().type);
      // Ignore zero width fields, like empty bundles.
      if (fieldBitwidth == 0)
        continue;
      Value src = getSubWhatever(op.getInput(), field.index());
      // The src could be an aggregate type, bitcast it to a UInt type.
      src = builder->createOrFold<BitCastOp>(
          UIntType::get(context, fieldBitwidth), src);
      // Take the first field, or else Cat the previous fields with this field.
      if (uptoBits == 0)
        srcLoweredVal = src;
      else
        srcLoweredVal = builder->create<CatPrimOp>(src, srcLoweredVal);
      // Record the total bits already accumulated.
      uptoBits += fieldBitwidth;
    }
  } else {
    srcLoweredVal = builder->createOrFold<AsUIntPrimOp>(srcLoweredVal);
  }
  // Now the input has been cast to srcLoweredVal, which is of UInt type.
  // If the result is an aggregate type, then use lowerProducer.
  if (type_isa<BundleType, FVectorType>(op.getResult().getType())) {
    // uptoBits is used to keep track of the bits that have been extracted.
    size_t uptoBits = 0;
    auto clone = [&](const FlatBundleFieldEntry &field,
                     ArrayAttr attrs) -> Value {
      // All the fields must have valid bitwidth, a requirement for BitCastOp.
      auto fieldBits = *getBitWidth(field.type);
      // If empty field, then it doesnot have any use, so replace it with an
      // invalid op, which should be trivially removed.
      if (fieldBits == 0)
        return builder->create<InvalidValueOp>(field.type);

      // Assign the field to the corresponding bits from the input.
      // Bitcast the field, incase its an aggregate type.
      auto extractBits = builder->create<BitsPrimOp>(
          srcLoweredVal, uptoBits + fieldBits - 1, uptoBits);
      uptoBits += fieldBits;
      return builder->create<BitCastOp>(field.type, extractBits);
    };
    return lowerProducer(op, clone);
  }

  // If ground type, then replace the result.
  if (type_isa<SIntType>(op.getType()))
    srcLoweredVal = builder->create<AsSIntPrimOp>(srcLoweredVal);
  op.getResult().replaceAllUsesWith(srcLoweredVal);
  return true;
}

bool TypeLoweringVisitor::visitExpr(RefSendOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    return builder->create<RefSendOp>(
        getSubWhatever(op.getBase(), field.index));
  };
  // Be careful re:what gets lowered, consider ref.send of non-passive
  // and whether we're using the ref or the base type to choose
  // whether this should be lowered.
  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitExpr(RefResolveOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    Value src = getSubWhatever(op.getRef(), field.index);
    return builder->create<RefResolveOp>(src);
  };
  // Lower according to lowering of the reference.
  // Particularly, preserve if rwprobe.
  return lowerProducer(op, clone, op.getRef().getType());
}

bool TypeLoweringVisitor::visitExpr(RefCastOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    auto input = getSubWhatever(op.getInput(), field.index);
    return builder->create<RefCastOp>(RefType::get(field.type), input);
  };
  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitDecl(InstanceOp op) {
  bool skip = true;
  SmallVector<Type, 8> resultTypes;
  SmallVector<int64_t, 8> endFields; // Compressed sparse row encoding
  auto oldPortAnno = op.getPortAnnotations();
  SmallVector<Direction> newDirs;
  SmallVector<Attribute> newNames;
  SmallVector<Attribute> newPortAnno;
  PreserveAggregate::PreserveMode mode =
      getPreservationModeForModule(op.getReferencedModule(symTbl));

  endFields.push_back(0);
  for (size_t i = 0, e = op.getNumResults(); i != e; ++i) {
    auto srcType = type_cast<FIRRTLType>(op.getType(i));

    // Flatten any nested bundle types the usual way.
    SmallVector<FlatBundleFieldEntry, 8> fieldTypes;
    if (!peelType(srcType, fieldTypes, mode)) {
      newDirs.push_back(op.getPortDirection(i));
      newNames.push_back(op.getPortName(i));
      resultTypes.push_back(srcType);
      newPortAnno.push_back(oldPortAnno[i]);
    } else {
      skip = false;
      auto oldName = op.getPortNameStr(i);
      auto oldDir = op.getPortDirection(i);
      // Store the flat type for the new bundle type.
      for (const auto &field : fieldTypes) {
        newDirs.push_back(direction::get((unsigned)oldDir ^ field.isOutput));
        newNames.push_back(builder->getStringAttr(oldName + field.suffix));
        resultTypes.push_back(
            mapBaseType(srcType, [&](auto base) { return field.type; }));
        auto annos = filterAnnotations(
            context, oldPortAnno[i].dyn_cast_or_null<ArrayAttr>(), srcType,
            field);
        newPortAnno.push_back(annos);
      }
    }
    endFields.push_back(resultTypes.size());
  }

  auto sym = getInnerSymName(op);

  if (skip) {
    return false;
  }

  // FIXME: annotation update
  auto newInstance = builder->create<InstanceOp>(
      resultTypes, op.getModuleNameAttr(), op.getNameAttr(),
      op.getNameKindAttr(), direction::packAttribute(context, newDirs),
      builder->getArrayAttr(newNames), op.getAnnotations(),
      builder->getArrayAttr(newPortAnno), op.getLowerToBindAttr(),
      sym ? hw::InnerSymAttr::get(sym) : hw::InnerSymAttr());

  // Copy over any attributes which have not already been copied over by
  // arguments to the builder.
  auto attrNames = InstanceOp::getAttributeNames();
  DenseSet<StringRef> attrSet(attrNames.begin(), attrNames.end());
  SmallVector<NamedAttribute> newAttrs(newInstance->getAttrs());
  for (auto i : llvm::make_filter_range(op->getAttrs(), [&](auto namedAttr) {
         return !attrSet.count(namedAttr.getName());
       }))
    newAttrs.push_back(i);
  newInstance->setAttrs(newAttrs);

  SmallVector<Value> lowered;
  for (size_t aggIndex = 0, eAgg = op.getNumResults(); aggIndex != eAgg;
       ++aggIndex) {
    lowered.clear();
    for (size_t fieldIndex = endFields[aggIndex],
                eField = endFields[aggIndex + 1];
         fieldIndex < eField; ++fieldIndex)
      lowered.push_back(newInstance.getResult(fieldIndex));
    if (lowered.size() != 1 ||
        op.getType(aggIndex) != resultTypes[endFields[aggIndex]])
      processUsers(op.getResult(aggIndex), lowered);
    else
      op.getResult(aggIndex).replaceAllUsesWith(lowered[0]);
  }
  return true;
}

bool TypeLoweringVisitor::visitExpr(SubaccessOp op) {
  auto input = op.getInput();
  FVectorType vType = input.getType();

  // Check for empty vectors
  if (vType.getNumElements() == 0) {
    Value inv = builder->create<InvalidValueOp>(vType.getElementType());
    op.replaceAllUsesWith(inv);
    return true;
  }

  // Check for constant instances
  if (ConstantOp arg =
          llvm::dyn_cast_or_null<ConstantOp>(op.getIndex().getDefiningOp())) {
    auto sio = builder->create<SubindexOp>(op.getInput(),
                                           arg.getValue().getExtValue());
    op.replaceAllUsesWith(sio.getResult());
    return true;
  }

  // Construct a multibit mux
  SmallVector<Value> inputs;
  inputs.reserve(vType.getNumElements());
  for (int index = vType.getNumElements() - 1; index >= 0; index--)
    inputs.push_back(builder->create<SubindexOp>(input, index));

  Value multibitMux = builder->create<MultibitMuxOp>(op.getIndex(), inputs);
  op.replaceAllUsesWith(multibitMux);
  return true;
}

bool TypeLoweringVisitor::visitExpr(VectorCreateOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    return op.getOperand(field.index);
  };
  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitExpr(BundleCreateOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    return op.getOperand(field.index);
  };
  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitExpr(ElementwiseOrPrimOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    Value operands[] = {getSubWhatever(op.getLhs(), field.index),
                        getSubWhatever(op.getRhs(), field.index)};
    return type_isa<BundleType, FVectorType>(field.type)
               ? (Value)builder->create<ElementwiseOrPrimOp>(field.type,
                                                             operands)
               : (Value)builder->create<OrPrimOp>(operands);
  };

  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitExpr(ElementwiseAndPrimOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    Value operands[] = {getSubWhatever(op.getLhs(), field.index),
                        getSubWhatever(op.getRhs(), field.index)};
    return type_isa<BundleType, FVectorType>(field.type)
               ? (Value)builder->create<ElementwiseAndPrimOp>(field.type,
                                                              operands)
               : (Value)builder->create<AndPrimOp>(operands);
  };

  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitExpr(ElementwiseXorPrimOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    Value operands[] = {getSubWhatever(op.getLhs(), field.index),
                        getSubWhatever(op.getRhs(), field.index)};
    return type_isa<BundleType, FVectorType>(field.type)
               ? (Value)builder->create<ElementwiseXorPrimOp>(field.type,
                                                              operands)
               : (Value)builder->create<XorPrimOp>(operands);
  };

  return lowerProducer(op, clone);
}

bool TypeLoweringVisitor::visitExpr(MultibitMuxOp op) {
  auto clone = [&](const FlatBundleFieldEntry &field,
                   ArrayAttr attrs) -> Value {
    SmallVector<Value> newInputs;
    newInputs.reserve(op.getInputs().size());
    for (auto input : op.getInputs()) {
      auto inputSub = getSubWhatever(input, field.index);
      newInputs.push_back(inputSub);
    }
    return builder->create<MultibitMuxOp>(op.getIndex(), newInputs);
  };
  return lowerProducer(op, clone);
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerTypesPass : public LowerFIRRTLTypesBase<LowerTypesPass> {
  LowerTypesPass(
      circt::firrtl::PreserveAggregate::PreserveMode preserveAggregateFlag,
      circt::firrtl::PreserveAggregate::PreserveMode preserveMemoriesFlag) {
    preserveAggregate = preserveAggregateFlag;
    preserveMemories = preserveMemoriesFlag;
  }
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerTypesPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===- Running LowerTypes Pass "
                      "------------------------------------------------===\n");
  std::vector<FModuleLike> ops;
  // Symbol Table
  auto &symTbl = getAnalysis<SymbolTable>();
  // Cached attr
  AttrCache cache(&getContext());

  DenseMap<FModuleLike, Convention> conventionTable;
  auto circuit = getOperation();
  for (auto module : circuit.getOps<FModuleLike>()) {
    conventionTable.insert({module, module.getConvention()});
    ops.push_back(module);
  }

  // This lambda, executes in parallel for each Op within the circt.
  auto lowerModules = [&](FModuleLike op) -> LogicalResult {
    auto tl =
        TypeLoweringVisitor(&getContext(), preserveAggregate, preserveMemories,
                            symTbl, cache, conventionTable);
    tl.lowerModule(op);

    return LogicalResult::failure(tl.isFailed());
  };

  auto result = failableParallelForEach(&getContext(), ops, lowerModules);

  if (failed(result))
    signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerFIRRTLTypesPass(
    PreserveAggregate::PreserveMode mode,
    PreserveAggregate::PreserveMode memoryMode) {
  return std::make_unique<LowerTypesPass>(mode, memoryMode);
}
