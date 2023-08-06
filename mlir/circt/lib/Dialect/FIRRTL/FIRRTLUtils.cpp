//===- FIRRTLUtils.cpp - FIRRTL IR Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilties to help generate and process FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

void circt::firrtl::emitConnect(OpBuilder &builder, Location loc, Value dst,
                                Value src) {
  ImplicitLocOpBuilder locBuilder(loc, builder.getInsertionBlock(),
                                  builder.getInsertionPoint());
  emitConnect(locBuilder, dst, src);
  builder.restoreInsertionPoint(locBuilder.saveInsertionPoint());
}

/// Emit a connect between two values.
void circt::firrtl::emitConnect(ImplicitLocOpBuilder &builder, Value dst,
                                Value src) {
  auto dstFType = type_cast<FIRRTLType>(dst.getType());
  auto srcFType = type_cast<FIRRTLType>(src.getType());
  auto dstType = type_dyn_cast<FIRRTLBaseType>(dstFType);
  auto srcType = type_dyn_cast<FIRRTLBaseType>(srcFType);
  // Special Connects (non-base, foreign):
  if (!dstType) {
    // References use ref.define.  Add cast if types don't match.
    if (type_isa<RefType>(dstFType)) {
      if (dstFType != srcFType)
        src = builder.create<RefCastOp>(dstFType, src);
      builder.create<RefDefineOp>(dst, src);
    } else // Other types, give up and leave a connect
      builder.create<ConnectOp>(dst, src);
    return;
  }

  // If the types are the exact same we can just connect them.
  if (dstType == srcType && dstType.isPassive() &&
      !dstType.hasUninferredWidth()) {
    builder.create<StrictConnectOp>(dst, src);
    return;
  }

  if (auto dstBundle = type_dyn_cast<BundleType>(dstType)) {
    // Connect all the bundle elements pairwise.
    auto numElements = dstBundle.getNumElements();
    // Check if we are trying to create an illegal connect - just create the
    // connect and let the verifier catch it.
    auto srcBundle = type_dyn_cast<BundleType>(srcType);
    if (!srcBundle || numElements != srcBundle.getNumElements()) {
      builder.create<ConnectOp>(dst, src);
      return;
    }
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubfieldOp>(dst, i);
      auto srcField = builder.create<SubfieldOp>(src, i);
      if (dstBundle.getElement(i).isFlip)
        std::swap(dstField, srcField);
      emitConnect(builder, dstField, srcField);
    }
    return;
  }

  if (auto dstVector = type_dyn_cast<FVectorType>(dstType)) {
    // Connect all the vector elements pairwise.
    auto numElements = dstVector.getNumElements();
    // Check if we are trying to create an illegal connect - just create the
    // connect and let the verifier catch it.
    auto srcVector = type_dyn_cast<FVectorType>(srcType);
    if (!srcVector || numElements != srcVector.getNumElements()) {
      builder.create<ConnectOp>(dst, src);
      return;
    }
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubindexOp>(dst, i);
      auto srcField = builder.create<SubindexOp>(src, i);
      emitConnect(builder, dstField, srcField);
    }
    return;
  }

  if ((dstType.hasUninferredReset() || srcType.hasUninferredReset()) &&
      dstType != srcType) {
    srcType = dstType.getConstType(srcType.isConst());
    src = builder.create<UninferredResetCastOp>(srcType, src);
  }

  // Handle ground types with possibly uninferred widths.
  auto dstWidth = dstType.getBitWidthOrSentinel();
  auto srcWidth = srcType.getBitWidthOrSentinel();
  if (dstWidth < 0 || srcWidth < 0) {
    // If one of these types has an uninferred width, we connect them with a
    // regular connect operation.

    // Const-cast as needed, using widthless version of dest.
    // (dest is either widthless already, or source is and if the types
    //  can be const-cast'd, do so)
    assert(srcType.isGround() && dstType.isGround());
    if (dstType != srcType && dstType.getWidthlessType() != srcType &&
        areTypesConstCastable(dstType.getWidthlessType(), srcType)) {
      src = builder.create<ConstCastOp>(dstType.getWidthlessType(), src);
    }

    builder.create<ConnectOp>(dst, src);
    return;
  }

  // The source must be extended or truncated.
  if (dstWidth < srcWidth) {
    // firrtl.tail always returns uint even for sint operands.
    IntType tmpType =
        type_cast<IntType>(dstType).getConstType(srcType.isConst());
    bool isSignedDest = tmpType.isSigned();
    if (isSignedDest)
      tmpType =
          UIntType::get(dstType.getContext(), dstWidth, srcType.isConst());
    src = builder.create<TailPrimOp>(tmpType, src, srcWidth - dstWidth);
    // Insert the cast back to signed if needed.
    if (isSignedDest)
      src = builder.create<AsSIntPrimOp>(
          dstType.getConstType(tmpType.isConst()), src);
  } else if (srcWidth < dstWidth) {
    // Need to extend arg.
    src = builder.create<PadPrimOp>(src, dstWidth);
  }

  if (auto srcType = type_cast<FIRRTLBaseType>(src.getType());
      srcType && dstType != srcType &&
      areTypesConstCastable(dstType, srcType)) {
    src = builder.create<ConstCastOp>(dstType, src);
  }

  // Strict connect requires the types to be completely equal, including
  // connecting uint<1> to abstract reset types.
  if (dstType == src.getType() && dstType.isPassive() &&
      !dstType.hasUninferredWidth()) {
    builder.create<StrictConnectOp>(dst, src);
  } else
    builder.create<ConnectOp>(dst, src);
}

IntegerAttr circt::firrtl::getIntAttr(Type type, const APInt &value) {
  auto intType = type_cast<IntType>(type);
  assert((!intType.hasWidth() ||
          (unsigned)intType.getWidthOrSentinel() == value.getBitWidth()) &&
         "value / type width mismatch");
  auto intSign =
      intType.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
  auto attrType =
      IntegerType::get(type.getContext(), value.getBitWidth(), intSign);
  return IntegerAttr::get(attrType, value);
}

/// Return an IntegerAttr filled with zeros for the specified FIRRTL integer
/// type. This handles both the known width and unknown width case.
IntegerAttr circt::firrtl::getIntZerosAttr(Type type) {
  int32_t width = abs(type_cast<IntType>(type).getWidthOrSentinel());
  return getIntAttr(type, APInt(width, 0));
}

/// Return an IntegerAttr filled with ones for the specified FIRRTL integer
/// type. This handles both the known width and unknown width case.
IntegerAttr circt::firrtl::getIntOnesAttr(Type type) {
  int32_t width = abs(type_cast<IntType>(type).getWidthOrSentinel());
  return getIntAttr(type, APInt(width, -1));
}

/// Return the single assignment to a Property value. It is assumed that the
/// single assigment invariant is enforced elsewhere.
PropAssignOp circt::firrtl::getPropertyAssignment(FIRRTLPropertyValue value) {
  for (auto *user : value.getUsers())
    if (auto propassign = dyn_cast<PropAssignOp>(user))
      if (propassign.getDest() == value)
        return propassign;

  // The invariant that there is a single assignment should be enforced
  // elsewhere. If for some reason a user called this on a Property value that
  // is not assigned (like a module input port), just return null.
  return nullptr;
}

/// Return the value that drives another FIRRTL value within module scope.  Only
/// look backwards through one connection.  This is intended to be used in
/// situations where you only need to look at the most recent connect, e.g., to
/// know if a wire has been driven to a constant.  Return null if no driver via
/// a connect was found.
Value circt::firrtl::getDriverFromConnect(Value val) {
  for (auto *user : val.getUsers()) {
    if (auto connect = dyn_cast<FConnectLike>(user)) {
      if (connect.getDest() != val)
        continue;
      return connect.getSrc();
    }
  }
  return nullptr;
}

Value circt::firrtl::getModuleScopedDriver(Value val, bool lookThroughWires,
                                           bool lookThroughNodes,
                                           bool lookThroughCasts) {
  // Update `val` to the source of the connection driving `thisVal`.  This walks
  // backwards across users to find the first connection and updates `val` to
  // the source.  This assumes that only one connect is driving `thisVal`, i.e.,
  // this pass runs after `ExpandWhens`.
  auto updateVal = [&](Value thisVal) {
    for (auto *user : thisVal.getUsers()) {
      if (auto connect = dyn_cast<FConnectLike>(user)) {
        if (connect.getDest() != val)
          continue;
        val = connect.getSrc();
        return;
      }
    }
    val = nullptr;
    return;
  };

  while (val) {
    // The value is a port.
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      FModuleOp op = cast<FModuleOp>(val.getParentBlock()->getParentOp());
      auto direction = op.getPortDirection(blockArg.getArgNumber());
      // Base case: this is one of the module's input ports.
      if (direction == Direction::In)
        return blockArg;
      updateVal(blockArg);
      continue;
    }

    auto *op = val.getDefiningOp();

    // The value is an instance port.
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto resultNo = cast<OpResult>(val).getResultNumber();
      // Base case: this is an instance's output port.
      if (inst.getPortDirection(resultNo) == Direction::Out)
        return inst.getResult(resultNo);
      updateVal(val);
      continue;
    }

    // If told to look through wires, continue from the driver of the wire.
    if (lookThroughWires && isa<WireOp>(op)) {
      updateVal(op->getResult(0));
      continue;
    }

    // If told to look through nodes, continue from the node input.
    if (lookThroughNodes && isa<NodeOp>(op)) {
      val = cast<NodeOp>(op).getInput();
      continue;
    }

    if (lookThroughCasts &&
        isa<AsUIntPrimOp, AsSIntPrimOp, AsClockPrimOp, AsAsyncResetPrimOp>(
            op)) {
      val = op->getOperand(0);
      continue;
    }

    // Look through unary ops generated by emitConnect
    if (isa<PadPrimOp, TailPrimOp>(op)) {
      val = op->getOperand(0);
      continue;
    }

    // Base case: this is a constant/invalid or primop.
    //
    // TODO: If needed, this could be modified to look through unary ops which
    // have an unambiguous single driver.  This should only be added if a need
    // arises for it.
    break;
  };
  return val;
}

bool circt::firrtl::walkDrivers(FIRRTLBaseValue value, bool lookThroughWires,
                                bool lookThroughNodes, bool lookThroughCasts,
                                WalkDriverCallback callback) {
  // TODO: what do we want to happen when there are flips in the type? Do we
  // want to filter out fields which have reverse flow?
  assert(value.getType().isPassive() && "this code was not tested with flips");

  // This method keeps a stack of wires (or ports) and subfields of those that
  // it still has to process.  It keeps track of which fields in the
  // destination are attached to which fields of the source, as well as which
  // subfield of the source we are currently investigating.  The fieldID is
  // used to filter which subfields of the current operation which we should
  // visit. As an example, the src might be an aggregate wire, but the current
  // value might be a subfield of that wire. The `src` FieldRef will represent
  // all subaccesses to the target, but `fieldID` for the current op only needs
  // to represent the all subaccesses between the current op and the target.
  struct StackElement {
    StackElement(FieldRef dst, FieldRef src, Value current, unsigned fieldID)
        : dst(dst), src(src), current(current), it(current.user_begin()),
          fieldID(fieldID) {}
    // The elements of the destination that this refers to.
    FieldRef dst;
    // The elements of the source that this refers to.
    FieldRef src;

    // These next fields are tied to the value we are currently iterating. This
    // is used so we can check if a connect op is reading or driving from this
    // value.
    Value current;
    // An iterator of the users of the current value. An end() iterator can be
    // constructed from the `current` value.
    Value::user_iterator it;
    // A filter for which fields of the current value we care about.
    unsigned fieldID;
  };
  SmallVector<StackElement> workStack;

  // Helper to add record a new wire to be processed in the worklist.  This will
  // add the wire itself to the worklist, which will lead to all subaccesses
  // being eventually processed as well.
  auto addToWorklist = [&](FieldRef dst, FieldRef src) {
    auto value = src.getValue();
    workStack.emplace_back(dst, src, value, src.getFieldID());
  };

  // Create an initial fieldRef from the input value.  As a starting state, the
  // dst and src are the same value.
  auto original = getFieldRefFromValue(value);
  auto fieldRef = original;

  // This loop wraps the worklist, which processes wires. Initially the worklist
  // is empty.
  while (true) {
    // This loop looks through simple operations like casts and nodes.  If it
    // encounters a wire it will stop and add the wire to the worklist.
    while (true) {
      auto val = fieldRef.getValue();

      // The value is a port.
      if (auto blockArg = dyn_cast<BlockArgument>(val)) {
        auto *parent = val.getParentBlock()->getParentOp();
        auto module = cast<FModuleLike>(parent);
        auto direction = module.getPortDirection(blockArg.getArgNumber());
        // Base case: this is one of the module's input ports.
        if (direction == Direction::In) {
          if (!callback(original, fieldRef))
            return false;
          break;
        }
        addToWorklist(original, fieldRef);
        break;
      }

      auto *op = val.getDefiningOp();

      // The value is an instance port.
      if (auto inst = dyn_cast<InstanceOp>(op)) {
        auto resultNo = cast<OpResult>(val).getResultNumber();
        // Base case: this is an instance's output port.
        if (inst.getPortDirection(resultNo) == Direction::Out) {
          if (!callback(original, fieldRef))
            return false;
          break;
        }
        addToWorklist(original, fieldRef);
        break;
      }

      // If told to look through wires, continue from the driver of the wire.
      if (lookThroughWires && isa<WireOp>(op)) {
        addToWorklist(original, fieldRef);
        break;
      }

      // If told to look through nodes, continue from the node input.
      if (lookThroughNodes && isa<NodeOp>(op)) {
        auto input = cast<NodeOp>(op).getInput();
        auto next = getFieldRefFromValue(input);
        fieldRef = next.getSubField(fieldRef.getFieldID());
        continue;
      }

      // If told to look through casts, continue from the cast input.
      if (lookThroughCasts &&
          isa<AsUIntPrimOp, AsSIntPrimOp, AsClockPrimOp, AsAsyncResetPrimOp>(
              op)) {
        auto input = op->getOperand(0);
        auto next = getFieldRefFromValue(input);
        fieldRef = next.getSubField(fieldRef.getFieldID());
        continue;
      }

      // Look through unary ops generated by emitConnect.
      if (isa<PadPrimOp, TailPrimOp>(op)) {
        auto input = op->getOperand(0);
        auto next = getFieldRefFromValue(input);
        fieldRef = next.getSubField(fieldRef.getFieldID());
        continue;
      }

      // Base case: this is a constant/invalid or primop.
      //
      // TODO: If needed, this could be modified to look through unary ops which
      // have an unambiguous single driver.  This should only be added if a need
      // arises for it.
      if (!callback(original, fieldRef))
        return false;
      break;
    }

    // Process the next element on the stack.
    while (true) {
      // If there is nothing left in the workstack, we are done.
      if (workStack.empty())
        return true;
      auto &back = workStack.back();
      auto current = back.current;
      // Pop the current element if we have processed all users.
      if (back.it == current.user_end()) {
        workStack.pop_back();
        continue;
      }

      original = back.dst;
      fieldRef = back.src;
      auto *user = *back.it++;
      auto fieldID = back.fieldID;

      if (auto subfield = dyn_cast<SubfieldOp>(user)) {
        BundleType bundleType = subfield.getInput().getType();
        auto index = subfield.getFieldIndex();
        auto subID = bundleType.getFieldID(index);
        // If the index of this operation doesn't match the target, skip it.
        if (fieldID && index != bundleType.getIndexForFieldID(fieldID))
          continue;
        auto subRef = fieldRef.getSubField(subID);
        auto subOriginal = original.getSubField(subID);
        auto value = subfield.getResult();
        workStack.emplace_back(subOriginal, subRef, value, fieldID - subID);
      } else if (auto subindex = dyn_cast<SubindexOp>(user)) {
        FVectorType vectorType = subindex.getInput().getType();
        auto index = subindex.getIndex();
        auto subID = vectorType.getFieldID(index);
        // If the index of this operation doesn't match the target, skip it.
        if (fieldID && index != vectorType.getIndexForFieldID(fieldID))
          continue;
        auto subRef = fieldRef.getSubField(subID);
        auto subOriginal = original.getSubField(subID);
        auto value = subindex.getResult();
        workStack.emplace_back(subOriginal, subRef, value, fieldID - subID);
      } else if (auto connect = dyn_cast<FConnectLike>(user)) {
        // Make sure that this connect is driving the value.
        if (connect.getDest() != current)
          continue;
        // If the value is driven by a connect, we don't have to recurse,
        // just update the current value.
        fieldRef = getFieldRefFromValue(connect.getSrc());
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// FieldRef helpers
//===----------------------------------------------------------------------===//

FieldRef circt::firrtl::getFieldRefFromValue(Value value) {
  // This code walks upwards from the subfield and calculates the field ID at
  // each level. At each stage, it must take the current id, and re-index it as
  // a nested bundle under the parent field.. This is accomplished by using the
  // parent field's ID as a base, and adding the field ID of the child.
  unsigned id = 0;
  while (value) {
    Operation *op = value.getDefiningOp();

    // If this is a block argument, we are done.
    if (!op)
      break;

    auto handled =
        TypeSwitch<Operation *, bool>(op)
            .Case<SubfieldOp, OpenSubfieldOp>([&](auto subfieldOp) {
              value = subfieldOp.getInput();
              typename decltype(subfieldOp)::InputType bundleType =
                  subfieldOp.getInput().getType();
              // Rebase the current index on the parent field's
              // index.
              id += bundleType.getFieldID(subfieldOp.getFieldIndex());
              return true;
            })
            .Case<SubindexOp, OpenSubindexOp>([&](auto subindexOp) {
              value = subindexOp.getInput();
              typename decltype(subindexOp)::InputType vecType =
                  subindexOp.getInput().getType();
              // Rebase the current index on the parent field's
              // index.
              id += vecType.getFieldID(subindexOp.getIndex());
              return true;
            })
            .Case<RefSubOp>([&](RefSubOp refSubOp) {
              value = refSubOp.getInput();
              auto refInputType = refSubOp.getInput().getType();
              id += FIRRTLTypeSwitch<FIRRTLBaseType, size_t>(
                        refInputType.getType())
                        .Case<FVectorType, BundleType>([&](auto type) {
                          return type.getFieldID(refSubOp.getIndex());
                        });
              return true;
            })
            .Default(false);
    if (!handled)
      break;
  }
  return {value, id};
}

/// Get the string name of a value which is a direct child of a declaration op.
static void getDeclName(Value value, SmallString<64> &string, bool nameSafe) {
  // Treat the value as a worklist to allow for recursion.
  while (value) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      // Get the module ports and get the name.
      auto *op = arg.getOwner()->getParentOp();
      TypeSwitch<Operation *>(op).Case<FModuleOp, ClassOp>([&](auto op) {
        auto name = cast<StringAttr>(op.getPortNames()[arg.getArgNumber()]);
        string += name.getValue();
      });
      return;
    }

    auto *op = value.getDefiningOp();
    TypeSwitch<Operation *>(op)
        .Case<InstanceOp, MemOp>([&](auto op) {
          string += op.getName();
          string += nameSafe ? "_" : ".";
          string += op.getPortName(cast<OpResult>(value).getResultNumber())
                        .getValue();
          value = nullptr;
        })
        .Case<FNamableOp>([&](auto op) {
          string += op.getName();
          value = nullptr;
        })
        .Case<mlir::UnrealizedConversionCastOp>(
            [&](mlir::UnrealizedConversionCastOp cast) {
              // Forward through 1:1 conversion cast ops.
              if (cast.getNumResults() == 1 && cast.getNumOperands() == 1 &&
                  cast.getResult(0).getType() == cast.getOperand(0).getType()) {
                value = cast.getInputs()[0];
              } else {
                // Can't name this.
                string.clear();
                value = nullptr;
              }
            })
        .Default([&](auto) {
          // Can't name this.
          string.clear();
          value = nullptr;
        });
  }
}

std::pair<std::string, bool>
circt::firrtl::getFieldName(const FieldRef &fieldRef, bool nameSafe) {
  SmallString<64> name;
  auto value = fieldRef.getValue();
  getDeclName(value, name, nameSafe);
  bool rootKnown = !name.empty();

  auto type = value.getType();
  auto localID = fieldRef.getFieldID();
  while (localID) {
    // Index directly into ref inner type.
    if (auto refTy = type_dyn_cast<RefType>(type))
      type = refTy.getType();

    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      auto index = bundleType.getIndexForFieldID(localID);
      // Add the current field string, and recurse into a subfield.
      auto &element = bundleType.getElements()[index];
      if (!name.empty())
        name += nameSafe ? "_" : ".";
      name += element.name.getValue();
      // Recurse in to the element type.
      type = element.type;
      localID = localID - bundleType.getFieldID(index);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      auto index = vecType.getIndexForFieldID(localID);
      name += nameSafe ? "_" : "[";
      name += std::to_string(index);
      if (!nameSafe)
        name += "]";
      // Recurse in to the element type.
      type = vecType.getElementType();
      localID = localID - vecType.getFieldID(index);
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      auto index = enumType.getIndexForFieldID(localID);
      auto &element = enumType.getElements()[index];
      name += nameSafe ? "_" : ".";
      name += element.name.getValue();
      type = element.type;
      localID = localID - enumType.getFieldID(index);
    } else {
      // If we reach here, the field ref is pointing inside some aggregate type
      // that isn't a bundle or a vector. If the type is a ground type, then the
      // localID should be 0 at this point, and we should have broken from the
      // loop.
      llvm_unreachable("unsupported type");
    }
  }

  return {name.str().str(), rootKnown};
}

/// This gets the value targeted by a field id.  If the field id is targeting
/// the value itself, it returns it unchanged. If it is targeting a single field
/// in a aggregate value, such as a bundle or vector, this will create the
/// necessary subaccesses to get the value.
Value circt::firrtl::getValueByFieldID(ImplicitLocOpBuilder builder,
                                       Value value, unsigned fieldID) {
  // When the fieldID hits 0, we've found the target value.
  while (fieldID != 0) {
    FIRRTLTypeSwitch<Type, void>(value.getType())
        .Case<BundleType, OpenBundleType>([&](auto bundle) {
          auto index = bundle.getIndexForFieldID(fieldID);
          value = builder.create<SubfieldOp>(value, index);
          fieldID -= bundle.getFieldID(index);
        })
        .Case<FVectorType, OpenVectorType>([&](auto vector) {
          auto index = vector.getIndexForFieldID(fieldID);
          value = builder.create<SubindexOp>(value, index);
          fieldID -= vector.getFieldID(index);
        })
        .Case<RefType>([&](auto reftype) {
          FIRRTLTypeSwitch<FIRRTLBaseType, void>(reftype.getType())
              .template Case<BundleType, FVectorType>([&](auto type) {
                auto index = type.getIndexForFieldID(fieldID);
                value = builder.create<RefSubOp>(value, index);
                fieldID -= type.getFieldID(index);
              })
              .Default([&](auto _) {
                llvm::report_fatal_error(
                    "unrecognized type for indexing through with fieldID");
              });
        })
        // TODO: Plumb error case out and handle in callers.
        .Default([&](auto _) {
          llvm::report_fatal_error(
              "unrecognized type for indexing through with fieldID");
        });
  }
  return value;
}

/// Walk leaf ground types in the `firrtlType` and apply the function `fn`.
/// The first argument of `fn` is field ID, and the second argument is a
/// leaf ground type.
void circt::firrtl::walkGroundTypes(
    FIRRTLType firrtlType,
    llvm::function_ref<void(uint64_t, FIRRTLBaseType)> fn) {
  auto type = getBaseType(firrtlType);

  // If this is not a base type, return.
  if (!type)
    return;

  // If this is a ground type, don't call recursive functions.
  if (type.isGround())
    return fn(0, type);

  uint64_t fieldID = 0;
  auto recurse = [&](auto &&f, FIRRTLBaseType type) -> void {
    FIRRTLTypeSwitch<FIRRTLBaseType>(type)
        .Case<BundleType>([&](BundleType bundle) {
          for (size_t i = 0, e = bundle.getNumElements(); i < e; ++i) {
            fieldID++;
            f(f, bundle.getElementType(i));
          }
        })
        .template Case<FVectorType>([&](FVectorType vector) {
          for (size_t i = 0, e = vector.getNumElements(); i < e; ++i) {
            fieldID++;
            f(f, vector.getElementType());
          }
        })
        .template Case<FEnumType>([&](FEnumType fenum) {
          for (size_t i = 0, e = fenum.getNumElements(); i < e; ++i) {
            fieldID++;
            f(f, fenum.getElementType(i));
          }
        })
        .Default([&](FIRRTLBaseType groundType) {
          assert(groundType.isGround() &&
                 "only ground types are expected here");
          fn(fieldID, groundType);
        });
  };
  recurse(recurse, type);
}

/// Returns an operation's `inner_sym`, adding one if necessary.
StringAttr circt::firrtl::getOrAddInnerSym(
    const hw::InnerSymTarget &target,
    llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace) {

  // Return InnerSymAttr with sym on specified fieldID.
  auto getOrAdd = [&](auto mod, hw::InnerSymAttr attr,
                      auto fieldID) -> std::pair<hw::InnerSymAttr, StringAttr> {
    assert(mod);
    auto *context = mod.getContext();

    SmallVector<hw::InnerSymPropertiesAttr> props;
    if (attr) {
      // If already present, return it.
      if (auto sym = attr.getSymIfExists(fieldID))
        return {attr, sym};
      llvm::append_range(props, attr.getProps());
    }

    // Otherwise, create symbol and add to list.
    auto sym = StringAttr::get(context, getNamespace(mod).newName("sym"));
    props.push_back(hw::InnerSymPropertiesAttr::get(
        context, sym, fieldID, StringAttr::get(context, "public")));
    // TODO: store/ensure always sorted, insert directly, faster search.
    // For now, just be good and sort by fieldID.
    llvm::sort(props, [](auto &p, auto &q) {
      return p.getFieldID() < q.getFieldID();
    });
    return {hw::InnerSymAttr::get(context, props), sym};
  };

  if (target.isPort()) {
    if (auto mod = dyn_cast<FModuleOp>(target.getOp())) {
      auto portIdx = target.getPort();
      assert(portIdx < mod.getNumPorts());
      auto [attr, sym] =
          getOrAdd(mod, mod.getPortSymbolAttr(portIdx), target.getField());
      mod.setPortSymbolsAttr(portIdx, attr);
      return sym;
    }
  } else {
    // InnerSymbols only supported if op implements the interface.
    if (auto symOp = dyn_cast<hw::InnerSymbolOpInterface>(target.getOp())) {
      auto mod = symOp->getParentOfType<FModuleOp>();
      assert(mod);
      auto [attr, sym] =
          getOrAdd(mod, symOp.getInnerSymAttr(), target.getField());
      symOp.setInnerSymbolAttr(attr);
      return sym;
    }
  }

  assert(0 && "target must be port of FModuleOp or InnerSymbol");
  return {};
}

/// Obtain an inner reference to an operation, possibly adding an `inner_sym`
/// to that operation.
hw::InnerRefAttr circt::firrtl::getInnerRefTo(
    const hw::InnerSymTarget &target,
    llvm::function_ref<ModuleNamespace &(FModuleOp)> getNamespace) {
  auto mod = target.isPort() ? dyn_cast<FModuleOp>(target.getOp())
                             : target.getOp()->getParentOfType<FModuleOp>();
  assert(mod &&
         "must be an operation inside an FModuleOp or port of FModuleOp");
  return hw::InnerRefAttr::get(SymbolTable::getSymbolName(mod),
                               getOrAddInnerSym(target, getNamespace));
}

/// Parse a string that may encode a FIRRTL location into a LocationAttr.
std::pair<bool, std::optional<mlir::LocationAttr>>
circt::firrtl::maybeStringToLocation(StringRef spelling, bool skipParsing,
                                     StringAttr &locatorFilenameCache,
                                     FileLineColLoc &fileLineColLocCache,
                                     MLIRContext *context) {
  // The spelling of the token looks something like "@[Decoupled.scala 221:8]".
  if (!spelling.startswith("@[") || !spelling.endswith("]"))
    return {false, std::nullopt};

  spelling = spelling.drop_front(2).drop_back(1);

  // Decode the locator in "spelling", returning the filename and filling in
  // lineNo and colNo on success.  On failure, this returns an empty filename.
  auto decodeLocator = [&](StringRef input, unsigned &resultLineNo,
                           unsigned &resultColNo) -> StringRef {
    // Split at the last space.
    auto spaceLoc = input.find_last_of(' ');
    if (spaceLoc == StringRef::npos)
      return {};

    auto filename = input.take_front(spaceLoc);
    auto lineAndColumn = input.drop_front(spaceLoc + 1);

    // Decode the line/column.  If the colon is missing, then it will be empty
    // here.
    StringRef lineStr, colStr;
    std::tie(lineStr, colStr) = lineAndColumn.split(':');

    // Decode the line number and the column number if present.
    if (lineStr.getAsInteger(10, resultLineNo))
      return {};
    if (!colStr.empty()) {
      if (colStr.front() != '{') {
        if (colStr.getAsInteger(10, resultColNo))
          return {};
      } else {
        // compound locator, just parse the first part for now
        if (colStr.drop_front().split(',').first.getAsInteger(10, resultColNo))
          return {};
      }
    }
    return filename;
  };

  // Decode the locator spelling, reporting an error if it is malformed.
  unsigned lineNo = 0, columnNo = 0;
  StringRef filename = decodeLocator(spelling, lineNo, columnNo);
  if (filename.empty())
    return {false, std::nullopt};

  // If info locators are ignored, don't actually apply them.  We still do all
  // the verification above though.
  if (skipParsing)
    return {true, std::nullopt};

  /// Return an FileLineColLoc for the specified location, but use a bit of
  /// caching to reduce thrasing the MLIRContext.
  auto getFileLineColLoc = [&](StringRef filename, unsigned lineNo,
                               unsigned columnNo) -> FileLineColLoc {
    // Check our single-entry cache for this filename.
    StringAttr filenameId = locatorFilenameCache;
    if (filenameId.str() != filename) {
      // We missed!  Get the right identifier.
      locatorFilenameCache = filenameId = StringAttr::get(context, filename);

      // If we miss in the filename cache, we also miss in the FileLineColLoc
      // cache.
      return fileLineColLocCache =
                 FileLineColLoc::get(filenameId, lineNo, columnNo);
    }

    // If we hit the filename cache, check the FileLineColLoc cache.
    auto result = fileLineColLocCache;
    if (result && result.getLine() == lineNo && result.getColumn() == columnNo)
      return result;

    return fileLineColLocCache =
               FileLineColLoc::get(filenameId, lineNo, columnNo);
  };

  // Compound locators will be combined with spaces, like:
  //  @[Foo.scala 123:4 Bar.scala 309:14]
  // and at this point will be parsed as a-long-string-with-two-spaces at
  // 309:14.   We'd like to parse this into two things and represent it as an
  // MLIR fused locator, but we want to be conservatively safe for filenames
  // that have a space in it.  As such, we are careful to make sure we can
  // decode the filename/loc of the result.  If so, we accumulate results,
  // backward, in this vector.
  SmallVector<Location> extraLocs;
  auto spaceLoc = filename.find_last_of(' ');
  while (spaceLoc != StringRef::npos) {
    // Try decoding the thing before the space.  Validates that there is another
    // space and that the file/line can be decoded in that substring.
    unsigned nextLineNo = 0, nextColumnNo = 0;
    auto nextFilename =
        decodeLocator(filename.take_front(spaceLoc), nextLineNo, nextColumnNo);

    // On failure we didn't have a joined locator.
    if (nextFilename.empty())
      break;

    // On success, remember what we already parsed (Bar.Scala / 309:14), and
    // move on to the next chunk.
    auto loc =
        getFileLineColLoc(filename.drop_front(spaceLoc + 1), lineNo, columnNo);
    extraLocs.push_back(loc);
    filename = nextFilename;
    lineNo = nextLineNo;
    columnNo = nextColumnNo;
    spaceLoc = filename.find_last_of(' ');
  }

  mlir::LocationAttr result = getFileLineColLoc(filename, lineNo, columnNo);
  if (!extraLocs.empty()) {
    extraLocs.push_back(result);
    std::reverse(extraLocs.begin(), extraLocs.end());
    result = FusedLoc::get(context, extraLocs);
  }
  return {true, result};
}

/// Given a type, return the corresponding lowered type for the HW dialect.
/// Non-FIRRTL types are simply passed through. This returns a null type if it
/// cannot be lowered.
Type circt::firrtl::lowerType(
    Type type, std::optional<Location> loc,
    llvm::function_ref<hw::TypeAliasType(Type, BaseTypeAliasType, Location)>
        getTypeDeclFn) {
  auto firType = type_dyn_cast<FIRRTLBaseType>(type);
  if (!firType)
    return type;

  // If not known how to lower alias types, then ignore the alias.
  if (getTypeDeclFn)
    if (BaseTypeAliasType aliasType = dyn_cast<BaseTypeAliasType>(firType)) {
      if (!loc)
        loc = UnknownLoc::get(type.getContext());
      type = lowerType(aliasType.getInnerType(), loc, getTypeDeclFn);
      return getTypeDeclFn(type, aliasType, *loc);
    }
  // Ignore flip types.
  firType = firType.getPassiveType();

  if (auto bundle = type_dyn_cast<BundleType>(firType)) {
    mlir::SmallVector<hw::StructType::FieldInfo, 8> hwfields;
    for (auto element : bundle) {
      Type etype = lowerType(element.type, loc, getTypeDeclFn);
      if (!etype)
        return {};
      hwfields.push_back(hw::StructType::FieldInfo{element.name, etype});
    }
    return hw::StructType::get(type.getContext(), hwfields);
  }
  if (auto vec = type_dyn_cast<FVectorType>(firType)) {
    auto elemTy = lowerType(vec.getElementType(), loc, getTypeDeclFn);
    if (!elemTy)
      return {};
    return hw::ArrayType::get(elemTy, vec.getNumElements());
  }
  if (auto fenum = type_dyn_cast<FEnumType>(firType)) {
    mlir::SmallVector<hw::UnionType::FieldInfo, 8> hwfields;
    SmallVector<Attribute> names;
    bool simple = true;
    for (auto element : fenum) {
      Type etype = lowerType(element.type, loc, getTypeDeclFn);
      if (!etype)
        return {};
      hwfields.push_back(hw::UnionType::FieldInfo{element.name, etype, 0});
      names.push_back(element.name);
      if (!isa<UIntType>(element.type) ||
          element.type.getBitWidthOrSentinel() != 0)
        simple = false;
    }
    auto tagTy = hw::EnumType::get(type.getContext(),
                                   ArrayAttr::get(type.getContext(), names));
    if (simple)
      return tagTy;
    auto bodyTy = hw::UnionType::get(type.getContext(), hwfields);
    hw::StructType::FieldInfo fields[2] = {
        {StringAttr::get(type.getContext(), "tag"), tagTy},
        {StringAttr::get(type.getContext(), "body"), bodyTy}};
    return hw::StructType::get(type.getContext(), fields);
  }

  auto width = firType.getBitWidthOrSentinel();
  if (width >= 0) // IntType, analog with known width, clock, etc.
    return IntegerType::get(type.getContext(), width);

  return {};
}
