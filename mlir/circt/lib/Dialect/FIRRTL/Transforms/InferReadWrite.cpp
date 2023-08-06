//===- InferReadWrite.cpp - Infer Read Write Memory -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferReadWrite pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-infer-read-write"

using namespace circt;
using namespace firrtl;

namespace {
struct InferReadWritePass : public InferReadWriteBase<InferReadWritePass> {

  /// This pass performs two memory transformations:
  ///  1. If the multi-bit enable port is connected to a constant 1,
  ///     then, replace with a single bit mask. Create a new memory with a
  ///     1 bit mask, and replace the old memory with it. The single bit mask
  ///     memory is always lowered to an unmasked memory.
  ///  2. If the read and write enable ports are trivially mutually exclusive,
  ///     then create a new memory with a single read/write port, and replace
  ///     the old memory with it.
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "\n Running Infer Read Write on module:"
                            << getOperation().getName());
    SmallVector<Operation *> opsToErase;
    for (MemOp memOp : llvm::make_early_inc_range(
             getOperation().getBodyBlock()->getOps<MemOp>())) {
      inferUnmasked(memOp, opsToErase);
      size_t nReads, nWrites, nRWs, nDbgs;
      memOp.getNumPorts(nReads, nWrites, nRWs, nDbgs);
      // Run the analysis only for Seq memories (latency=1) and a single read
      // and write ports.
      if (!(nReads == 1 && nWrites == 1 && nRWs == 0) ||
          !(memOp.getReadLatency() == 1 && memOp.getWriteLatency() == 1))
        continue;
      SmallVector<Attribute, 4> resultNames;
      SmallVector<Type, 4> resultTypes;
      SmallVector<Attribute> portAtts;
      SmallVector<Attribute, 4> portAnnotations;
      Value rClock, wClock;
      // The memory has exactly two ports.
      SmallVector<Value> readTerms, writeTerms;
      for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
        Attribute portAnno;
        portAnno = memOp.getPortAnnotation(portIt.index());
        if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug) {
          resultNames.push_back(memOp.getPortName(portIt.index()));
          resultTypes.push_back(memOp.getResult(portIt.index()).getType());
          portAnnotations.push_back(portAnno);
          continue;
        }
        // Append the annotations from the two ports.
        if (!cast<ArrayAttr>(portAnno).empty())
          portAtts.push_back(memOp.getPortAnnotation(portIt.index()));
        // Get the port value.
        Value portVal = portIt.value();
        // Get the port kind.
        bool isReadPort =
            memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read;
        // Iterate over all users of the port.
        for (Operation *u : portVal.getUsers())
          if (auto sf = dyn_cast<SubfieldOp>(u)) {
            // Get the field name.
            auto fName = sf.getInput().getType().get().getElementName(
                sf.getFieldIndex());
            // If this is the enable field, record the product terms(the And
            // expression tree).
            if (fName.equals("en"))
              getProductTerms(sf, isReadPort ? readTerms : writeTerms);

            else if (fName.equals("clk")) {
              if (isReadPort)
                rClock = getConnectSrc(sf);
              else
                wClock = getConnectSrc(sf);
            }
          }
        // End of loop for getting MemOp port users.
      }
      if (!sameDriver(rClock, wClock))
        continue;

      rClock = wClock;
      LLVM_DEBUG(
          llvm::dbgs() << "\n read clock:" << rClock
                       << " --- write clock:" << wClock;
          llvm::dbgs() << "\n Read terms==>"; for (auto t
                                                   : readTerms) llvm::dbgs()
                                              << "\n term::" << t;

          llvm::dbgs() << "\n Write terms==>"; for (auto t
                                                    : writeTerms) llvm::dbgs()
                                               << "\n term::" << t;

      );
      // If the read and write clocks are the same, and if any of the write
      // enable product terms are a complement of the read enable, then return
      // the write enable term.
      auto complementTerm = checkComplement(readTerms, writeTerms);
      if (!complementTerm)
        continue;

      // Create the merged rw port for the new memory.
      resultNames.push_back(StringAttr::get(memOp.getContext(), "rw"));
      // Set the type of the rw port.
      resultTypes.push_back(MemOp::getTypeForPort(
          memOp.getDepth(), memOp.getDataType(), MemOp::PortKind::ReadWrite,
          memOp.getMaskBits()));
      ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
      portAnnotations.push_back(builder.getArrayAttr(portAtts));
      // Create the new rw memory.
      auto rwMem = builder.create<MemOp>(
          resultTypes, memOp.getReadLatency(), memOp.getWriteLatency(),
          memOp.getDepth(), RUWAttr::Undefined,
          builder.getArrayAttr(resultNames), memOp.getNameAttr(),
          memOp.getNameKind(), memOp.getAnnotations(),
          builder.getArrayAttr(portAnnotations), memOp.getInnerSymAttr(),
          memOp.getInitAttr(), memOp.getPrefixAttr());
      ++numRWPortMemoriesInferred;
      auto rwPort = rwMem->getResult(nDbgs);
      // Create the subfield access to all fields of the port.
      // The addr should be connected to read/write address depending on the
      // read/write mode.
      auto addr = builder.create<SubfieldOp>(rwPort, "addr");
      // Enable is high whenever the memory is written or read.
      auto enb = builder.create<SubfieldOp>(rwPort, "en");
      // Read/Write clock.
      auto clk = builder.create<SubfieldOp>(rwPort, "clk");
      auto readData = builder.create<SubfieldOp>(rwPort, "rdata");
      // wmode is high when the port is in write mode. That is this can be
      // connected to the write enable.
      auto wmode = builder.create<SubfieldOp>(rwPort, "wmode");
      auto writeData = builder.create<SubfieldOp>(rwPort, "wdata");
      auto mask = builder.create<SubfieldOp>(rwPort, "wmask");
      // Temp wires to replace the original memory connects.
      auto rAddr =
          builder.create<WireOp>(addr.getType(), "readAddr").getResult();
      auto wAddr =
          builder.create<WireOp>(addr.getType(), "writeAddr").getResult();
      auto wEnWire =
          builder.create<WireOp>(enb.getType(), "writeEnable").getResult();
      auto rEnWire =
          builder.create<WireOp>(enb.getType(), "readEnable").getResult();
      auto writeClock =
          builder.create<WireOp>(ClockType::get(enb.getContext())).getResult();
      // addr = Mux(WriteEnable, WriteAddress, ReadAddress).
      builder.create<StrictConnectOp>(
          addr, builder.create<MuxPrimOp>(wEnWire, wAddr, rAddr));
      // Enable = Or(WriteEnable, ReadEnable).
      builder.create<StrictConnectOp>(
          enb, builder.create<OrPrimOp>(rEnWire, wEnWire));
      builder.setInsertionPointToEnd(wmode->getBlock());
      builder.create<StrictConnectOp>(wmode, complementTerm);
      // Now iterate over the original memory read and write ports.
      size_t dbgsIndex = 0;
      for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
        // Get the port value.
        Value portVal = portIt.value();
        if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug) {
          memOp.getResult(portIt.index())
              .replaceAllUsesWith(rwMem.getResult(dbgsIndex));
          dbgsIndex++;
          continue;
        }
        // Get the port kind.
        bool isReadPort =
            memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read;
        // Iterate over all users of the port, which are the subfield ops, and
        // replace them.
        for (Operation *u : portVal.getUsers())
          if (auto sf = dyn_cast<SubfieldOp>(u)) {
            StringRef fName = sf.getInput().getType().get().getElementName(
                sf.getFieldIndex());
            Value repl;
            if (isReadPort)
              repl = llvm::StringSwitch<Value>(fName)
                         .Case("en", rEnWire)
                         .Case("clk", clk)
                         .Case("addr", rAddr)
                         .Case("data", readData);
            else
              repl = llvm::StringSwitch<Value>(fName)
                         .Case("en", wEnWire)
                         .Case("clk", writeClock)
                         .Case("addr", wAddr)
                         .Case("data", writeData)
                         .Case("mask", mask);
            sf.replaceAllUsesWith(repl);
            // Once all the uses of the subfield op replaced, delete it.
            opsToErase.push_back(sf);
          }
      }
      // All uses for all results of mem removed, now erase the memOp.
      opsToErase.push_back(memOp);
    }
    for (auto *o : opsToErase)
      o->erase();
  }

private:
  // Get the source value which is connected to the dst.
  Value getConnectSrc(Value dst) {
    for (auto *c : dst.getUsers())
      if (auto connect = dyn_cast<FConnectLike>(c))
        if (connect.getDest() == dst)
          return connect.getSrc();

    return nullptr;
  }

  /// If the ports are not directly connected to the same clock, then check
  /// if indirectly connected to the same clock.
  bool sameDriver(Value rClock, Value wClock) {
    if (rClock == wClock)
      return true;
    DenseSet<Value> rClocks, wClocks;
    while (rClock) {
      // Record all the values which are indirectly connected to the clock
      // port.
      rClocks.insert(rClock);
      rClock = getConnectSrc(rClock);
    }

    bool sameClock = false;
    // Now check all the indirect connections to the write memory clock
    // port.
    while (wClock) {
      if (rClocks.find(wClock) != rClocks.end()) {
        sameClock = true;
        break;
      }
      wClock = getConnectSrc(wClock);
    }
    return sameClock;
  }

  void getProductTerms(Value enValue, SmallVector<Value> &terms) {
    if (!enValue)
      return;
    SmallVector<Value> worklist;
    worklist.push_back(enValue);
    while (!worklist.empty()) {
      auto term = worklist.back();
      worklist.pop_back();
      terms.push_back(term);
      if (isa<BlockArgument>(term))
        continue;
      TypeSwitch<Operation *>(term.getDefiningOp())
          .Case<NodeOp>([&](auto n) { worklist.push_back(n.getInput()); })
          .Case<AndPrimOp>([&](AndPrimOp andOp) {
            worklist.push_back(andOp.getOperand(0));
            worklist.push_back(andOp.getOperand(1));
          })
          .Case<MuxPrimOp>([&](auto muxOp) {
            // Check for the pattern when low is 0, which is equivalent to (sel
            // & high)
            // term = mux (sel, high, 0) => term = sel & high
            if (ConstantOp cLow = dyn_cast_or_null<ConstantOp>(
                    muxOp.getLow().getDefiningOp()))
              if (cLow.getValue().isZero()) {
                worklist.push_back(muxOp.getSel());
                worklist.push_back(muxOp.getHigh());
              }
          })
          .Default([&](auto) {
            if (auto src = getConnectSrc(term))
              worklist.push_back(src);
          });
    }
  }

  /// If any of the terms in the read enable, prodTerms[0] is a complement of
  /// any of the terms in the write enable prodTerms[1], return the
  /// corresponding write enable term. prodTerms[0], prodTerms[1] is a vector of
  /// Value, each of which correspond to the two product terms of read and write
  /// enable respectively.
  Value checkComplement(const SmallVector<Value> &readTerms,
                        const SmallVector<Value> &writeTerms) {
    // Foreach Value in first term, check if it is the complement of any of the
    // Value in second term.
    for (auto t1 : readTerms)
      for (auto t2 : writeTerms) {
        // Return t2, t1 is a Not of t2.
        if (!isa<BlockArgument>(t1) && isa<NotPrimOp>(t1.getDefiningOp()))
          if (cast<NotPrimOp>(t1.getDefiningOp()).getInput() == t2)
            return t2;
        // Else Return t2, if t2 is a Not of t1.
        if (!isa<BlockArgument>(t2) && isa<NotPrimOp>(t2.getDefiningOp()))
          if (cast<NotPrimOp>(t2.getDefiningOp()).getInput() == t1)
            return t2;
      }

    return {};
  }

  void inferUnmasked(MemOp &memOp, SmallVector<Operation *> &opsToErase) {
    bool isMasked = true;

    // Iterate over all results, and check if the mask field of the result is
    // connected to a multi-bit constant 1.
    for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
      // Read ports donot have the mask field.
      if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read ||
          memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug)
        continue;
      Value portVal = portIt.value();
      // Iterate over all users of the write/rw port.
      for (Operation *u : portVal.getUsers())
        if (auto sf = dyn_cast<SubfieldOp>(u)) {
          // Get the field name.
          auto fName =
              sf.getInput().getType().get().getElementName(sf.getFieldIndex());
          // Check if this is the mask field.
          if (fName.contains("mask")) {
            // Already 1 bit, nothing to do.
            if (sf.getResult().getType().getBitWidthOrSentinel() == 1)
              continue;
            // Check what is the mask field directly connected to.
            // If, a constant 1, then we can replace with unMasked memory.
            if (auto maskVal = getConnectSrc(sf))
              if (auto constVal = dyn_cast<ConstantOp>(maskVal.getDefiningOp()))
                if (constVal.getValue().isAllOnes())
                  isMasked = false;
          }
        }
    }

    if (!isMasked) {
      // Replace with a new memory of 1 bit mask.
      ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
      // Copy the result type, except the mask bits.
      SmallVector<Type, 4> resultTypes;
      for (size_t i = 0, e = memOp.getNumResults(); i != e; ++i)
        resultTypes.push_back(
            MemOp::getTypeForPort(memOp.getDepth(), memOp.getDataType(),
                                  memOp.getPortKind(i), /*maskBits=*/1));

      // Copy everything from old memory, except the result type.
      auto newMem = builder.create<MemOp>(
          resultTypes, memOp.getReadLatency(), memOp.getWriteLatency(),
          memOp.getDepth(), memOp.getRuw(), memOp.getPortNames().getValue(),
          memOp.getNameAttr(), memOp.getNameKind(),
          memOp.getAnnotations().getValue(),
          memOp.getPortAnnotations().getValue(), memOp.getInnerSymAttr());
      // Now replace the result of old memory with the new one.
      for (const auto &portIt : llvm::enumerate(memOp.getResults())) {
        // Old result.
        Value oldPort = portIt.value();
        // New result.
        auto newPortVal = newMem->getResult(portIt.index());
        // If read port, then blindly replace.
        if (memOp.getPortKind(portIt.index()) == MemOp::PortKind::Read ||
            memOp.getPortKind(portIt.index()) == MemOp::PortKind::Debug) {
          oldPort.replaceAllUsesWith(newPortVal);
          continue;
        }
        // Otherwise, all fields can be blindly replaced, except mask field.
        for (Operation *u : oldPort.getUsers()) {
          auto oldRes = dyn_cast<SubfieldOp>(u);
          auto sf =
              builder.create<SubfieldOp>(newPortVal, oldRes.getFieldIndex());
          auto fName =
              sf.getInput().getType().get().getElementName(sf.getFieldIndex());
          // Replace all mask fields with a one bit constant 1.
          // Replace all other fields with the new port.
          if (fName.contains("mask")) {
            WireOp dummy = builder.create<WireOp>(oldRes.getType());
            oldRes->replaceAllUsesWith(dummy);
            builder.create<StrictConnectOp>(
                sf, builder.create<ConstantOp>(
                        UIntType::get(builder.getContext(), 1), APInt(1, 1)));
          } else
            oldRes->replaceAllUsesWith(sf);

          opsToErase.push_back(oldRes);
        }
      }
      opsToErase.push_back(memOp);
      memOp = newMem;
    }
  }
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createInferReadWritePass() {
  return std::make_unique<InferReadWritePass>();
}
