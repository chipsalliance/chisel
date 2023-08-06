//===- HWMemSimImpl.cpp - HW Memory Implementation Pass -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts generated FIRRTL memory modules to
// simulation models.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWMemSimImplPass Pass
//===----------------------------------------------------------------------===//

namespace {
struct FirMemory {
  size_t numReadPorts;
  size_t numWritePorts;
  size_t numReadWritePorts;
  size_t dataWidth;
  size_t depth;
  size_t maskGran;
  size_t readLatency;
  size_t writeLatency;
  seq::RUW readUnderWrite;
  seq::WUW writeUnderWrite;
  SmallVector<int32_t> writeClockIDs;
  StringRef initFilename;
  bool initIsBinary;
  bool initIsInline;
};
} // end anonymous namespace

namespace {

class HWMemSimImpl {
  bool ignoreReadEnable;
  bool addMuxPragmas;
  bool disableMemRandomization;
  bool disableRegRandomization;
  bool addVivadoRAMAddressConflictSynthesisBugWorkaround;

  SmallVector<sv::RegOp> registers;

  Value addPipelineStages(ImplicitLocOpBuilder &b,
                          ModuleNamespace &moduleNamespace, size_t stages,
                          Value clock, Value data, const Twine &name,
                          Value gate = {});
  sv::AlwaysOp lastPipelineAlwaysOp;

public:
  Namespace &mlirModuleNamespace;

  HWMemSimImpl(bool ignoreReadEnable, bool addMuxPragmas,
               bool disableMemRandomization, bool disableRegRandomization,
               bool addVivadoRAMAddressConflictSynthesisBugWorkaround,
               Namespace &mlirModuleNamespace)
      : ignoreReadEnable(ignoreReadEnable), addMuxPragmas(addMuxPragmas),
        disableMemRandomization(disableMemRandomization),
        disableRegRandomization(disableRegRandomization),
        addVivadoRAMAddressConflictSynthesisBugWorkaround(
            addVivadoRAMAddressConflictSynthesisBugWorkaround),
        mlirModuleNamespace(mlirModuleNamespace) {}

  void generateMemory(HWModuleOp op, FirMemory mem);
};

struct HWMemSimImplPass : public sv::HWMemSimImplBase<HWMemSimImplPass> {
  void runOnOperation() override;

  using sv::HWMemSimImplBase<HWMemSimImplPass>::ignoreReadEnable;
  using sv::HWMemSimImplBase<HWMemSimImplPass>::replSeqMem;
  using sv::HWMemSimImplBase<HWMemSimImplPass>::addMuxPragmas;
  using sv::HWMemSimImplBase<HWMemSimImplPass>::disableMemRandomization;
  using sv::HWMemSimImplBase<HWMemSimImplPass>::disableRegRandomization;
  using sv::HWMemSimImplBase<
      HWMemSimImplPass>::addVivadoRAMAddressConflictSynthesisBugWorkaround;
};

} // end anonymous namespace

static FirMemory analyzeMemOp(HWModuleGeneratedOp op) {
  FirMemory mem;
  mem.depth = op->getAttrOfType<IntegerAttr>("depth").getInt();
  mem.numReadPorts = op->getAttrOfType<IntegerAttr>("numReadPorts").getUInt();
  mem.numWritePorts = op->getAttrOfType<IntegerAttr>("numWritePorts").getUInt();
  mem.numReadWritePorts =
      op->getAttrOfType<IntegerAttr>("numReadWritePorts").getUInt();
  mem.readLatency = op->getAttrOfType<IntegerAttr>("readLatency").getUInt();
  mem.writeLatency = op->getAttrOfType<IntegerAttr>("writeLatency").getUInt();
  mem.dataWidth = op->getAttrOfType<IntegerAttr>("width").getUInt();
  if (op->hasAttrOfType<IntegerAttr>("maskGran"))
    mem.maskGran = op->getAttrOfType<IntegerAttr>("maskGran").getUInt();
  else
    mem.maskGran = mem.dataWidth;
  mem.readUnderWrite =
      op->getAttrOfType<seq::RUWAttr>("readUnderWrite").getValue();
  mem.writeUnderWrite =
      op->getAttrOfType<seq::WUWAttr>("writeUnderWrite").getValue();
  if (auto clockIDsAttr = op->getAttrOfType<ArrayAttr>("writeClockIDs"))
    for (auto clockID : clockIDsAttr)
      mem.writeClockIDs.push_back(
          clockID.cast<IntegerAttr>().getValue().getZExtValue());
  mem.initFilename = op->getAttrOfType<StringAttr>("initFilename").getValue();
  mem.initIsBinary = op->getAttrOfType<BoolAttr>("initIsBinary").getValue();
  mem.initIsInline = op->getAttrOfType<BoolAttr>("initIsInline").getValue();
  return mem;
}

/// A helper that returns true if a value definition (or block argument) is
/// visible to another operation, either because it's a block argument or
/// because the defining op is before that other op.
static bool valueDefinedBeforeOp(Value value, Operation *op) {
  Operation *valueOp = value.getDefiningOp();
  Block *valueBlock =
      valueOp ? valueOp->getBlock() : value.cast<BlockArgument>().getOwner();
  while (op->getBlock() && op->getBlock() != valueBlock)
    op = op->getParentOp();
  return valueBlock == op->getBlock() &&
         (!valueOp || valueOp->isBeforeInBlock(op));
}

//
// Construct memory read annotated with mux pragmas in the following
// form:
// ```
//   wire GEN;
//   /* synopsys infer_mux_override */
//   assign GEN = memory[addr] /* cadence map_to_mux */;
// ```
// If `addMuxPragmas` is enabled, just return the read value without
// annotations.
static Value getMemoryRead(ImplicitLocOpBuilder &b, Value memory, Value addr,
                           bool addMuxPragmas) {
  auto slot =
      b.create<sv::ReadInOutOp>(b.create<sv::ArrayIndexInOutOp>(memory, addr));
  // If we don't want to add mux pragmas, just return the read value.
  if (!addMuxPragmas || memory.getType()
                                .cast<hw::InOutType>()
                                .getElementType()
                                .cast<hw::UnpackedArrayType>()
                                .getSize() <= 1)
    return slot;
  circt::sv::setSVAttributes(
      slot, sv::SVAttributeAttr::get(b.getContext(), "cadence map_to_mux",
                                     /*emitAsComment=*/true));
  auto valWire = b.create<sv::WireOp>(slot.getType());
  auto assignOp = b.create<sv::AssignOp>(valWire, slot);
  sv::setSVAttributes(assignOp,
                      sv::SVAttributeAttr::get(b.getContext(),
                                               "synopsys infer_mux_override",
                                               /*emitAsComment=*/true));

  return b.create<sv::ReadInOutOp>(valWire);
}

Value HWMemSimImpl::addPipelineStages(ImplicitLocOpBuilder &b,
                                      ModuleNamespace &moduleNamespace,
                                      size_t stages, Value clock, Value data,
                                      const Twine &name, Value gate) {
  if (!stages)
    return data;

  // Try to reuse the previous always block. This is only possible if the clocks
  // agree and the data and gate all dominate the always block.
  auto alwaysOp = lastPipelineAlwaysOp;
  if (alwaysOp) {
    if (alwaysOp.getClocks() != ValueRange{clock} ||
        !valueDefinedBeforeOp(data, alwaysOp) ||
        (gate && !valueDefinedBeforeOp(gate, alwaysOp)))
      alwaysOp = {};
  }
  if (!alwaysOp)
    alwaysOp = b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock);

  // Add the necessary registers.
  auto savedIP = b.saveInsertionPoint();
  SmallVector<sv::RegOp> regs;
  b.setInsertionPoint(alwaysOp);
  for (unsigned i = 0; i < stages; ++i) {
    auto regName =
        b.getStringAttr(moduleNamespace.newName("_" + name + "_d" + Twine(i)));
    auto reg = b.create<sv::RegOp>(data.getType(), regName,
                                   hw::InnerSymAttr::get(regName));
    regs.push_back(reg);
    registers.push_back(reg);
  }

  // Populate the assignments in the always block.
  b.setInsertionPointToEnd(alwaysOp.getBodyBlock());
  for (unsigned i = 0; i < stages; ++i) {
    if (i > 0)
      data = b.create<sv::ReadInOutOp>(data);
    auto emitAssign = [&] { b.create<sv::PAssignOp>(regs[i], data); };
    if (gate)
      b.create<sv::IfOp>(gate, [&]() { emitAssign(); });
    else
      emitAssign();
    data = regs[i];
    gate = {};
  }
  b.restoreInsertionPoint(savedIP);
  data = b.create<sv::ReadInOutOp>(data);

  lastPipelineAlwaysOp = alwaysOp;
  return data;
}

void HWMemSimImpl::generateMemory(HWModuleOp op, FirMemory mem) {
  ImplicitLocOpBuilder b(op.getLoc(), op.getBody());

  ModuleNamespace moduleNamespace(op);

  // Compute total number of mask bits.
  if (mem.maskGran == 0)
    mem.maskGran = mem.dataWidth;
  auto maskBits = mem.dataWidth / mem.maskGran;
  bool isMasked = maskBits > 1;
  // Each mask bit controls mask-granularity number of data bits.
  auto dataType = b.getIntegerType(mem.dataWidth);

  // Count the total number of ports.
  unsigned numPorts =
      mem.numReadPorts + mem.numWritePorts + mem.numReadWritePorts;

  // Create registers for the memory.
  sv::RegOp reg = b.create<sv::RegOp>(
      UnpackedArrayType::get(dataType, mem.depth), b.getStringAttr("Memory"));

  if (addVivadoRAMAddressConflictSynthesisBugWorkaround) {
    if (mem.readLatency == 0) {
      // If the read latency is zero, we regard the memory as write-first.
      // We add a SV attribute to specify a ram style to use LUTs for Vivado
      // to avoid a bug that miscompiles the write-first memory. See "RAM
      // address conflict and Vivado synthesis bug" issue in the vivado forum
      // for the more detail.
      circt::sv::setSVAttributes(
          reg, sv::SVAttributeAttr::get(b.getContext(), "ram_style",
                                        R"("distributed")",
                                        /*emitAsComment=*/false));
    } else if (mem.readLatency == 1 && numPorts > 1) {
      // If the read address is registered and the RAM has multiple ports,
      // force write-first behaviour by setting rw_addr_collision. This avoids
      // unpredictable behaviour. Downstreams flows should watch for `VPL
      // 8-6430`.
      circt::sv::setSVAttributes(
          reg, sv::SVAttributeAttr::get(b.getContext(), "rw_addr_collision",
                                        R"("yes")", /*emitAsComment=*/false));
    }
  }

  SmallVector<Value, 4> outputs;

  size_t inArg = 0;
  for (size_t i = 0; i < mem.numReadPorts; ++i) {
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    // Add pipeline stages
    if (ignoreReadEnable) {
      for (size_t j = 0, e = mem.readLatency; j != e; ++j) {
        auto enLast = en;
        if (j < e - 1)
          en = addPipelineStages(b, moduleNamespace, 1, clock, en,
                                 "R" + Twine(i) + "_en");
        addr = addPipelineStages(b, moduleNamespace, 1, clock, addr,
                                 "R" + Twine(i) + "_addr", enLast);
      }
    } else {
      en = addPipelineStages(b, moduleNamespace, mem.readLatency, clock, en,
                             "R" + Twine(i) + "_en");
      addr = addPipelineStages(b, moduleNamespace, mem.readLatency, clock, addr,
                               "R" + Twine(i) + "_addr");
    }

    // Read Logic
    Value rdata = getMemoryRead(b, reg, addr, addMuxPragmas);
    if (!ignoreReadEnable) {
      Value x = b.create<sv::ConstantXOp>(rdata.getType());
      rdata = b.create<comb::MuxOp>(en, rdata, x, false);
    }
    outputs.push_back(rdata);
  }

  for (size_t i = 0; i < mem.numReadWritePorts; ++i) {
    auto numReadStages = mem.readLatency;
    auto numWriteStages = mem.writeLatency - 1;
    auto numCommonStages = std::min(numReadStages, numWriteStages);
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    Value wmode = op.getBody().getArgument(inArg++);
    Value wdataIn = op.getBody().getArgument(inArg++);
    Value wmaskBits;
    // There are no input mask ports, if maskBits =1. Create a dummy true value
    // for mask.
    if (isMasked)
      wmaskBits = op.getBody().getArgument(inArg++);
    else
      wmaskBits = b.create<ConstantOp>(b.getIntegerAttr(en.getType(), 1));

    // Add common pipeline stages.
    addr = addPipelineStages(b, moduleNamespace, numCommonStages, clock, addr,
                             "RW" + Twine(i) + "_addr");
    en = addPipelineStages(b, moduleNamespace, numCommonStages, clock, en,
                           "RW" + Twine(i) + "_en");
    wmode = addPipelineStages(b, moduleNamespace, numCommonStages, clock, wmode,
                              "RW" + Twine(i) + "_mode");

    // Add read-only pipeline stages.
    Value readAddr = addr;
    Value readEn = en;
    if (ignoreReadEnable) {
      for (size_t j = 0, e = mem.readLatency; j != e; ++j) {
        auto enLast = en;
        if (j < e - 1)
          readEn = addPipelineStages(b, moduleNamespace, 1, clock, en,
                                     "RW" + Twine(i) + "_ren");
        readAddr = addPipelineStages(b, moduleNamespace, 1, clock, addr,
                                     "RW" + Twine(i) + "_raddr", enLast);
      }
    } else {
      readAddr =
          addPipelineStages(b, moduleNamespace, numReadStages - numCommonStages,
                            clock, addr, "RW" + Twine(i) + "_raddr");
      readEn =
          addPipelineStages(b, moduleNamespace, numReadStages - numCommonStages,
                            clock, en, "RW" + Twine(i) + "_ren");
    }
    auto readWMode =
        addPipelineStages(b, moduleNamespace, numReadStages - numCommonStages,
                          clock, wmode, "RW" + Twine(i) + "_rmode");

    // Add write-only pipeline stages.
    auto writeAddr =
        addPipelineStages(b, moduleNamespace, numWriteStages - numCommonStages,
                          clock, addr, "RW" + Twine(i) + "_waddr");
    auto writeEn =
        addPipelineStages(b, moduleNamespace, numWriteStages - numCommonStages,
                          clock, en, "RW" + Twine(i) + "_wen");
    auto writeWMode =
        addPipelineStages(b, moduleNamespace, numWriteStages - numCommonStages,
                          clock, wmode, "RW" + Twine(i) + "_wmode");
    wdataIn = addPipelineStages(b, moduleNamespace, numWriteStages, clock,
                                wdataIn, "RW" + Twine(i) + "_wdata");
    if (isMasked)
      wmaskBits = addPipelineStages(b, moduleNamespace, numWriteStages, clock,
                                    wmaskBits, "RW" + Twine(i) + "_wmask");

    SmallVector<Value, 4> maskValues(maskBits);
    SmallVector<Value, 4> dataValues(maskBits);
    // For multi-bit mask, extract corresponding write data bits of
    // mask-granularity size each. Each of the extracted data bits will be
    // written to a register, gaurded by the corresponding mask bit.
    for (size_t i = 0; i < maskBits; ++i) {
      maskValues[i] = b.createOrFold<comb::ExtractOp>(wmaskBits, i, 1);
      dataValues[i] = b.createOrFold<comb::ExtractOp>(wdataIn, i * mem.maskGran,
                                                      mem.maskGran);
    }

    // wire to store read result
    auto rWire = b.create<sv::WireOp>(wdataIn.getType());
    Value rdata = b.create<sv::ReadInOutOp>(rWire);

    // Read logic.
    Value rcond = b.createOrFold<comb::AndOp>(
        readEn,
        b.createOrFold<comb::ICmpOp>(
            comb::ICmpPredicate::eq, readWMode,
            b.createOrFold<ConstantOp>(readWMode.getType(), 0), false),
        false);

    auto val = getMemoryRead(b, reg, readAddr, addMuxPragmas);
    if (!ignoreReadEnable) {
      Value x = b.create<sv::ConstantXOp>(val.getType());
      val = b.create<comb::MuxOp>(rcond, val, x, false);
    }
    b.create<sv::AssignOp>(rWire, val);

    // Write logic gaurded by the corresponding mask bit.
    for (auto wmask : llvm::enumerate(maskValues)) {
      b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock, [&]() {
        auto wcond = b.createOrFold<comb::AndOp>(
            writeEn,
            b.createOrFold<comb::AndOp>(wmask.value(), writeWMode, false),
            false);
        b.create<sv::IfOp>(wcond, [&]() {
          Value slotReg = b.create<sv::ArrayIndexInOutOp>(reg, writeAddr);
          b.create<sv::PAssignOp>(
              b.createOrFold<sv::IndexedPartSelectInOutOp>(
                  slotReg,
                  b.createOrFold<ConstantOp>(b.getIntegerType(32),
                                             wmask.index() * mem.maskGran),
                  mem.maskGran),
              dataValues[wmask.index()]);
        });
      });
    }
    outputs.push_back(rdata);
  }

  DenseMap<unsigned, Operation *> writeProcesses;
  for (size_t i = 0; i < mem.numWritePorts; ++i) {
    auto numStages = mem.writeLatency - 1;
    Value addr = op.getBody().getArgument(inArg++);
    Value en = op.getBody().getArgument(inArg++);
    Value clock = op.getBody().getArgument(inArg++);
    Value wdataIn = op.getBody().getArgument(inArg++);
    Value wmaskBits;
    // There are no input mask ports, if maskBits =1. Create a dummy true value
    // for mask.
    if (isMasked)
      wmaskBits = op.getBody().getArgument(inArg++);
    else
      wmaskBits = b.create<ConstantOp>(b.getIntegerAttr(en.getType(), 1));
    // Add pipeline stages
    addr = addPipelineStages(b, moduleNamespace, numStages, clock, addr,
                             "W" + Twine(i) + "addr");
    en = addPipelineStages(b, moduleNamespace, numStages, clock, en,
                           "W" + Twine(i) + "en");
    wdataIn = addPipelineStages(b, moduleNamespace, numStages, clock, wdataIn,
                                "W" + Twine(i) + "data");
    if (isMasked)
      wmaskBits = addPipelineStages(b, moduleNamespace, numStages, clock,
                                    wmaskBits, "W" + Twine(i) + "mask");

    SmallVector<Value, 4> maskValues(maskBits);
    SmallVector<Value, 4> dataValues(maskBits);
    // For multi-bit mask, extract corresponding write data bits of
    // mask-granularity size each. Each of the extracted data bits will be
    // written to a register, gaurded by the corresponding mask bit.
    for (size_t i = 0; i < maskBits; ++i) {
      maskValues[i] = b.createOrFold<comb::ExtractOp>(wmaskBits, i, 1);
      dataValues[i] = b.createOrFold<comb::ExtractOp>(wdataIn, i * mem.maskGran,
                                                      mem.maskGran);
    }
    // Build write port logic.
    auto writeLogic = [&] {
      // For each register, create the connections to write the corresponding
      // data into it.
      for (auto wmask : llvm::enumerate(maskValues)) {
        // Guard by corresponding mask bit.
        auto wcond = b.createOrFold<comb::AndOp>(en, wmask.value(), false);
        b.create<sv::IfOp>(wcond, [&]() {
          auto slot = b.create<sv::ArrayIndexInOutOp>(reg, addr);
          b.create<sv::PAssignOp>(
              b.createOrFold<sv::IndexedPartSelectInOutOp>(
                  slot,
                  b.createOrFold<ConstantOp>(b.getIntegerType(32),
                                             wmask.index() * mem.maskGran),
                  mem.maskGran),
              dataValues[wmask.index()]);
        });
      }
    };

    // Build a new always block with write port logic.
    auto alwaysBlock = [&] {
      return b.create<sv::AlwaysOp>(sv::EventControl::AtPosEdge, clock,
                                    [&]() { writeLogic(); });
    };

    switch (mem.writeUnderWrite) {
    // Undefined write order:  lower each write port into a separate always
    // block.
    case seq::WUW::Undefined:
      alwaysBlock();
      break;
    // Port-ordered write order:  lower each write port into an always block
    // based on its clock ID.
    case seq::WUW::PortOrder:
      if (auto *existingAlwaysBlock =
              writeProcesses.lookup(mem.writeClockIDs[i])) {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPointToEnd(
            cast<sv::AlwaysOp>(existingAlwaysBlock).getBodyBlock());
        writeLogic();
      } else {
        writeProcesses[i] = alwaysBlock();
      }
    }
  }

  auto *outputOp = op.getBodyBlock()->getTerminator();
  outputOp->setOperands(outputs);

  // Add logic to initialize the memory based on a file emission request.  This
  // disables randomization.
  if (!mem.initFilename.empty()) {
    // Set an inner symbol on the register if one does not exist.
    if (!reg.getInnerSymAttr())
      reg.setInnerSymAttr(hw::InnerSymAttr::get(
          b.getStringAttr(moduleNamespace.newName(reg.getName()))));

    if (mem.initIsInline) {
      b.create<sv::IfDefOp>("ENABLE_INITIAL_MEM_", [&]() {
        b.create<sv::InitialOp>([&]() {
          b.create<sv::ReadMemOp>(reg, mem.initFilename,
                                  mem.initIsBinary
                                      ? MemBaseTypeAttr::MemBaseBin
                                      : MemBaseTypeAttr::MemBaseHex);
        });
      });
    } else {
      OpBuilder::InsertionGuard guard(b);

      // Create a new module with the readmem op.
      b.setInsertionPointAfter(op);
      auto boundModule = b.create<HWModuleOp>(
          b.getStringAttr(mlirModuleNamespace.newName(op.getName() + "_init")),
          ArrayRef<PortInfo>());

      auto filename = op->getAttrOfType<OutputFileAttr>("output_file");
      if (filename) {
        if (!filename.isDirectory()) {
          SmallString<128> dir(filename.getFilename().getValue());
          llvm::sys::path::remove_filename(dir);
          filename = hw::OutputFileAttr::getFromDirectoryAndFilename(
              b.getContext(), dir, boundModule.getName() + ".sv");
        }
      } else {
        filename = hw::OutputFileAttr::getFromFilename(
            b.getContext(), boundModule.getName() + ".sv");
      }

      // Build the hierpathop
      auto path = b.create<hw::HierPathOp>(
          mlirModuleNamespace.newName(op.getName() + "_path"),
          b.getArrayAttr(
              ::InnerRefAttr::get(op.getNameAttr(), reg.getInnerNameAttr())));

      boundModule->setAttr("output_file", filename);
      b.setInsertionPointToStart(op.getBodyBlock());
      b.setInsertionPointToStart(boundModule.getBodyBlock());
      b.create<sv::InitialOp>([&]() {
        auto xmr = b.create<sv::XMRRefOp>(reg.getType(), path.getSymNameAttr());
        b.create<sv::ReadMemOp>(xmr, mem.initFilename,
                                mem.initIsBinary ? MemBaseTypeAttr::MemBaseBin
                                                 : MemBaseTypeAttr::MemBaseHex);
      });

      // Instantiate this new module inside the memory module.
      b.setInsertionPointAfter(reg);
      auto boundInstance = b.create<hw::InstanceOp>(
          boundModule, boundModule.getName(), ArrayRef<Value>());
      boundInstance->setAttr(
          "inner_sym",
          hw::InnerSymAttr::get(b.getStringAttr(
              moduleNamespace.newName(boundInstance.getInstanceName()))));
      boundInstance->setAttr("doNotPrint", b.getBoolAttr(true));

      // Bind the new module.
      b.setInsertionPointAfter(boundModule);
      auto bind = b.create<sv::BindOp>(hw::InnerRefAttr::get(
          op.getNameAttr(), boundInstance.getInnerSymAttr().getSymName()));
      bind->setAttr("output_file", filename);
    }
  }

  // Add logic to initialize the memory and any internal registers to random
  // values.
  if (disableMemRandomization && disableRegRandomization)
    return;

  constexpr unsigned randomWidth = 32;
  b.create<sv::IfDefOp>("ENABLE_INITIAL_MEM_", [&]() {
    sv::RegOp randReg;
    SmallVector<sv::RegOp> randRegs;
    if (!disableRegRandomization) {
      b.create<sv::IfDefOp>("RANDOMIZE_REG_INIT", [&]() {
        signed totalWidth = 0;
        for (sv::RegOp &reg : registers)
          totalWidth += reg.getElementType().getIntOrFloatBitWidth();
        while (totalWidth > 0) {
          auto name = b.getStringAttr(moduleNamespace.newName("_RANDOM"));
          auto innerSym = hw::InnerSymAttr::get(name);
          randRegs.push_back(b.create<sv::RegOp>(b.getIntegerType(randomWidth),
                                                 name, innerSym));
          totalWidth -= randomWidth;
        }
      });
    }
    auto randomMemReg = b.create<sv::RegOp>(
        b.getIntegerType(llvm::divideCeil(mem.dataWidth, randomWidth) *
                         randomWidth),
        b.getStringAttr("_RANDOM_MEM"));
    b.create<sv::InitialOp>([&]() {
      b.create<sv::VerbatimOp>("`INIT_RANDOM_PROLOG_");

      // Memory randomization logic.  The entire memory is randomized.
      if (!disableMemRandomization) {
        b.create<sv::IfDefProceduralOp>("RANDOMIZE_MEM_INIT", [&]() {
          auto outerLoopIndVarType =
              b.getIntegerType(llvm::Log2_64_Ceil(mem.depth + 1));
          auto innerUpperBoundWidth = randomMemReg.getType()
                                          .getElementType()
                                          .cast<IntegerType>()
                                          .getWidth();
          auto innerLoopIndVarType =
              b.getIntegerType(llvm::Log2_64_Ceil(innerUpperBoundWidth + 1));
          // Construct the following nested for loops:
          // ```
          //   for (int i = 0; i < mem.depth; i++) begin
          //     for (int j = 0; j < randomMeg.size; j += 32)
          //       randomMem[j+31:j] = `RANDOM
          //     Memory[i] = randomMem[mem.dataWidth - 1: 0];
          // ```
          b.create<sv::ForOp>(
              0, mem.depth, 1, outerLoopIndVarType, "i",
              [&](BlockArgument outerIndVar) {
                b.create<sv::ForOp>(
                    0, innerUpperBoundWidth, randomWidth, innerLoopIndVarType,
                    "j", [&](BlockArgument innerIndVar) {
                      auto rhs = b.create<sv::MacroRefExprSEOp>(
                          b.getIntegerType(randomWidth), "RANDOM");
                      auto lhs = b.create<sv::IndexedPartSelectInOutOp>(
                          randomMemReg, innerIndVar, randomWidth, false);
                      b.create<sv::BPAssignOp>(lhs, rhs);
                    });

                Value iterValue = outerIndVar;
                // Truncate the induction variable if necessary.
                if (!outerIndVar.getType().isInteger(
                        llvm::Log2_64_Ceil(mem.depth)))
                  iterValue = b.createOrFold<comb::ExtractOp>(
                      iterValue, 0, llvm::Log2_64_Ceil(mem.depth));
                auto lhs = b.create<sv::ArrayIndexInOutOp>(reg, iterValue);
                auto rhs = b.createOrFold<comb::ExtractOp>(
                    b.create<sv::ReadInOutOp>(randomMemReg), 0, mem.dataWidth);
                b.create<sv::BPAssignOp>(lhs, rhs);
              });
        });
      }

      // Register randomization logic.  Randomize every register to a random
      // making efficient use of available randomization registers.
      //
      // TODO: This shares a lot of common logic with LowerToHW.  Combine
      // these two in a common randomization utility.
      if (!disableRegRandomization) {
        b.create<sv::IfDefProceduralOp>("RANDOMIZE_REG_INIT", [&]() {
          unsigned bits = randomWidth;
          for (sv::RegOp &reg : randRegs)
            b.create<sv::VerbatimOp>(
                b.getStringAttr("{{0}} = {`RANDOM};"), ValueRange{},
                b.getArrayAttr(hw::InnerRefAttr::get(op.getNameAttr(),
                                                     reg.getInnerNameAttr())));
          auto randRegIdx = 0;
          for (sv::RegOp &reg : registers) {
            SmallVector<std::pair<Attribute, std::pair<size_t, size_t>>> values;
            auto width = reg.getElementType().getIntOrFloatBitWidth();
            auto widthRemaining = width;
            while (widthRemaining > 0) {
              if (bits == randomWidth) {
                randReg = randRegs[randRegIdx++];
                bits = 0;
              }
              auto innerRef = hw::InnerRefAttr::get(op.getNameAttr(),
                                                    randReg.getInnerNameAttr());
              if (widthRemaining <= randomWidth - bits) {
                values.push_back({innerRef, {bits + widthRemaining - 1, bits}});
                bits += widthRemaining;
                widthRemaining = 0;
                continue;
              }
              values.push_back({innerRef, {randomWidth - 1, bits}});
              widthRemaining -= (randomWidth - bits);
              bits = randomWidth;
            }
            SmallString<32> rhs("{{0}} = ");
            unsigned idx = 1;
            assert(reg.getInnerSymAttr());
            SmallVector<Attribute, 4> symbols({hw::InnerRefAttr::get(
                op.getNameAttr(), reg.getInnerNameAttr())});
            if (values.size() > 1)
              rhs.append("{");
            for (auto &v : values) {
              if (idx > 1)
                rhs.append(", ");
              auto [sym, range] = v;
              symbols.push_back(sym);
              rhs.append(("{{" + Twine(idx++) + "}}").str());
              // Do not emit a part select as the whole value is used.
              if (range.first == randomWidth - 1 && range.second == 0)
                continue;
              // Emit a single bit part select, e.g., "[3]"
              if (range.first == range.second) {
                rhs.append(("[" + Twine(range.first) + "]").str());
                continue;
              }
              // Emit a part select, e.g., "[4:2]"
              rhs.append(
                  ("[" + Twine(range.first) + ":" + Twine(range.second) + "]")
                      .str());
            }
            if (values.size() > 1)
              rhs.append("}");
            rhs.append(";");
            b.create<sv::VerbatimOp>(rhs, ValueRange{},
                                     b.getArrayAttr(symbols));
          }
        });
      }
    });
  });
}

void HWMemSimImplPass::runOnOperation() {
  auto topModule = getOperation();

  // Populate a namespace from the symbols visible to the top-level MLIR module.
  // Memories with initializations create modules and these need to be legal
  // symbols.
  SymbolCache symbolCache;
  symbolCache.addDefinitions(topModule);
  Namespace mlirModuleNamespace;
  mlirModuleNamespace.add(symbolCache);

  SmallVector<HWModuleGeneratedOp> toErase;
  bool anythingChanged = false;

  for (auto op :
       llvm::make_early_inc_range(topModule.getOps<HWModuleGeneratedOp>())) {
    auto oldModule = cast<HWModuleGeneratedOp>(op);
    auto gen = oldModule.getGeneratorKind();
    auto genOp = cast<HWGeneratorSchemaOp>(
        SymbolTable::lookupSymbolIn(getOperation(), gen));

    if (genOp.getDescriptor() == "FIRRTL_Memory") {
      auto mem = analyzeMemOp(oldModule);

      OpBuilder builder(oldModule);
      auto nameAttr = builder.getStringAttr(oldModule.getName());

      // The requirements for macro replacement:
      // 1. read latency and write latency of one.
      // 2. undefined read-under-write behavior.
      if (replSeqMem && ((mem.readLatency == 1 && mem.writeLatency == 1) &&
                         mem.dataWidth > 0)) {
        builder.create<HWModuleExternOp>(oldModule.getLoc(), nameAttr,
                                         oldModule.getPortList());
      } else {
        auto newModule = builder.create<HWModuleOp>(
            oldModule.getLoc(), nameAttr, oldModule.getPortList());
        if (auto outdir = oldModule->getAttr("output_file"))
          newModule->setAttr("output_file", outdir);
        newModule.setCommentAttr(
            builder.getStringAttr("VCS coverage exclude_file"));

        HWMemSimImpl(ignoreReadEnable, addMuxPragmas, disableMemRandomization,
                     disableRegRandomization,
                     addVivadoRAMAddressConflictSynthesisBugWorkaround,
                     mlirModuleNamespace)
            .generateMemory(newModule, mem);
      }

      oldModule.erase();
      anythingChanged = true;
    }
  }

  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWMemSimImplPass(
    bool replSeqMem, bool ignoreReadEnable, bool addMuxPragmas,
    bool disableMemRandomization, bool disableRegRandomization,
    bool addVivadoRAMAddressConflictSynthesisBugWorkaround) {
  auto pass = std::make_unique<HWMemSimImplPass>();
  pass->replSeqMem = replSeqMem;
  pass->ignoreReadEnable = ignoreReadEnable;
  pass->addMuxPragmas = addMuxPragmas;
  pass->disableMemRandomization = disableMemRandomization;
  pass->disableRegRandomization = disableRegRandomization;
  pass->addVivadoRAMAddressConflictSynthesisBugWorkaround =
      addVivadoRAMAddressConflictSynthesisBugWorkaround;
  return pass;
}
