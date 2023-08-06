//===- LowerFirMem.cpp - Seq FIRRTL memory lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq FirMem ops to instances of HW generated modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/Namespace.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

#define DEBUG_TYPE "lower-firmem"

using namespace circt;
using namespace seq;
using namespace hw;
using hw::HWModuleGeneratedOp;
using llvm::MapVector;
using llvm::SmallDenseSet;

//===----------------------------------------------------------------------===//
// FIR Memory Parametrization
//===----------------------------------------------------------------------===//

namespace {
/// The configuration of a FIR memory.
struct FirMemConfig {
  size_t numReadPorts = 0;
  size_t numWritePorts = 0;
  size_t numReadWritePorts = 0;
  size_t dataWidth = 0;
  size_t depth = 0;
  size_t readLatency = 0;
  size_t writeLatency = 0;
  size_t maskBits = 0;
  RUW readUnderWrite = RUW::Undefined;
  WUW writeUnderWrite = WUW::Undefined;
  SmallVector<int32_t, 1> writeClockIDs;
  StringRef initFilename;
  bool initIsBinary = false;
  bool initIsInline = false;
  Attribute outputFile;
  StringRef prefix;

  llvm::hash_code hashValue() const {
    return llvm::hash_combine(numReadPorts, numWritePorts, numReadWritePorts,
                              dataWidth, depth, readLatency, writeLatency,
                              maskBits, readUnderWrite, writeUnderWrite,
                              initFilename, initIsBinary, initIsInline,
                              outputFile, prefix) ^
           llvm::hash_combine_range(writeClockIDs.begin(), writeClockIDs.end());
  }

  auto getTuple() const {
    return std::make_tuple(numReadPorts, numWritePorts, numReadWritePorts,
                           dataWidth, depth, readLatency, writeLatency,
                           maskBits, readUnderWrite, writeUnderWrite,
                           writeClockIDs, initFilename, initIsBinary,
                           initIsInline, outputFile, prefix);
  }

  bool operator==(const FirMemConfig &other) const {
    return getTuple() == other.getTuple();
  }
};
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<FirMemConfig> {
  static inline FirMemConfig getEmptyKey() {
    FirMemConfig cfg;
    cfg.depth = DenseMapInfo<size_t>::getEmptyKey();
    return cfg;
  }
  static inline FirMemConfig getTombstoneKey() {
    FirMemConfig cfg;
    cfg.depth = DenseMapInfo<size_t>::getTombstoneKey();
    return cfg;
  }
  static unsigned getHashValue(const FirMemConfig &cfg) {
    return cfg.hashValue();
  }
  static bool isEqual(const FirMemConfig &lhs, const FirMemConfig &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_DEF_LOWERFIRMEM
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct LowerFirMemPass : public impl::LowerFirMemBase<LowerFirMemPass> {
  /// A vector of unique `FirMemConfig`s and all the `FirMemOp`s that use it.
  using UniqueConfig = std::pair<FirMemConfig, SmallVector<FirMemOp, 1>>;
  using UniqueConfigs = SmallVector<UniqueConfig>;

  void runOnOperation() override;

  UniqueConfigs collectMemories(ArrayRef<HWModuleOp> modules);
  FirMemConfig collectMemory(FirMemOp op);

  SmallVector<HWModuleGeneratedOp>
  createMemoryModules(MutableArrayRef<UniqueConfig> configs);
  HWModuleGeneratedOp createMemoryModule(UniqueConfig &config,
                                         OpBuilder &builder,
                                         FlatSymbolRefAttr schemaSymRef,
                                         Namespace &globalNamespace);

  void lowerMemoriesInModule(
      HWModuleOp module,
      ArrayRef<std::tuple<FirMemConfig *, HWModuleGeneratedOp, FirMemOp>> mems);
};
} // namespace

void LowerFirMemPass::runOnOperation() {
  // Gather all HW modules. We'll parallelize over them.
  SmallVector<HWModuleOp> modules;
  getOperation().walk([&](HWModuleOp op) {
    modules.push_back(op);
    return WalkResult::skip();
  });
  LLVM_DEBUG(llvm::dbgs() << "Lowering memories in " << modules.size()
                          << " modules\n");

  // Gather all `FirMemOp`s in the HW modules and group them by configuration.
  auto uniqueMems = collectMemories(modules);
  LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueMems.size()
                          << " unique memory congiurations\n");
  if (uniqueMems.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  // Create the `HWModuleGeneratedOp`s for each unique configuration. The result
  // is a vector of the same size as `uniqueMems`, with a `HWModuleGeneratedOp`
  // for every unique memory configuration.
  auto genOps = createMemoryModules(uniqueMems);

  // Group the list of memories that we need to update per HW module. This will
  // allow us to parallelize across HW modules.
  MapVector<
      HWModuleOp,
      SmallVector<std::tuple<FirMemConfig *, HWModuleGeneratedOp, FirMemOp>>>
      memsToLowerByModule;

  for (auto [config, genOp] : llvm::zip(uniqueMems, genOps))
    for (auto memOp : config.second)
      memsToLowerByModule[memOp->getParentOfType<HWModuleOp>()].push_back(
          {&config.first, genOp, memOp});

  // Replace all `FirMemOp`s with instances of the generated module.
  if (getContext().isMultithreadingEnabled()) {
    llvm::parallelForEach(memsToLowerByModule, [&](auto pair) {
      lowerMemoriesInModule(pair.first, pair.second);
    });
  } else {
    for (auto [module, mems] : memsToLowerByModule)
      lowerMemoriesInModule(module, mems);
  }
}

/// Collect the memories in a list of HW modules.
LowerFirMemPass::UniqueConfigs
LowerFirMemPass::collectMemories(ArrayRef<HWModuleOp> modules) {
  // For each module in the list populate a separate vector of `FirMemOp`s in
  // that module. This allows for the traversal of the HW modules to be
  // parallelized.
  using ModuleMemories = SmallVector<std::pair<FirMemConfig, FirMemOp>, 0>;
  SmallVector<ModuleMemories> memories(modules.size());

  auto collect = [&](HWModuleOp module, ModuleMemories &memories) {
    // TODO: Check if this module is in the DUT hierarchy.
    // bool isInDut = state.isInDUT(module);
    module.walk([&](seq::FirMemOp op) {
      memories.push_back({collectMemory(op), op});
    });
  };

  if (getContext().isMultithreadingEnabled()) {
    llvm::parallelFor(0, modules.size(),
                      [&](auto idx) { collect(modules[idx], memories[idx]); });
  } else {
    for (auto [module, moduleMemories] : llvm::zip(modules, memories))
      collect(module, moduleMemories);
  }

  // Group the gathered memories by unique `FirMemConfig` details.
  MapVector<FirMemConfig, SmallVector<FirMemOp, 1>> grouped;
  for (auto [module, moduleMemories] : llvm::zip(modules, memories))
    for (auto [summary, memOp] : moduleMemories)
      grouped[summary].push_back(memOp);

  return grouped.takeVector();
}

/// Trace a value through wires to its original definition.
static Value lookThroughWires(Value value) {
  while (value) {
    if (auto wireOp = value.getDefiningOp<WireOp>()) {
      value = wireOp.getInput();
      continue;
    }
    break;
  }
  return value;
}

/// Determine the exact parametrization of the memory that should be generated
/// for a given `FirMemOp`.
FirMemConfig LowerFirMemPass::collectMemory(FirMemOp op) {
  FirMemConfig cfg;
  cfg.dataWidth = op.getType().getWidth();
  cfg.depth = op.getType().getDepth();
  cfg.readLatency = op.getReadLatency();
  cfg.writeLatency = op.getWriteLatency();
  cfg.maskBits = op.getType().getMaskWidth().value_or(1);
  cfg.readUnderWrite = op.getRuw();
  cfg.writeUnderWrite = op.getWuw();
  if (auto init = op.getInitAttr()) {
    cfg.initFilename = init.getFilename();
    cfg.initIsBinary = init.getIsBinary();
    cfg.initIsInline = init.getIsInline();
  }
  cfg.outputFile = op.getOutputFileAttr();
  if (auto prefix = op.getPrefixAttr())
    cfg.prefix = prefix.getValue();
  // TODO: Handle modName (maybe not?)
  // TODO: Handle groupID (maybe not?)

  // Count the read, write, and read-write ports, and identify the clocks
  // driving the write ports.
  SmallDenseMap<Value, unsigned> clockValues;
  for (auto *user : op->getUsers()) {
    if (isa<FirMemReadOp>(user))
      ++cfg.numReadPorts;
    else if (isa<FirMemWriteOp>(user))
      ++cfg.numWritePorts;
    else if (isa<FirMemReadWriteOp>(user))
      ++cfg.numReadWritePorts;

    // Assign IDs to the values used as clock. This allows later passes to
    // easily detect which clocks are effectively driven by the same value.
    if (isa<FirMemWriteOp, FirMemReadWriteOp>(user)) {
      auto clock = lookThroughWires(user->getOperand(2));
      cfg.writeClockIDs.push_back(
          clockValues.insert({clock, clockValues.size()}).first->second);
    }
  }

  return cfg;
}

/// Create the `HWModuleGeneratedOp` for a list of memory parametrizations.
SmallVector<HWModuleGeneratedOp>
LowerFirMemPass::createMemoryModules(MutableArrayRef<UniqueConfig> configs) {
  ModuleOp circuit = getOperation();

  // Create or re-use the generator schema.
  hw::HWGeneratorSchemaOp schemaOp;
  for (auto op : circuit.getOps<hw::HWGeneratorSchemaOp>()) {
    if (op.getDescriptor() == "FIRRTL_Memory") {
      schemaOp = op;
      break;
    }
  }
  if (!schemaOp) {
    auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
    std::array<StringRef, 14> schemaFields = {
        "depth",          "numReadPorts",
        "numWritePorts",  "numReadWritePorts",
        "readLatency",    "writeLatency",
        "width",          "maskGran",
        "readUnderWrite", "writeUnderWrite",
        "writeClockIDs",  "initFilename",
        "initIsBinary",   "initIsInline"};
    schemaOp = builder.create<hw::HWGeneratorSchemaOp>(
        getOperation().getLoc(), "FIRRTLMem", "FIRRTL_Memory",
        builder.getStrArrayAttr(schemaFields));
  }
  auto schemaSymRef = FlatSymbolRefAttr::get(schemaOp);

  // Determine the insertion point for each of the memory modules. We basically
  // put them ahead of the first module that instantiates that memory. Do this
  // here in one go such that the `isBeforeInBlock` calls don't have to
  // re-enumerate the entire IR every time we insert one of the memory modules.
  SmallVector<Operation *> insertionPoints;
  insertionPoints.reserve(configs.size());
  for (auto &config : configs) {
    Operation *op = nullptr;
    for (auto memOp : config.second)
      if (auto parent = memOp->getParentOfType<HWModuleOp>())
        if (!op || parent->isBeforeInBlock(op))
          op = parent;
    insertionPoints.push_back(op);
  }

  // Create the individual memory modules.
  SymbolCache symbolCache;
  symbolCache.addDefinitions(getOperation());
  Namespace globalNamespace;
  globalNamespace.add(symbolCache);

  SmallVector<HWModuleGeneratedOp> genOps;
  genOps.reserve(configs.size());
  for (auto [config, insertBefore] : llvm::zip(configs, insertionPoints)) {
    OpBuilder builder(circuit.getContext());
    builder.setInsertionPoint(insertBefore);
    genOps.push_back(
        createMemoryModule(config, builder, schemaSymRef, globalNamespace));
  }

  return genOps;
}

/// Create the `HWModuleGeneratedOp` for a single memory parametrization.
HWModuleGeneratedOp
LowerFirMemPass::createMemoryModule(UniqueConfig &config, OpBuilder &builder,
                                    FlatSymbolRefAttr schemaSymRef,
                                    Namespace &globalNamespace) {
  const auto &mem = config.first;
  auto &memOps = config.second;

  // Pick a name for the memory. Honor the optional prefix and try to include
  // the common part of the names of the memory instances that use this
  // configuration. The resulting name is of the form:
  //
  //   <prefix>_<commonName>_<depth>x<width>
  //
  StringRef baseName = "";
  bool firstFound = false;
  for (auto memOp : memOps) {
    if (auto memName = memOp.getName()) {
      if (!firstFound) {
        baseName = *memName;
        firstFound = true;
        continue;
      }
      unsigned idx = 0;
      for (; idx < memName->size() && idx < baseName.size(); ++idx)
        if ((*memName)[idx] != baseName[idx])
          break;
      baseName = baseName.take_front(idx);
    }
  }
  baseName = baseName.rtrim('_');

  SmallString<32> nameBuffer;
  nameBuffer += mem.prefix;
  if (!baseName.empty()) {
    nameBuffer += baseName;
  } else {
    nameBuffer += "mem";
  }
  nameBuffer += "_";
  (Twine(mem.depth) + "x" + Twine(mem.dataWidth)).toVector(nameBuffer);
  auto name = builder.getStringAttr(globalNamespace.newName(nameBuffer));

  LLVM_DEBUG(llvm::dbgs() << "Creating " << name << " for " << mem.depth
                          << " x " << mem.dataWidth << " memory\n");

  bool withMask = mem.maskBits > 1;
  SmallVector<hw::PortInfo> ports;

  // Common types used for memory ports.
  Type bitType = IntegerType::get(&getContext(), 1);
  Type dataType =
      IntegerType::get(&getContext(), std::max((size_t)1, mem.dataWidth));
  Type maskType = IntegerType::get(&getContext(), mem.maskBits);
  Type addrType = IntegerType::get(&getContext(),
                                   std::max(1U, llvm::Log2_64_Ceil(mem.depth)));

  // Helper to add an input port.
  size_t inputIdx = 0;
  auto addInput = [&](StringRef prefix, size_t idx, StringRef suffix,
                      Type type) {
    ports.push_back({{builder.getStringAttr(prefix + Twine(idx) + suffix), type,
                      ModulePort::Direction::Input},
                     inputIdx++});
  };

  // Helper to add an output port.
  size_t outputIdx = 0;
  auto addOutput = [&](StringRef prefix, size_t idx, StringRef suffix,
                       Type type) {
    ports.push_back({{builder.getStringAttr(prefix + Twine(idx) + suffix), type,
                      ModulePort::Direction::Output},
                     outputIdx++});
  };

  // Helper to add the ports common to read, read-write, and write ports.
  auto addCommonPorts = [&](StringRef prefix, size_t idx) {
    addInput(prefix, idx, "_addr", addrType);
    addInput(prefix, idx, "_en", bitType);
    addInput(prefix, idx, "_clk", bitType);
  };

  // Add the read ports.
  for (size_t i = 0, e = mem.numReadPorts; i != e; ++i) {
    addCommonPorts("R", i);
    addOutput("R", i, "_data", dataType);
  }

  // Add the read-write ports.
  for (size_t i = 0, e = mem.numReadWritePorts; i != e; ++i) {
    addCommonPorts("RW", i);
    addInput("RW", i, "_wmode", bitType);
    addInput("RW", i, "_wdata", dataType);
    addOutput("RW", i, "_rdata", dataType);
    if (withMask)
      addInput("RW", i, "_wmask", maskType);
  }

  // Add the write ports.
  for (size_t i = 0, e = mem.numWritePorts; i != e; ++i) {
    addCommonPorts("W", i);
    addInput("W", i, "_data", dataType);
    if (withMask)
      addInput("W", i, "_mask", maskType);
  }

  // Mask granularity is the number of data bits that each mask bit can
  // guard. By default it is equal to the data bitwidth.
  auto genAttr = [&](StringRef name, Attribute attr) {
    return builder.getNamedAttr(name, attr);
  };
  auto genAttrUI32 = [&](StringRef name, uint32_t value) {
    return genAttr(name, builder.getUI32IntegerAttr(value));
  };
  NamedAttribute genAttrs[] = {
      genAttr("depth", builder.getI64IntegerAttr(mem.depth)),
      genAttrUI32("numReadPorts", mem.numReadPorts),
      genAttrUI32("numWritePorts", mem.numWritePorts),
      genAttrUI32("numReadWritePorts", mem.numReadWritePorts),
      genAttrUI32("readLatency", mem.readLatency),
      genAttrUI32("writeLatency", mem.writeLatency),
      genAttrUI32("width", mem.dataWidth),
      genAttrUI32("maskGran", mem.dataWidth / mem.maskBits),
      genAttr("readUnderWrite",
              seq::RUWAttr::get(builder.getContext(), mem.readUnderWrite)),
      genAttr("writeUnderWrite",
              seq::WUWAttr::get(builder.getContext(), mem.writeUnderWrite)),
      genAttr("writeClockIDs", builder.getI32ArrayAttr(mem.writeClockIDs)),
      genAttr("initFilename", builder.getStringAttr(mem.initFilename)),
      genAttr("initIsBinary", builder.getBoolAttr(mem.initIsBinary)),
      genAttr("initIsInline", builder.getBoolAttr(mem.initIsInline))};

  // Combine the locations of all actual `FirMemOp`s to be the location of the
  // generated memory.
  Location loc = memOps.front().getLoc();
  if (memOps.size() > 1) {
    SmallVector<Location> locs;
    for (auto memOp : memOps)
      locs.push_back(memOp.getLoc());
    loc = FusedLoc::get(&getContext(), locs);
  }

  // Create the module.
  auto genOp = builder.create<hw::HWModuleGeneratedOp>(
      loc, schemaSymRef, name, ports, StringRef{}, ArrayAttr{}, genAttrs);
  if (mem.outputFile)
    genOp->setAttr("output_file", mem.outputFile);

  return genOp;
}

/// Replace all `FirMemOp`s in an HW module with an instance of the
/// corresponding generated module.
void LowerFirMemPass::lowerMemoriesInModule(
    HWModuleOp module,
    ArrayRef<std::tuple<FirMemConfig *, HWModuleGeneratedOp, FirMemOp>> mems) {
  LLVM_DEBUG(llvm::dbgs() << "Lowering " << mems.size() << " memories in "
                          << module.getName() << "\n");

  hw::ConstantOp constOneOp;
  auto constOne = [&] {
    if (!constOneOp) {
      auto builder = OpBuilder::atBlockBegin(module.getBodyBlock());
      constOneOp = builder.create<hw::ConstantOp>(module.getLoc(),
                                                  builder.getI1Type(), 1);
    }
    return constOneOp;
  };
  auto valueOrOne = [&](Value value) { return value ? value : constOne(); };

  for (auto [config, genOp, memOp] : mems) {
    LLVM_DEBUG(llvm::dbgs() << "- Lowering " << memOp.getName() << "\n");
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;

    auto addInput = [&](Value value) { inputs.push_back(value); };
    auto addOutput = [&](Value value) { outputs.push_back(value); };

    // Add the read ports.
    for (auto *op : memOp->getUsers()) {
      auto port = dyn_cast<FirMemReadOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClock());
      addOutput(port.getData());
    }

    // Add the read-write ports.
    for (auto *op : memOp->getUsers()) {
      auto port = dyn_cast<FirMemReadWriteOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClock());
      addInput(port.getMode());
      addInput(port.getWriteData());
      addOutput(port.getReadData());
      if (config->maskBits > 1)
        addInput(valueOrOne(port.getMask()));
    }

    // Add the write ports.
    for (auto *op : memOp->getUsers()) {
      auto port = dyn_cast<FirMemWriteOp>(op);
      if (!port)
        continue;
      addInput(port.getAddress());
      addInput(valueOrOne(port.getEnable()));
      addInput(port.getClock());
      addInput(port.getData());
      if (config->maskBits > 1)
        addInput(valueOrOne(port.getMask()));
    }

    // Create the module instance.
    StringRef memName = "mem";
    if (auto name = memOp.getName(); name && !name->empty())
      memName = *name;
    ImplicitLocOpBuilder builder(memOp.getLoc(), memOp);
    auto instOp = builder.create<hw::InstanceOp>(
        genOp, builder.getStringAttr(memName + "_ext"), inputs, ArrayAttr{},
        memOp.getInnerSymAttr());
    for (auto [oldOutput, newOutput] : llvm::zip(outputs, instOp.getResults()))
      oldOutput.replaceAllUsesWith(newOutput);

    // Carry attributes over from the `FirMemOp` to the `InstanceOp`.
    auto defaultAttrNames = memOp.getAttributeNames();
    for (auto namedAttr : memOp->getAttrs())
      if (!llvm::is_contained(defaultAttrNames, namedAttr.getName()))
        instOp->setAttr(namedAttr.getName(), namedAttr.getValue());

    // Get rid of the `FirMemOp`.
    for (auto *user : llvm::make_early_inc_range(memOp->getUsers()))
      user->erase();
    memOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::seq::createLowerFirMemPass() {
  return std::make_unique<LowerFirMemPass>();
}
