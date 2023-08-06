//===- ExportVerilogInternals.h - Shared Internal Impl Details --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_EXPORTVERILOG_EXPORTVERILOGINTERNAL_H
#define CONVERSION_EXPORTVERILOG_EXPORTVERILOGINTERNAL_H

#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <atomic>

namespace circt {
struct LoweringOptions;

namespace ExportVerilog {
class GlobalNameResolver;

/// Check if the value is from read of a wire or reg or is a port.
bool isSimpleReadOrPort(Value v);

/// Given an expression that is spilled into a temporary wire, try to
/// synthesize a better name than "_T_42" based on the structure of the
/// expression.
StringAttr inferStructuralNameForTemporary(Value expr);

/// This class keeps track of global names at the module/interface level.
/// It is built in a global pass over the entire design and then frozen to allow
/// concurrent accesses.
struct GlobalNameTable {
  GlobalNameTable(GlobalNameTable &&) = default;

  /// Return the string to use for the specified parameter name in the specified
  /// module.  Parameters may be renamed for a variety of reasons (e.g.
  /// conflicting with ports or Verilog keywords), and this returns the
  /// legalized name to use.
  StringRef getParameterVerilogName(Operation *module,
                                    StringAttr paramName) const {
    auto it = renamedParams.find(std::make_pair(module, paramName));
    return (it != renamedParams.end() ? it->second : paramName).getValue();
  }

  StringAttr getEnumPrefix(Type type) const {
    auto it = enumPrefixes.find(type);
    return it != enumPrefixes.end() ? it->second : StringAttr();
  }

private:
  friend class GlobalNameResolver;
  GlobalNameTable() {}
  GlobalNameTable(const GlobalNameTable &) = delete;
  void operator=(const GlobalNameTable &) = delete;

  void addRenamedParam(Operation *module, StringAttr oldName,
                       StringRef newName) {
    renamedParams[{module, oldName}] =
        StringAttr::get(oldName.getContext(), newName);
  }

  /// This contains entries for any parameters that got renamed.  The key is a
  /// moduleop/paramName tuple, the value is the name to use.
  DenseMap<std::pair<Operation *, Attribute>, StringAttr> renamedParams;

  // This contains prefixes for any typedecl'd enum types. Keys are type-aliases
  // of enum types.
  DenseMap<Type, StringAttr> enumPrefixes;
};

//===----------------------------------------------------------------------===//
// NameCollisionResolver
//===----------------------------------------------------------------------===//

struct NameCollisionResolver {
  NameCollisionResolver() = default;

  /// Given a name that may have collisions or invalid symbols, return a
  /// replacement name to use, or the original name if it was ok.
  StringRef getLegalName(StringRef originalName);
  StringRef getLegalName(StringAttr originalName) {
    return getLegalName(originalName.getValue());
  }

  /// Insert a string as an already-used name.
  void insertUsedName(StringRef name) {
    nextGeneratedNameIDs.insert({name, 0});
  }

private:
  /// A map from used names to numeric suffix used as uniquification agent when
  /// resolving conflicts.
  llvm::StringMap<size_t> nextGeneratedNameIDs;

  NameCollisionResolver(const NameCollisionResolver &) = delete;
  void operator=(const NameCollisionResolver &) = delete;
};

//===----------------------------------------------------------------------===//
// FieldNameResolver
//===----------------------------------------------------------------------===//

struct FieldNameResolver {
  FieldNameResolver(const GlobalNameTable &globalNames)
      : globalNames(globalNames){};

  StringAttr getRenamedFieldName(StringAttr fieldName);

  /// Returns the field name for an enum field of a given enum field attr. In
  /// case a prefix can be inferred for the provided enum type (the enum type is
  /// a type alias), the prefix will be applied. If not, the raw field name
  /// is returned.
  std::string getEnumFieldName(hw::EnumFieldAttr attr);

private:
  void setRenamedFieldName(StringAttr fieldName, StringAttr newFieldName);

  /// Those contain entries for field names and types respectively. Struct types
  /// have names as field names, which must be renamed if they conflict with
  /// verilog keywords.
  DenseMap<StringAttr, StringAttr> renamedFieldNames;

  /// A map from used names to numeric suffix used as uniquification agent when
  /// resolving conflicts.
  llvm::StringMap<size_t> nextGeneratedNameIDs;

  // Handle to the global name table.
  const GlobalNameTable &globalNames;
};

//===----------------------------------------------------------------------===//
// SharedEmitterState
//===----------------------------------------------------------------------===//

/// Information to control the emission of a single operation into a file.
struct OpFileInfo {
  /// The operation to be emitted.
  Operation *op;

  /// Where among the replicated per-file operations the `op` above should be
  /// emitted.
  size_t position = 0;
};

/// Information to control the emission of a list of operations into a file.
struct FileInfo {
  /// The operations to be emitted into a separate file, and where among the
  /// replicated per-file operations the operation should be emitted.
  SmallVector<OpFileInfo, 1> ops;

  /// Whether to emit the replicated per-file operations.
  bool emitReplicatedOps = true;

  /// Whether to include this file as part of the emitted file list.
  bool addToFilelist = true;

  /// If true, the file is a header.
  bool isHeader = false;

  /// If true, the file is known to be (system) verilog source code.
  /// This flag is used to distinguish verilog from other files such as json.
  bool isVerilog = true;
};

/// This class wraps an operation or a fixed string that should be emitted.
class StringOrOpToEmit {
public:
  explicit StringOrOpToEmit(Operation *op) : pointerData(op), length(~0ULL) {}

  explicit StringOrOpToEmit(StringRef string) {
    pointerData = (Operation *)nullptr;
    setString(string);
  }

  ~StringOrOpToEmit() {
    if (const void *ptr = pointerData.dyn_cast<const void *>())
      free(const_cast<void *>(ptr));
  }

  /// If the value is an Operation*, return it.  Otherwise return null.
  Operation *getOperation() const {
    return pointerData.dyn_cast<Operation *>();
  }

  /// If the value wraps a string, return it.  Otherwise return null.
  StringRef getStringData() const {
    if (const void *ptr = pointerData.dyn_cast<const void *>())
      return StringRef((const char *)ptr, length);
    return StringRef();
  }

  /// This method transforms the entry from an operation to a string value.
  void setString(StringRef value) {
    assert(pointerData.is<Operation *>() && "shouldn't already be a string");
    length = value.size();
    void *data = malloc(length);
    memcpy(data, value.data(), length);
    pointerData = (const void *)data;
  }

  // These move just fine.
  StringOrOpToEmit(StringOrOpToEmit &&rhs)
      : pointerData(rhs.pointerData), length(rhs.length) {
    rhs.pointerData = (Operation *)nullptr;
  }

private:
  StringOrOpToEmit(const StringOrOpToEmit &) = delete;
  void operator=(const StringOrOpToEmit &) = delete;
  PointerUnion<Operation *, const void *> pointerData;
  size_t length;
};

/// This class tracks the top-level state for the emitters, which is built and
/// then shared across all per-file emissions that happen in parallel.
struct SharedEmitterState {
  /// The MLIR module to emit.
  ModuleOp designOp;

  /// The main file that collects all operations that are neither replicated
  /// per-file ops nor specifically assigned to a file.
  FileInfo rootFile;

  /// The additional files to emit, with the output file name as the key into
  /// the map.
  llvm::MapVector<StringAttr, FileInfo> files;

  /// The various file lists and their contents to emit
  llvm::StringMap<SmallVector<StringAttr>> fileLists;

  /// A list of operations replicated in each output file (e.g., `sv.verbatim`
  /// or `sv.ifdef` without dedicated output file).
  SmallVector<Operation *, 0> replicatedOps;

  /// Whether any error has been encountered during emission.
  std::atomic<bool> encounteredError = {};

  /// A cache of symbol -> defining ops built once and used by each of the
  /// verilog module emitters.  This is built at "gatherFiles" time.
  hw::HWSymbolCache symbolCache;

  // Emitter options extracted from the top-level module.
  const LoweringOptions &options;

  /// This is a set is populated at "gather" time, containing the hw.module
  /// operations that have a sv.bind in them.
  SmallPtrSet<Operation *, 8> modulesContainingBinds;

  /// Information about renamed global symbols, parameters, etc.
  const GlobalNameTable globalNames;

  explicit SharedEmitterState(ModuleOp designOp, const LoweringOptions &options,
                              GlobalNameTable globalNames)
      : designOp(designOp), options(options),
        globalNames(std::move(globalNames)) {}
  void gatherFiles(bool separateModules);

  using EmissionList = std::vector<StringOrOpToEmit>;

  void collectOpsForFile(const FileInfo &fileInfo, EmissionList &thingsToEmit,
                         bool emitHeader = false);
  void emitOps(EmissionList &thingsToEmit, raw_ostream &os, bool parallelize);
};

//===----------------------------------------------------------------------===//
// Other utilities
//===----------------------------------------------------------------------===//

/// Return true for operations that must always be inlined into a containing
/// expression for correctness.
static inline bool isExpressionAlwaysInline(Operation *op) {
  // We need to emit array indexes inline per verilog "lvalue" semantics.
  if (isa<sv::ArrayIndexInOutOp>(op) || isa<sv::StructFieldInOutOp>(op) ||
      isa<sv::IndexedPartSelectInOutOp>(op) || isa<sv::ReadInOutOp>(op))
    return true;

  // An SV interface modport is a symbolic name that is always inlined.
  if (isa<sv::GetModportOp>(op) || isa<sv::ReadInterfaceSignalOp>(op))
    return true;

  // XMRs can't be spilled if they are on the lhs.  Conservatively never spill
  // them.
  if (isa<sv::XMROp, sv::XMRRefOp>(op))
    return true;

  if (isa<sv::SampledOp>(op))
    return true;

  return false;
}

StringRef getSymOpName(Operation *symOp);

/// Return whether an operation is a constant.
static inline bool isConstantExpression(Operation *op) {
  return isa<hw::ConstantOp, sv::ConstantXOp, sv::ConstantZOp,
             sv::ConstantStrOp>(op);
}

/// This predicate returns true if the specified operation is considered a
/// potentially inlinable Verilog expression.  These nodes always have a single
/// result, but may have side effects (e.g. `sv.verbatim.expr.se`).
/// MemoryEffects should be checked if a client cares.
bool isVerilogExpression(Operation *op);

/// Return true if this is a zero bit type, e.g. a zero bit integer or array
/// thereof.
bool isZeroBitType(Type type);

/// Return true if this expression should be emitted inline into any statement
/// that uses it.
bool isExpressionEmittedInline(Operation *op, const LoweringOptions &options);

/// For each module we emit, do a prepass over the structure, pre-lowering and
/// otherwise rewriting operations we don't want to emit.
LogicalResult prepareHWModule(Block &block, const LoweringOptions &options);
LogicalResult prepareHWModule(hw::HWModuleOp module,
                              const LoweringOptions &options);

void pruneZeroValuedLogic(hw::HWModuleOp module);

/// Rewrite module names and interfaces to not conflict with each other or with
/// Verilog keywords.
GlobalNameTable legalizeGlobalNames(ModuleOp topLevel,
                                    const LoweringOptions &options);

} // namespace ExportVerilog
} // namespace circt

#endif // CONVERSION_EXPORTVERILOG_EXPORTVERILOGINTERNAL_H
