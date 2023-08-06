//===- LoweringOptions.h - CIRCT Lowering Options ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for controlling the lowering process and verilog exporting.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_LOWERINGOPTIONS_H
#define CIRCT_SUPPORT_LOWERINGOPTIONS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir {
class ModuleOp;
}

namespace circt {

/// Options which control the emission from CIRCT to Verilog.
struct LoweringOptions {
  /// Error callback type used to indicate errors parsing the options string.
  using ErrorHandlerT = llvm::function_ref<void(llvm::Twine)>;

  /// Create a LoweringOptions with the default values.
  LoweringOptions() = default;

  /// Create a LoweringOptions and read in options from a string,
  /// overriding only the set options in the string.
  LoweringOptions(llvm::StringRef options, ErrorHandlerT errorHandler);

  /// Create a LoweringOptions with values loaded from an MLIR ModuleOp. This
  /// loads a string attribute with the key `circt.loweringOptions`. If there is
  /// an error parsing the attribute this will print an error using the
  /// ModuleOp.
  LoweringOptions(mlir::ModuleOp module);

  /// Return the value of the `circt.loweringOptions` in the specified module
  /// if present, or a null attribute if not.
  static mlir::StringAttr getAttributeFrom(mlir::ModuleOp module);

  /// Read in options from a string, overriding only the set options in the
  /// string.
  void parse(llvm::StringRef options, ErrorHandlerT callback);

  /// Returns a string representation of the options.
  std::string toString() const;

  /// Write the verilog emitter options to a module's attributes.
  void setAsAttribute(mlir::ModuleOp module);

  /// Load any emitter options from the module. If there is an error validating
  /// the attribute, this will print an error using the ModuleOp.
  void parseFromAttribute(mlir::ModuleOp module);

  /// If true, emits `sv.alwayscomb` as Verilog `always @(*)` statements.
  /// Otherwise, print them as `always_comb`.
  bool noAlwaysComb = false;

  /// If true, expressions are allowed in the sensitivity list of `always`
  /// statements, otherwise they are forced to be simple wires. Some EDA
  /// tools rely on these being simple wires.
  bool allowExprInEventControl = false;

  /// If true, eliminate packed arrays for tools that don't support them (e.g.
  /// Yosys).
  bool disallowPackedArrays = false;

  /// If true, eliminate packed struct assignments in favor of a wire +
  /// assignments to the individual fields.
  bool disallowPackedStructAssignments = false;

  /// If true, do not emit SystemVerilog locally scoped "automatic" or logic
  /// declarations - emit top level wire and reg's instead.
  bool disallowLocalVariables = false;

  /// If true, verification statements like `assert`, `assume`, and `cover` will
  /// always be emitted with a label. If the statement has no label in the IR, a
  /// generic one will be created. Some EDA tools require verification
  /// statements to be labeled.
  bool enforceVerifLabels = false;

  /// This is the maximum number of terms in an expression before that
  /// expression spills a wire.
  enum { DEFAULT_TERM_LIMIT = 256 };
  unsigned maximumNumberOfTermsPerExpression = DEFAULT_TERM_LIMIT;

  /// This is the target width of lines in an emitted Verilog source file in
  /// columns.
  enum { DEFAULT_LINE_LENGTH = 90 };
  unsigned emittedLineLength = DEFAULT_LINE_LENGTH;

  /// Add an explicit bitcast for avoiding bitwidth mismatch LINT errors.
  bool explicitBitcast = false;

  /// If true, replicated ops are emitted to a header file.
  bool emitReplicatedOpsToHeader = false;

  /// This option controls emitted location information style.
  enum LocationInfoStyle {
    Plain,                 // Default.
    WrapInAtSquareBracket, // Wrap location info in @[..].
    None,                  // No location info comment.
  } locationInfoStyle = Plain;

  /// If true, every port is declared separately
  /// (each includes direction and type (e.g., `input [3:0]`)).
  /// When false (default), ports share declarations when possible.
  bool disallowPortDeclSharing = false;

  /// Print debug info.
  bool printDebugInfo = false;

  /// If true, every mux expression is spilled to a wire.
  bool disallowMuxInlining = false;

  /// This controls extra wire spilling performed in PrepareForEmission to
  /// improve readablitiy and debuggability.
  enum WireSpillingHeuristic : unsigned {
    SpillLargeTermsWithNamehints = 1, // Spill wires for expressions with
                                      // namehints if the term size is greater
                                      // than `wireSpillingNamehintTermLimit`.
  };

  unsigned wireSpillingHeuristicSet = 0;

  bool isWireSpillingHeuristicEnabled(WireSpillingHeuristic heurisic) const {
    return static_cast<bool>(wireSpillingHeuristicSet & heurisic);
  }

  enum { DEFAULT_NAMEHINT_TERM_LIMIT = 3 };
  unsigned wireSpillingNamehintTermLimit = DEFAULT_NAMEHINT_TERM_LIMIT;

  /// If true, every expression passed to an instance port is driven by a wire.
  /// Some lint tools dislike expressions being inlined into input ports so this
  /// option avoids such warnings.
  bool disallowExpressionInliningInPorts = false;

  /// If true, every expression used as an array index is driven by a wire, and
  /// the wire is marked as `(* keep = "true" *)`. Certain versions of Vivado
  /// produce incorrect synthesis results for certain arithmetic ops inlined
  /// into the array index.
  bool mitigateVivadoArrayIndexConstPropBug = false;

  /// If true, emit `wire` in port lists rather than nothing. Used in cases
  /// where `default_nettype is not set to wire.
  bool emitWireInPorts = false;

  /// If true, emit a comment wherever an instance wasn't printed, because
  /// it's emitted elsewhere as a bind.
  bool emitBindComments = false;

  /// If true, do not emit a version comment at the top of each verilog file.
  bool omitVersionComment = false;
};
} // namespace circt

#endif // CIRCT_SUPPORT_LOWERINGOPTIONS_H
