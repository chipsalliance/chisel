//===- LLVM.h - Import and forward declare core LLVM types ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file forward declares and imports various common LLVM and MLIR datatypes
// that we want to use unqualified.
//
// Note that most of these are forward declared and then imported into the circt
// namespace with using decls, rather than being #included.  This is because we
// want clients to explicitly #include the files they need.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_LLVM_H
#define CIRCT_SUPPORT_LLVM_H

// MLIR includes a lot of forward declarations of LLVM types, use them.
#include "mlir/Support/LLVM.h"

// Can not forward declare inline functions with default arguments, so we
// include the header directly.
#include "mlir/Support/LogicalResult.h"

// Import classes from the `mlir` namespace into the `circt` namespace.  All of
// the following classes have been already forward declared and imported from
// `llvm` in to the `mlir` namespace. For classes with default template
// arguments, MLIR does not import the type directly, it creates a templated
// using statement. This is due to the limitiation that only one declaration of
// a type can have default arguments. For those types, it is important to import
// the MLIR version, and not the LLVM version. To keep things simple, all
// classes here should be imported from the `mlir` namespace, not the `llvm`
// namespace.
namespace circt {
using mlir::APFloat;          // NOLINT(misc-unused-using-decls)
using mlir::APInt;            // NOLINT(misc-unused-using-decls)
using mlir::APSInt;           // NOLINT(misc-unused-using-decls)
using mlir::ArrayRef;         // NOLINT(misc-unused-using-decls)
using mlir::BitVector;        // NOLINT(misc-unused-using-decls)
using mlir::cast;             // NOLINT(misc-unused-using-decls)
using mlir::cast_or_null;     // NOLINT(misc-unused-using-decls)
using mlir::DenseMap;         // NOLINT(misc-unused-using-decls)
using mlir::DenseMapInfo;     // NOLINT(misc-unused-using-decls)
using mlir::DenseSet;         // NOLINT(misc-unused-using-decls)
using mlir::dyn_cast;         // NOLINT(misc-unused-using-decls)
using mlir::dyn_cast_or_null; // NOLINT(misc-unused-using-decls)
using mlir::function_ref;     // NOLINT(misc-unused-using-decls)
using mlir::isa;              // NOLINT(misc-unused-using-decls)
using mlir::isa_and_nonnull;  // NOLINT(misc-unused-using-decls)
using mlir::iterator_range;   // NOLINT(misc-unused-using-decls)
using mlir::MutableArrayRef;  // NOLINT(misc-unused-using-decls)
using mlir::PointerUnion;     // NOLINT(misc-unused-using-decls)
using mlir::raw_ostream;      // NOLINT(misc-unused-using-decls)
using mlir::SetVector;        // NOLINT(misc-unused-using-decls)
using mlir::SmallPtrSet;      // NOLINT(misc-unused-using-decls)
using mlir::SmallPtrSetImpl;  // NOLINT(misc-unused-using-decls)
using mlir::SmallString;      // NOLINT(misc-unused-using-decls)
using mlir::SmallVector;      // NOLINT(misc-unused-using-decls)
using mlir::SmallVectorImpl;  // NOLINT(misc-unused-using-decls)
using mlir::StringLiteral;    // NOLINT(misc-unused-using-decls)
using mlir::StringRef;        // NOLINT(misc-unused-using-decls)
using mlir::StringSet;        // NOLINT(misc-unused-using-decls)
using mlir::TinyPtrVector;    // NOLINT(misc-unused-using-decls)
using mlir::Twine;            // NOLINT(misc-unused-using-decls)
using mlir::TypeSwitch;       // NOLINT(misc-unused-using-decls)
} // namespace circt

// Forward declarations of LLVM classes to be imported in to the circt
// namespace.
namespace llvm {
template <typename KeyT, typename ValueT, unsigned InlineBuckets,
          typename KeyInfoT, typename BucketT>
class SmallDenseMap;
template <typename T, unsigned N, typename C>
class SmallSet;
} // namespace llvm

// Import things we want into our namespace.
namespace circt {
using llvm::SmallDenseMap; // NOLINT(misc-unused-using-decls)
using llvm::SmallSet;      // NOLINT(misc-unused-using-decls)
} // namespace circt

// Forward declarations of classes to be imported in to the circt namespace.
namespace mlir {
class ArrayAttr;
class AsmParser;
class AsmPrinter;
class Attribute;
class Block;
class TypedAttr;
class IRMapping;
class BlockArgument;
class BoolAttr;
class Builder;
class NamedAttrList;
class ConversionPattern;
class ConversionPatternRewriter;
class ConversionTarget;
class DenseElementsAttr;
class Diagnostic;
class Dialect;
class DialectAsmParser;
class DialectAsmPrinter;
class DictionaryAttr;
class ElementsAttr;
class FileLineColLoc;
class FlatSymbolRefAttr;
class FloatAttr;
class FunctionType;
class FusedLoc;
class ImplicitLocOpBuilder;
class IndexType;
class InFlightDiagnostic;
class IntegerAttr;
class IntegerType;
class Location;
class LocationAttr;
class MemRefType;
class MLIRContext;
class ModuleOp;
class MutableOperandRange;
class NamedAttribute;
class NamedAttrList;
class NoneType;
class OpAsmDialectInterface;
class OpAsmParser;
class OpAsmPrinter;
class OpBuilder;
class OperandRange;
class Operation;
class OpFoldResult;
class OpOperand;
class OpResult;
template <typename OpTy>
class OwningOpRef;
class ParseResult;
class Pass;
class PatternRewriter;
class Region;
class RewritePatternSet;
class ShapedType;
class SplatElementsAttr;
class StringAttr;
class SymbolRefAttr;
class SymbolTable;
class SymbolTableCollection;
class TupleType;
class Type;
class TypeAttr;
class TypeConverter;
class TypeID;
class TypeRange;
class TypeStorage;
class UnitAttr;
class UnknownLoc;
class Value;
class ValueRange;
class VectorType;
class WalkResult;
enum class RegionKind;
struct CallInterfaceCallable;
struct LogicalResult;
struct OperationState;
class OperationName;

namespace affine {
struct MemRefAccess;
} // namespace affine

template <typename T>
class FailureOr;
template <typename SourceOp>
class OpConversionPattern;
template <typename T>
class OperationPass;
template <typename SourceOp>
struct OpRewritePattern;

using DefaultTypeStorage = TypeStorage;
using OpAsmSetValueNameFn = function_ref<void(Value, StringRef)>;

namespace OpTrait {}

} // namespace mlir

// Import things we want into our namespace.
namespace circt {
// clang-tidy removes following using directives incorrectly. So force
// clang-tidy to ignore them.
// TODO: It is better to use `NOLINTBEGIN/END` comments to disable clang-tidy
// than adding `NOLINT` to every line. `NOLINTBEGIN/END` will supported from
// clang-tidy-14.
using mlir::ArrayAttr;                 // NOLINT(misc-unused-using-decls)
using mlir::AsmParser;                 // NOLINT(misc-unused-using-decls)
using mlir::AsmPrinter;                // NOLINT(misc-unused-using-decls)
using mlir::Attribute;                 // NOLINT(misc-unused-using-decls)
using mlir::Block;                     // NOLINT(misc-unused-using-decls)
using mlir::BlockArgument;             // NOLINT(misc-unused-using-decls)
using mlir::BoolAttr;                  // NOLINT(misc-unused-using-decls)
using mlir::Builder;                   // NOLINT(misc-unused-using-decls)
using mlir::CallInterfaceCallable;     // NOLINT(misc-unused-using-decls)
using mlir::ConversionPattern;         // NOLINT(misc-unused-using-decls)
using mlir::ConversionPatternRewriter; // NOLINT(misc-unused-using-decls)
using mlir::ConversionTarget;          // NOLINT(misc-unused-using-decls)
using mlir::DefaultTypeStorage;        // NOLINT(misc-unused-using-decls)
using mlir::DenseElementsAttr;         // NOLINT(misc-unused-using-decls)
using mlir::Diagnostic;                // NOLINT(misc-unused-using-decls)
using mlir::Dialect;                   // NOLINT(misc-unused-using-decls)
using mlir::DialectAsmParser;          // NOLINT(misc-unused-using-decls)
using mlir::DialectAsmPrinter;         // NOLINT(misc-unused-using-decls)
using mlir::DictionaryAttr;            // NOLINT(misc-unused-using-decls)
using mlir::ElementsAttr;              // NOLINT(misc-unused-using-decls)
using mlir::failed;                    // NOLINT(misc-unused-using-decls)
using mlir::failure;                   // NOLINT(misc-unused-using-decls)
using mlir::FailureOr;                 // NOLINT(misc-unused-using-decls)
using mlir::FileLineColLoc;            // NOLINT(misc-unused-using-decls)
using mlir::FlatSymbolRefAttr;         // NOLINT(misc-unused-using-decls)
using mlir::FloatAttr;                 // NOLINT(misc-unused-using-decls)
using mlir::FunctionType;              // NOLINT(misc-unused-using-decls)
using mlir::FusedLoc;                  // NOLINT(misc-unused-using-decls)
using mlir::ImplicitLocOpBuilder;      // NOLINT(misc-unused-using-decls)
using mlir::IndexType;                 // NOLINT(misc-unused-using-decls)
using mlir::InFlightDiagnostic;        // NOLINT(misc-unused-using-decls)
using mlir::IntegerAttr;               // NOLINT(misc-unused-using-decls)
using mlir::IntegerType;               // NOLINT(misc-unused-using-decls)
using mlir::IRMapping;                 // NOLINT(misc-unused-using-decls)
using mlir::Location;                  // NOLINT(misc-unused-using-decls)
using mlir::LocationAttr;              // NOLINT(misc-unused-using-decls)
using mlir::LogicalResult;             // NOLINT(misc-unused-using-decls)
using mlir::MemRefType;                // NOLINT(misc-unused-using-decls)
using mlir::MLIRContext;               // NOLINT(misc-unused-using-decls)
using mlir::ModuleOp;                  // NOLINT(misc-unused-using-decls)
using mlir::MutableOperandRange;       // NOLINT(misc-unused-using-decls)
using mlir::NamedAttribute;            // NOLINT(misc-unused-using-decls)
using mlir::NamedAttrList;             // NOLINT(misc-unused-using-decls)
using mlir::NoneType;                  // NOLINT(misc-unused-using-decls)
using mlir::OpAsmDialectInterface;     // NOLINT(misc-unused-using-decls)
using mlir::OpAsmParser;               // NOLINT(misc-unused-using-decls)
using mlir::OpAsmPrinter;              // NOLINT(misc-unused-using-decls)
using mlir::OpAsmSetValueNameFn;       // NOLINT(misc-unused-using-decls)
using mlir::OpBuilder;                 // NOLINT(misc-unused-using-decls)
using mlir::OpConversionPattern;       // NOLINT(misc-unused-using-decls)
using mlir::OperandRange;              // NOLINT(misc-unused-using-decls)
using mlir::Operation;                 // NOLINT(misc-unused-using-decls)
using mlir::OperationName;             // NOLINT(misc-unused-using-decls)
using mlir::OperationPass;             // NOLINT(misc-unused-using-decls)
using mlir::OperationState;            // NOLINT(misc-unused-using-decls)
using mlir::OpFoldResult;              // NOLINT(misc-unused-using-decls)
using mlir::OpOperand;                 // NOLINT(misc-unused-using-decls)
using mlir::OpResult;                  // NOLINT(misc-unused-using-decls)
using mlir::OpRewritePattern;          // NOLINT(misc-unused-using-decls)
using mlir::OwningOpRef;               // NOLINT(misc-unused-using-decls)
using mlir::ParseResult;               // NOLINT(misc-unused-using-decls)
using mlir::Pass;                      // NOLINT(misc-unused-using-decls)
using mlir::PatternRewriter;           // NOLINT(misc-unused-using-decls)
using mlir::Region;                    // NOLINT(misc-unused-using-decls)
using mlir::RegionKind;                // NOLINT(misc-unused-using-decls)
using mlir::RewritePatternSet;         // NOLINT(misc-unused-using-decls)
using mlir::ShapedType;                // NOLINT(misc-unused-using-decls)
using mlir::SplatElementsAttr;         // NOLINT(misc-unused-using-decls)
using mlir::StringAttr;                // NOLINT(misc-unused-using-decls)
using mlir::succeeded;                 // NOLINT(misc-unused-using-decls)
using mlir::success;                   // NOLINT(misc-unused-using-decls)
using mlir::SymbolRefAttr;             // NOLINT(misc-unused-using-decls)
using mlir::SymbolTable;               // NOLINT(misc-unused-using-decls)
using mlir::SymbolTableCollection;     // NOLINT(misc-unused-using-decls)
using mlir::TupleType;                 // NOLINT(misc-unused-using-decls)
using mlir::Type;                      // NOLINT(misc-unused-using-decls)
using mlir::TypeAttr;                  // NOLINT(misc-unused-using-decls)
using mlir::TypeConverter;             // NOLINT(misc-unused-using-decls)
using mlir::TypedAttr;                 // NOLINT(misc-unused-using-decls)
using mlir::TypeID;                    // NOLINT(misc-unused-using-decls)
using mlir::TypeRange;                 // NOLINT(misc-unused-using-decls)
using mlir::TypeStorage;               // NOLINT(misc-unused-using-decls)
using mlir::UnitAttr;                  // NOLINT(misc-unused-using-decls)
using mlir::UnknownLoc;                // NOLINT(misc-unused-using-decls)
using mlir::Value;                     // NOLINT(misc-unused-using-decls)
using mlir::ValueRange;                // NOLINT(misc-unused-using-decls)
using mlir::VectorType;                // NOLINT(misc-unused-using-decls)
using mlir::WalkResult;                // NOLINT(misc-unused-using-decls)
using mlir::affine::MemRefAccess;      // NOLINT(misc-unused-using-decls)
namespace OpTrait = mlir::OpTrait;
} // namespace circt

#endif // CIRCT_SUPPORT_LLVM_H
