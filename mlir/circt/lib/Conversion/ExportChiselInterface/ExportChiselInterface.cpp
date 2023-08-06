//===- ExportChiselInterface.cpp - Chisel Interface Emitter ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the Chisel interface emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportChiselInterface.h"
#include "../PassDetail.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/Version.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace circt;
using namespace firrtl;

#define DEBUG_TYPE "export-chisel-package"

//===----------------------------------------------------------------------===//
// Interface emission logic
//===----------------------------------------------------------------------===//

static const unsigned int indentIncrement = 2;

namespace {
class Emitter {
public:
  Emitter(llvm::raw_ostream &os) : os(os) {}

  bool hasEmittedProbeType() { return hasEmittedProbe; }

  /// Emits an `ExtModule` class with port declarations for `module`.
  LogicalResult emitModule(FModuleLike module) {
    os << "class " << module.getModuleName() << " extends ExtModule {\n";

    for (const auto &port : module.getPorts()) {
      if (failed(emitPort(port)))
        return failure();
    }

    os << "}\n";

    return success();
  }

private:
  /// Emits an `IO` for the `port`.
  LogicalResult emitPort(const PortInfo &port) {
    os.indent(indentIncrement) << "val " << port.getName() << " = IO(";
    if (failed(
            emitPortType(port.loc, port.type, port.direction, indentIncrement)))
      return failure();
    os << ")\n";

    return success();
  }

  /// Emits type construction expression for the port type, recursing into
  /// aggregate types as needed.
  LogicalResult emitPortType(Location location, Type type, Direction direction,
                             unsigned int indent,
                             bool hasEmittedDirection = false) {
    auto emitTypeWithArguments =
        [&]( // This is provided if the type is a base type, otherwise this is
             // null
            FIRRTLBaseType baseType, StringRef name,
            // A lambda of type (bool hasEmittedDirection) -> LogicalResult.
            auto emitArguments,
            // Indicates whether parentheses around type arguments should be
            // used.
            bool emitParentheses = true) -> LogicalResult {
      // Include the direction if the type is not a base (i.e. hardware) type or
      // is not composed of flips and analog signals and we haven't already
      // emitted the direction before recursing to this field.
      // Chisel direction functions override any internal directions. In other
      // words, Output(new Bundle {...}) erases all direction information inside
      // the bundle. Because of this, directions are placed on the outermost
      // passive members of a hardware type.
      bool emitDirection =
          !baseType || (!hasEmittedDirection && baseType.isPassive() &&
                        !baseType.containsAnalog());
      if (emitDirection) {
        switch (direction) {
        case Direction::In:
          os << "Input(";
          break;
        case Direction::Out:
          os << "Output(";
          break;
        }
      }

      bool emitConst = baseType && baseType.isConst();
      if (emitConst)
        os << "Const(";

      os << name;

      if (emitParentheses)
        os << "(";

      if (failed(emitArguments(hasEmittedDirection || emitDirection)))
        return failure();

      if (emitParentheses)
        os << ')';

      if (emitConst)
        os << ')';

      if (emitDirection)
        os << ')';

      return success();
    };

    // Emits a type that does not require arguments.
    auto emitType = [&](FIRRTLBaseType baseType,
                        StringRef name) -> LogicalResult {
      return emitTypeWithArguments(baseType, name,
                                   [](bool) { return success(); });
    };

    // Emits a type that requires a known width argument.
    auto emitWidthQualifiedType = [&](auto type,
                                      StringRef name) -> LogicalResult {
      auto width = type.getWidth();
      if (!width.has_value()) {
        return LogicalResult(emitError(
            location, "Expected width to be inferred for exported port"));
      }
      return emitTypeWithArguments(type, name, [&](bool) {
        os << *width << ".W";
        return success();
      });
    };

    return TypeSwitch<Type, LogicalResult>(type)
        .Case<ClockType>(
            [&](ClockType type) { return emitType(type, "Clock"); })
        .Case<AsyncResetType>(
            [&](AsyncResetType type) { return emitType(type, "AsyncReset"); })
        .Case<ResetType>([&](ResetType) {
          return emitError(
              location, "Expected reset type to be inferred for exported port");
        })
        .Case<UIntType>([&](UIntType uIntType) {
          return emitWidthQualifiedType(uIntType, "UInt");
        })
        .Case<SIntType>([&](SIntType sIntType) {
          return emitWidthQualifiedType(sIntType, "SInt");
        })
        .Case<AnalogType>([&](AnalogType analogType) {
          return emitWidthQualifiedType(analogType, "Analog");
        })
        .Case<BundleType>([&](BundleType bundleType) {
          // Emit an anonymous bundle, emitting a `val` for each field.
          return emitTypeWithArguments(
              bundleType, "new Bundle ",
              [&](bool hasEmittedDirection) {
                os << "{\n";
                unsigned int nestedIndent = indent + indentIncrement;
                for (const auto &element : bundleType.getElements()) {
                  os.indent(nestedIndent)
                      << "val " << element.name.getValue() << " = ";
                  auto elementResult = emitPortType(
                      location, element.type,
                      element.isFlip ? direction::flip(direction) : direction,
                      nestedIndent, hasEmittedDirection);
                  if (failed(elementResult))
                    return failure();
                  os << '\n';
                }
                os.indent(indent) << "}";
                return success();
              },
              false);
        })
        .Case<FVectorType>([&](FVectorType vectorType) {
          // Emit a vector type, emitting the type of its element as an
          // argument.
          return emitTypeWithArguments(
              vectorType, "Vec", [&](bool hasEmittedDirection) {
                os << vectorType.getNumElements() << ", ";
                return emitPortType(location, vectorType.getElementType(),
                                    direction, indent, hasEmittedDirection);
              });
        })
        .Case<RefType>([&](RefType refType) {
          hasEmittedProbe = true;
          StringRef name = refType.getForceable() ? "RWProbe" : "Probe";
          return emitTypeWithArguments(
              nullptr, name, [&](bool hasEmittedDirection) {
                return emitPortType(location, refType.getType(), direction,
                                    indent, hasEmittedDirection);
              });
        })
        .Default([&](Type type) {
          mlir::emitError(location) << "Unhandled type: " << type;
          return failure();
        });
  }

  llvm::raw_ostream &os;
  bool hasEmittedProbe = false;
};
} // namespace

/// Exports a Chisel interface to the output stream.
static LogicalResult exportChiselInterface(CircuitOp circuit,
                                           llvm::raw_ostream &os) {
  // Emit version, package, and import declarations
  os << circt::getCirctVersionComment() << "package shelf."
     << circuit.getName().lower()
     << "\n\nimport chisel3._\nimport chisel3.experimental._\n";

  std::string body;
  llvm::raw_string_ostream bodyStream(body);
  Emitter emitter(bodyStream);

  // Emit a class for the main circuit module.
  auto topModule = circuit.getMainModule();
  if (failed(emitter.emitModule(topModule)))
    return failure();

  // Emit an import for probe types if needed
  if (emitter.hasEmittedProbeType())
    os << "import chisel3.probe._\n";

  // Emit the body
  os << '\n' << body;

  return success();
}

/// Exports Chisel interface files for the circuit to the specified directory.
static LogicalResult exportSplitChiselInterface(CircuitOp circuit,
                                                StringRef outputDirectory) {
  // Create the output directory if needed.
  std::error_code error = llvm::sys::fs::create_directories(outputDirectory);
  if (error) {
    circuit.emitError("cannot create output directory \"")
        << outputDirectory << "\": " << error.message();
    return failure();
  }

  // Open the output file.
  SmallString<128> interfaceFilePath(outputDirectory);
  llvm::sys::path::append(interfaceFilePath, circuit.getName());
  llvm::sys::path::replace_extension(interfaceFilePath, "scala");
  std::string errorMessage;
  auto interfaceFile = mlir::openOutputFile(interfaceFilePath, &errorMessage);
  if (!interfaceFile) {
    circuit.emitError(errorMessage);
    return failure();
  }

  // Export the interface to the file.
  auto result = exportChiselInterface(circuit, interfaceFile->os());
  if (succeeded(result))
    interfaceFile->keep();
  return result;
}

//===----------------------------------------------------------------------===//
// ExportChiselInterfacePass and ExportSplitChiselInterfacePass
//===----------------------------------------------------------------------===//

namespace {
struct ExportChiselInterfacePass
    : public ExportChiselInterfaceBase<ExportChiselInterfacePass> {

  explicit ExportChiselInterfacePass(llvm::raw_ostream &os) : os(os) {}

  void runOnOperation() override {
    if (failed(exportChiselInterface(getOperation(), os)))
      signalPassFailure();
  }

private:
  llvm::raw_ostream &os;
};

struct ExportSplitChiselInterfacePass
    : public ExportSplitChiselInterfaceBase<ExportSplitChiselInterfacePass> {

  explicit ExportSplitChiselInterfacePass(StringRef directory) {
    directoryName = directory.str();
  }

  void runOnOperation() override {
    if (failed(exportSplitChiselInterface(getOperation(), directoryName)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass>
circt::createExportChiselInterfacePass(llvm::raw_ostream &os) {
  return std::make_unique<ExportChiselInterfacePass>(os);
}

std::unique_ptr<mlir::Pass>
circt::createExportSplitChiselInterfacePass(mlir::StringRef directory) {
  return std::make_unique<ExportSplitChiselInterfacePass>(directory);
}

std::unique_ptr<mlir::Pass> circt::createExportChiselInterfacePass() {
  return createExportChiselInterfacePass(llvm::outs());
}
