//===- ESITranslations.cpp - ESI translations -------------------*- C++ -*-===//
//
// ESI translations:
// - Cap'nProto schema generation
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/Format.h"

#include <algorithm>

#ifdef CAPNP
#include "capnp/ESICapnp.h"
#include "circt/Dialect/ESI/CosimSchema.h"
#endif

using namespace circt;
using namespace circt::esi;

//===----------------------------------------------------------------------===//
// ESI Cosim Cap'nProto schema generation.
//
// Cosimulation in ESI is done over capnp. This translation walks the IR, finds
// all the `esi.cosim` ops, and creates a schema for all the types. It requires
// CAPNP to be enabled.
//===----------------------------------------------------------------------===//

#ifdef CAPNP

namespace {

struct ErrorCountingHandler : public mlir::ScopedDiagnosticHandler {
  ErrorCountingHandler(mlir::MLIRContext *context)
      : mlir::ScopedDiagnosticHandler(context) {
    setHandler([this](Diagnostic &diag) -> LogicalResult {
      if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
        ++errorCount;
      return failure();
    });
  }

  size_t errorCount = 0;
};

struct ExportCosimSchema {
  ExportCosimSchema(ModuleOp module, llvm::raw_ostream &os)
      : module(module), os(os), handler(module.getContext()),
        unknown(UnknownLoc::get(module.getContext())) {}

  /// Emit the whole schema.
  LogicalResult emit();

  /// Collect the types for which we need to emit a schema. Output some metadata
  /// comments.
  LogicalResult visitEndpoint(CosimEndpointOp);

private:
  ModuleOp module;
  llvm::raw_ostream &os;
  ErrorCountingHandler handler;
  const Location unknown;

  // All the `esi.cosim` input and output types encountered during the IR walk.
  // This is NOT in a deterministic order!
  llvm::SmallVector<std::shared_ptr<capnp::CapnpTypeSchema>> types;
};
} // anonymous namespace

LogicalResult ExportCosimSchema::visitEndpoint(CosimEndpointOp ep) {
  auto sendTypeSchema =
      std::make_shared<capnp::CapnpTypeSchema>(ep.getSend().getType());
  if (!sendTypeSchema->isSupported())
    return ep.emitOpError("Type ")
           << ep.getSend().getType() << " not supported.";
  types.push_back(sendTypeSchema);

  auto recvTypeSchema =
      std::make_shared<capnp::CapnpTypeSchema>(ep.getRecv().getType());
  if (!recvTypeSchema->isSupported())
    return ep.emitOpError("Type '")
           << ep.getRecv().getType() << "' not supported.";
  types.push_back(recvTypeSchema);

  os << "# Endpoint ";
  StringAttr epName = ep->getAttrOfType<StringAttr>("name");
  if (epName)
    os << epName << " endpoint at " << ep.getLoc() << ":\n";
  os << "#   Send type: ";
  sendTypeSchema->writeMetadata(os);
  os << "\n";

  os << "#   Recv type: ";
  recvTypeSchema->writeMetadata(os);
  os << "\n";

  return success();
}

static void emitCosimSchemaBody(llvm::raw_ostream &os) {
  StringRef entireSchemaFile = circt::esi::cosim::CosimSchema;
  size_t idLocation = entireSchemaFile.find("@0x");
  size_t newlineAfter = entireSchemaFile.find('\n', idLocation);

  os << "\n\n"
     << "#########################################################\n"
     << "## Standard RPC interfaces.\n"
     << "#########################################################\n";
  os << entireSchemaFile.substr(newlineAfter) << "\n";
}

LogicalResult ExportCosimSchema::emit() {
  os << "#########################################################\n"
     << "## ESI generated schema.\n"
     << "#########################################################\n";

  // Walk and collect the type data.
  auto walkResult = module.walk([this](CosimEndpointOp ep) {
    if (failed(visitEndpoint(ep)))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  os << "#########################################################\n";

  // We need a sorted list to ensure determinism.
  llvm::sort(types.begin(), types.end(),
             [](auto &a, auto &b) { return a->typeID() > b->typeID(); });

  // Compute and emit the capnp file id.
  uint64_t fileHash = 2544816649379317016; // Some random number.
  for (auto &schema : types)
    fileHash = llvm::hashing::detail::hash_16_bytes(fileHash, schema->typeID());
  // Capnp IDs always have a '1' high bit.
  fileHash |= 0x8000000000000000;
  capnp::emitCapnpID(os, fileHash) << ";\n\n";

  os << "#########################################################\n"
     << "## Types for your design.\n"
     << "#########################################################\n\n";
  // Iterate through the various types and emit their schemas.
  auto end = std::unique(
      types.begin(), types.end(),
      [&](const auto &lhs, const auto &rhs) { return *lhs == *rhs; });
  for (auto typeIter = types.begin(); typeIter != end; ++typeIter) {
    if (failed((*typeIter)->write(os)))
      // If we fail during an emission, dump out early since the output may be
      // corrupted.
      return failure();
  }

  // Include the RPC schema in each generated file.
  emitCosimSchemaBody(os);

  return success(handler.errorCount == 0);
}

LogicalResult circt::esi::exportCosimSchema(ModuleOp module,
                                            llvm::raw_ostream &os) {
  ExportCosimSchema schema(module, os);
  return schema.emit();
}

#else // Not CAPNP

LogicalResult circt::esi::exportCosimSchema(ModuleOp module,
                                            llvm::raw_ostream &os) {
  return mlir::emitError(UnknownLoc::get(module.getContext()),
                         "Not compiled with CAPNP support");
}

#endif

//===----------------------------------------------------------------------===//
// Register all ESI translations.
//===----------------------------------------------------------------------===//

void circt::esi::registerESITranslations() {
#ifdef CAPNP
  mlir::TranslateFromMLIRRegistration cosimToCapnp(
      "export-esi-capnp", "ESI Cosim Cap'nProto schema generation",
      exportCosimSchema, [](mlir::DialectRegistry &registry) {
        registry.insert<ESIDialect, circt::hw::HWDialect, circt::sv::SVDialect,
                        mlir::func::FuncDialect, mlir::BuiltinDialect>();
      });
#endif
}
