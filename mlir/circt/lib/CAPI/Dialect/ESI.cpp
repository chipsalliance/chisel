//===- ESI.cpp - C Interface for the ESI Dialect --------------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/ESI.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

using namespace circt::esi;
using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ESI, esi, circt::esi::ESIDialect)

void registerESIPasses() { circt::esi::registerESIPasses(); }

MlirLogicalResult circtESIExportCosimSchema(MlirModule module,
                                            MlirStringCallback callback,
                                            void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(circt::esi::exportCosimSchema(unwrap(module), stream));
}

bool circtESITypeIsAChannelType(MlirType type) {
  return unwrap(type).isa<ChannelType>();
}

MlirType circtESIChannelTypeGet(MlirType inner, uint32_t signaling) {
  auto signalEnum = symbolizeChannelSignaling(signaling);
  if (!signalEnum)
    return {};
  auto cppInner = unwrap(inner);
  return wrap(ChannelType::get(cppInner.getContext(), cppInner, *signalEnum));
}

MlirType circtESIChannelGetInner(MlirType channelType) {
  return wrap(unwrap(channelType).cast<ChannelType>().getInner());
}
uint32_t circtESIChannelGetSignaling(MlirType channelType) {
  return (uint32_t)unwrap(channelType).cast<ChannelType>().getSignaling();
}

bool circtESITypeIsAnAnyType(MlirType type) {
  return unwrap(type).isa<AnyType>();
}
MlirType circtESIAnyTypeGet(MlirContext ctxt) {
  return wrap(AnyType::get(unwrap(ctxt)));
}

bool circtESITypeIsAListType(MlirType type) {
  return unwrap(type).isa<ListType>();
}

MlirType circtESIListTypeGet(MlirType inner) {
  auto cppInner = unwrap(inner);
  return wrap(ListType::get(cppInner.getContext(), cppInner));
}

MlirType circtESIListTypeGetElementType(MlirType list) {
  return wrap(unwrap(list).cast<ListType>().getElementType());
}

MlirOperation circtESIWrapModule(MlirOperation cModOp, long numPorts,
                                 const MlirStringRef *ports) {
  mlir::Operation *modOp = unwrap(cModOp);
  llvm::SmallVector<llvm::StringRef, 8> portNamesRefs;
  for (long i = 0; i < numPorts; ++i)
    portNamesRefs.push_back(ports[i].data);
  llvm::SmallVector<ESIPortValidReadyMapping, 8> portTriples;
  resolvePortNames(modOp, portNamesRefs, portTriples);
  mlir::OpBuilder b(modOp);
  mlir::Operation *wrapper = buildESIWrapper(b, modOp, portTriples);
  return wrap(wrapper);
}

void circtESIAppendMlirFile(MlirModule cMod, MlirStringRef filename) {
  ModuleOp modOp = unwrap(cMod);
  auto loadedMod =
      parseSourceFile<ModuleOp>(unwrap(filename), modOp.getContext());
  Block *loadedBlock = loadedMod->getBody();
  assert(!modOp->getRegions().empty());
  if (modOp.getBodyRegion().empty()) {
    modOp.getBodyRegion().push_back(loadedBlock);
    return;
  }
  auto &ops = modOp.getBody()->getOperations();
  ops.splice(ops.end(), loadedBlock->getOperations());
}
MlirOperation circtESILookup(MlirModule mod, MlirStringRef symbol) {
  return wrap(SymbolTable::lookupSymbolIn(unwrap(mod), unwrap(symbol)));
}

void circtESIRegisterGlobalServiceGenerator(
    MlirStringRef impl_type, CirctESIServiceGeneratorFunc genFunc,
    void *userData) {
  ServiceGeneratorDispatcher::globalDispatcher().registerGenerator(
      unwrap(impl_type), [genFunc, userData](ServiceImplementReqOp req,
                                             ServiceDeclOpInterface decl) {
        return unwrap(genFunc(wrap(req), wrap(decl.getOperation()), userData));
      });
}
