//===- Handshake.cpp - C Interface for the Handshake Dialect --------------===//
//
//  Implements a C Interface for the Handshake Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Handshake.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Transforms/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerHandshakePasses() {
  circt::handshake::registerPasses();
  circt::registerStandardToHandshakePass();
  circt::registerHandshakeToHWPass();
}
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Handshake, handshake,
                                      circt::handshake::HandshakeDialect)
