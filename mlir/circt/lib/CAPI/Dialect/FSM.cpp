//===- FSMDialect.cpp - C Interface for the FSM Dialect -------------------===//
//
//  Implements a C Interface for the FSM Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FSM.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::fsm;

void registerFSMPasses() {
  registerPasses();
  circt::registerConvertFSMToSVPass();
}
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FSM, fsm, FSMDialect)
