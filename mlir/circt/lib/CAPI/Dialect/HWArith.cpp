//===- HWArith.cpp - C Interface for the HWArith Dialect ----------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/HWArith.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerHWArithPasses() { circt::registerHWArithToHWPass(); }
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(HWArith, hwarith,
                                      circt::hwarith::HWArithDialect)
