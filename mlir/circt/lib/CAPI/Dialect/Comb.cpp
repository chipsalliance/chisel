//===- SVDialect.cpp - C Interface for the Comb Dialect -------------------===//
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Comb.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Combinational, comb,
                                      circt::comb::CombDialect)
