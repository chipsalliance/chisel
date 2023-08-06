//===- ExportFIRRTL.cpp - C Interface to ExportFIRRTL ---------------------===//
//
//  Implements a C Interface for export FIRRTL.
//
//===----------------------------------------------------------------------===//

#include "circt-c/ExportFIRRTL.h"

#include "circt/Dialect/FIRRTL/FIREmitter.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace firrtl;

MlirLogicalResult mlirExportFIRRTL(MlirModule module,
                                   MlirStringCallback callback, void *userData,
                                   FIRVersion version) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(exportFIRFile(unwrap(module), stream, {}, {3, 0, 0}));
}
