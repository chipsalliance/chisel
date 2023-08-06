//===- ExportVerilog.cpp - C Interface to ExportVerilog -------------------===//
//
//  Implements a C Interface for export Verilog.
//
//===----------------------------------------------------------------------===//

#include "circt-c/ExportVerilog.h"

#include "circt/Conversion/ExportVerilog.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;

MlirLogicalResult mlirExportVerilog(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(exportVerilog(unwrap(module), stream));
}

MlirLogicalResult mlirExportSplitVerilog(MlirModule module,
                                         MlirStringRef directory) {
  return wrap(exportSplitVerilog(unwrap(module), unwrap(directory)));
}
