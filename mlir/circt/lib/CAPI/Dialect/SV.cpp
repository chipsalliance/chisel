//===- SVDialect.cpp - C Interface for the SV Dialect -------------------===//
//
//  Implements a C Interface for the SV Dialect
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/SV.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::sv;

void registerSVPasses() { registerPasses(); }
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SystemVerilog, sv, SVDialect)

bool svAttrIsASVAttributeAttr(MlirAttribute cAttr) {
  return unwrap(cAttr).isa<SVAttributeAttr>();
}

MlirAttribute svSVAttributeAttrGet(MlirContext cCtxt, MlirStringRef cName,
                                   MlirStringRef cExpression,
                                   bool emitAsComment) {
  mlir::MLIRContext *ctxt = unwrap(cCtxt);
  mlir::StringAttr expr;
  if (cExpression.data != nullptr)
    expr = mlir::StringAttr::get(ctxt, unwrap(cExpression));
  return wrap(
      SVAttributeAttr::get(ctxt, mlir::StringAttr::get(ctxt, unwrap(cName)),
                           expr, mlir::BoolAttr::get(ctxt, emitAsComment)));
}

MlirStringRef svSVAttributeAttrGetName(MlirAttribute cAttr) {
  return wrap(unwrap(cAttr).cast<SVAttributeAttr>().getName().getValue());
}

MlirStringRef svSVAttributeAttrGetExpression(MlirAttribute cAttr) {
  auto expr = unwrap(cAttr).cast<SVAttributeAttr>().getExpression();
  if (expr)
    return wrap(expr.getValue());
  return {nullptr, 0};
}

bool svSVAttributeAttrGetEmitAsComment(MlirAttribute attribute) {
  return unwrap(attribute)
      .cast<SVAttributeAttr>()
      .getEmitAsComment()
      .getValue();
}
