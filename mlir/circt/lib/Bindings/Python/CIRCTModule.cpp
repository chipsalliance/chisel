//===- CIRCTModule.cpp - Main pybind module -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/Comb.h"
#include "circt-c/Dialect/ESI.h"
#include "circt-c/Dialect/FSM.h"
#include "circt-c/Dialect/HW.h"
#include "circt-c/Dialect/HWArith.h"
#include "circt-c/Dialect/Handshake.h"
#include "circt-c/Dialect/MSFT.h"
#include "circt-c/Dialect/OM.h"
#include "circt-c/Dialect/SV.h"
#include "circt-c/Dialect/Seq.h"
#include "circt-c/ExportVerilog.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Transforms.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

#include "PybindUtils.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

static void registerPasses() {
  registerSeqPasses();
  registerSVPasses();
  registerFSMPasses();
  registerHWArithPasses();
  registerHandshakePasses();
  mlirRegisterTransformsPasses();
}

PYBIND11_MODULE(_circt, m) {
  m.doc() = "CIRCT Python Native Extension";
  registerPasses();
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  m.def(
      "register_dialects",
      [](py::object capsule) {
        // Get the MlirContext capsule from PyMlirContext capsule.
        auto wrappedCapsule = capsule.attr(MLIR_PYTHON_CAPI_PTR_ATTR);
        MlirContext context = mlirPythonCapsuleToContext(wrappedCapsule.ptr());

        // Collect CIRCT dialects to register.
        MlirDialectHandle comb = mlirGetDialectHandle__comb__();
        mlirDialectHandleRegisterDialect(comb, context);
        mlirDialectHandleLoadDialect(comb, context);

        MlirDialectHandle esi = mlirGetDialectHandle__esi__();
        mlirDialectHandleRegisterDialect(esi, context);
        mlirDialectHandleLoadDialect(esi, context);

        MlirDialectHandle msft = mlirGetDialectHandle__msft__();
        mlirDialectHandleRegisterDialect(msft, context);
        mlirDialectHandleLoadDialect(msft, context);

        MlirDialectHandle hw = mlirGetDialectHandle__hw__();
        mlirDialectHandleRegisterDialect(hw, context);
        mlirDialectHandleLoadDialect(hw, context);

        MlirDialectHandle hwarith = mlirGetDialectHandle__hwarith__();
        mlirDialectHandleRegisterDialect(hwarith, context);
        mlirDialectHandleLoadDialect(hwarith, context);

        MlirDialectHandle om = mlirGetDialectHandle__om__();
        mlirDialectHandleRegisterDialect(om, context);
        mlirDialectHandleLoadDialect(om, context);

        MlirDialectHandle seq = mlirGetDialectHandle__seq__();
        mlirDialectHandleRegisterDialect(seq, context);
        mlirDialectHandleLoadDialect(seq, context);

        MlirDialectHandle sv = mlirGetDialectHandle__sv__();
        mlirDialectHandleRegisterDialect(sv, context);
        mlirDialectHandleLoadDialect(sv, context);

        MlirDialectHandle fsm = mlirGetDialectHandle__fsm__();
        mlirDialectHandleRegisterDialect(fsm, context);
        mlirDialectHandleLoadDialect(fsm, context);

        MlirDialectHandle handshake = mlirGetDialectHandle__handshake__();
        mlirDialectHandleRegisterDialect(handshake, context);
        mlirDialectHandleLoadDialect(handshake, context);
      },
      "Register CIRCT dialects on a PyMlirContext.");

  m.def("export_verilog", [](MlirModule mod, py::object fileObject) {
    circt::python::PyFileAccumulator accum(fileObject, false);
    py::gil_scoped_release();
    mlirExportVerilog(mod, accum.getCallback(), accum.getUserData());
  });

  m.def("export_split_verilog", [](MlirModule mod, std::string directory) {
    auto cDirectory = mlirStringRefCreateFromCString(directory.c_str());
    mlirExportSplitVerilog(mod, cDirectory);
  });

  py::module esi = m.def_submodule("_esi", "ESI API");
  circt::python::populateDialectESISubmodule(esi);
  py::module msft = m.def_submodule("_msft", "MSFT API");
  circt::python::populateDialectMSFTSubmodule(msft);
  py::module hw = m.def_submodule("_hw", "HW API");
  circt::python::populateDialectHWSubmodule(hw);
  py::module om = m.def_submodule("_om", "OM API");
  circt::python::populateDialectOMSubmodule(om);
  py::module sv = m.def_submodule("_sv", "SV API");
  circt::python::populateDialectSVSubmodule(sv);
}
