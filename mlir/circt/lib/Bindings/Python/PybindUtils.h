//===- PybindUtils.h - Utilities for interop with python ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file copied from NPCOMP project. Omissions will be added.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDINGS_PYTHON_PYBINDUTILS_H
#define CIRCT_BINDINGS_PYTHON_PYBINDUTILS_H

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"

#include <optional>

namespace py = pybind11;

namespace circt {
namespace python {

/// Taken from PybindUtils.h in MLIR.
/// Accumulates into a python file-like object, either writing text (default)
/// or binary.
class PyFileAccumulator {
public:
  PyFileAccumulator(pybind11::object fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire();
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

private:
  pybind11::object pyWriteFunction;
  bool binary;
};
} // namespace python
} // namespace circt

namespace pybind11 {

/// Raises a python exception with the given message.
/// Correct usage:
//   throw RaiseValueError(PyExc_ValueError, "Foobar'd");
inline pybind11::error_already_set raisePyError(PyObject *exc_class,
                                                const char *message) {
  PyErr_SetString(exc_class, message);
  return pybind11::error_already_set();
}

/// Raises a value error with the given message.
/// Correct usage:
///   throw RaiseValueError("Foobar'd");
inline pybind11::error_already_set raiseValueError(const char *message) {
  return raisePyError(PyExc_ValueError, message);
}

/// Raises a value error with the given message.
/// Correct usage:
///   throw RaiseValueError(message);
inline pybind11::error_already_set raiseValueError(const std::string &message) {
  return raisePyError(PyExc_ValueError, message.c_str());
}

} // namespace pybind11

#endif // CIRCT_BINDINGS_PYTHON_PYBINDUTILS_H
