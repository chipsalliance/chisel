//===- DialectModules.h - Populate submodules -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions to populate each dialect's submodule (if provided).
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_BINDINGS_PYTHON_DIALECTMODULES_H
#define CIRCT_BINDINGS_PYTHON_DIALECTMODULES_H

#include <pybind11/pybind11.h>

namespace circt {
namespace python {

void populateDialectESISubmodule(pybind11::module &m);
void populateDialectHWSubmodule(pybind11::module &m);
void populateDialectMSFTSubmodule(pybind11::module &m);
void populateDialectOMSubmodule(pybind11::module &m);
void populateDialectSVSubmodule(pybind11::module &m);

} // namespace python
} // namespace circt

#endif // CIRCT_BINDINGS_PYTHON_DIALECTMODULES_H
