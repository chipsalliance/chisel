//===- OMModule.cpp - OM API pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"
#include "circt-c/Dialect/OM.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

namespace {

/// Provides an Object class by simply wrapping the OMObject CAPI.
struct Object {
  // Instantiate an Object with a reference to the underlying OMObject.
  Object(OMObject object) : object(object) {}
  Object(const Object &object) : object(object.object) {}

  /// Get the Type from an Object, which will be a ClassType.
  MlirType getType() { return omEvaluatorObjectGetType(object); }

  // Get a field from the Object, using pybind's support for variant to return a
  // Python object that is either an Object or Attribute.
  std::variant<Object, MlirAttribute> getField(const std::string &name) {
    // Wrap the requested field name in an attribute.
    MlirContext context = mlirTypeGetContext(omEvaluatorObjectGetType(object));
    MlirStringRef cName = mlirStringRefCreateFromCString(name.c_str());
    MlirAttribute nameAttr = mlirStringAttrGet(context, cName);

    // Get the field's ObjectValue via the CAPI.
    OMObjectValue result = omEvaluatorObjectGetField(object, nameAttr);

    // If the ObjectValue is null, something failed. Diagnostic handling is
    // implemented in pure Python, so nothing to do here besides throwing an
    // error to halt execution.
    if (omEvaluatorObjectValueIsNull(result))
      throw py::value_error("unable to get field, see previous error(s)");

    // If the field was an Object, return a new Object.
    if (omEvaluatorObjectValueIsAObject(result))
      return Object(omEvaluatorObjectValueGetObject(result));

    // If the field was a primitive, return the Attribute.
    assert(omEvaluatorObjectValueIsAPrimitive(result));
    return omEvaluatorObjectValueGetPrimitive(result);
  }

  // Get a list with the names of all the fields in the Object.
  std::vector<std::string> getFieldNames() {
    MlirAttribute fieldNames = omEvaluatorObjectGetFieldNames(object);
    intptr_t numFieldNames = mlirArrayAttrGetNumElements(fieldNames);

    std::vector<std::string> pyFieldNames;
    for (intptr_t i = 0; i < numFieldNames; ++i) {
      MlirAttribute fieldName = mlirArrayAttrGetElement(fieldNames, i);
      MlirStringRef fieldNameStr = mlirStringAttrGetValue(fieldName);
      pyFieldNames.emplace_back(fieldNameStr.data, fieldNameStr.length);
    }

    return pyFieldNames;
  }

private:
  // The underlying CAPI OMObject.
  OMObject object;
};

/// Provides an Evaluator class by simply wrapping the OMEvaluator CAPI.
struct Evaluator {
  // Instantiate an Evaluator with a reference to the underlying OMEvaluator.
  Evaluator(MlirModule mod) : evaluator(omEvaluatorNew(mod)) {}

  // Instantiate an Object.
  Object instantiate(MlirAttribute className,
                     std::vector<MlirAttribute> actualParams) {
    // Instantiate the Object via the CAPI.
    OMObject result = omEvaluatorInstantiate(
        evaluator, className, actualParams.size(), actualParams.data());

    // If the Object is null, something failed. Diagnostic handling is
    // implemented in pure Python, so nothing to do here besides throwing an
    // error to halt execution.
    if (omEvaluatorObjectIsNull(result))
      throw py::value_error(
          "unable to instantiate object, see previous error(s)");

    // Return a new Object.
    return Object(result);
  }

  // Get the Module the Evaluator is built from.
  MlirModule getModule() { return omEvaluatorGetModule(evaluator); }

private:
  // The underlying CAPI OMEvaluator.
  OMEvaluator evaluator;
};

class PyListAttrIterator {
public:
  PyListAttrIterator(MlirAttribute attr) : attr(std::move(attr)) {}

  PyListAttrIterator &dunderIter() { return *this; }

  MlirAttribute dunderNext() {
    if (nextIndex >= omListAttrGetNumElements(attr))
      throw py::stop_iteration();
    return omListAttrGetElement(attr, nextIndex++);
  }

  static void bind(py::module &m) {
    py::class_<PyListAttrIterator>(m, "ListAttributeIterator",
                                   py::module_local())
        .def("__iter__", &PyListAttrIterator::dunderIter)
        .def("__next__", &PyListAttrIterator::dunderNext);
  }

private:
  MlirAttribute attr;
  intptr_t nextIndex = 0;
};

} // namespace

/// Populate the OM Python module.
void circt::python::populateDialectOMSubmodule(py::module &m) {
  m.doc() = "OM dialect Python native extension";

  // Add the Evaluator class definition.
  py::class_<Evaluator>(m, "Evaluator")
      .def(py::init<MlirModule>(), py::arg("module"))
      .def("instantiate", &Evaluator::instantiate, "Instantiate an Object",
           py::arg("class_name"), py::arg("actual_params"))
      .def_property_readonly("module", &Evaluator::getModule,
                             "The Module the Evaluator is built from");

  // Add the Object class definition.
  py::class_<Object>(m, "Object")
      .def(py::init<Object>(), py::arg("object"))
      .def("__getattr__", &Object::getField, "Get a field from an Object",
           py::arg("name"))
      .def_property_readonly("field_names", &Object::getFieldNames,
                             "Get field names from an Object")
      .def_property_readonly("type", &Object::getType,
                             "The Type of the Object");

  // Add the ReferenceAttr definition
  mlir_attribute_subclass(m, "ReferenceAttr", omAttrIsAReferenceAttr)
      .def_property_readonly("inner_ref", [](MlirAttribute self) {
        return omReferenceAttrGetInnerRef(self);
      });

  // Add the OMListAttr definition
  mlir_attribute_subclass(m, "ListAttr", omAttrIsAListAttr)
      .def("__getitem__", &omListAttrGetElement)
      .def("__len__", &omListAttrGetElement)
      .def("__iter__",
           [](MlirAttribute arr) { return PyListAttrIterator(arr); });
  PyListAttrIterator::bind(m);

  // Add the ClassType class definition.
  mlir_type_subclass(m, "ClassType", omTypeIsAClassType);
}
