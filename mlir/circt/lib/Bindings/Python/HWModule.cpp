//===- HWModule.cpp - HW API pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/HW.h"

#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "PybindUtils.h"
#include "mlir-c/Support.h"
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace mlir::python::adaptors;

/// Populate the hw python module.
void circt::python::populateDialectHWSubmodule(py::module &m) {
  m.doc() = "HW dialect Python native extension";

  m.def("get_bitwidth", &hwGetBitWidth);

  mlir_type_subclass(m, "InOutType", hwTypeIsAInOut)
      .def_classmethod("get",
                       [](py::object cls, MlirType innerType) {
                         return cls(hwInOutTypeGet(innerType));
                       })
      .def_property_readonly("element_type", [](MlirType self) {
        return hwInOutTypeGetElementType(self);
      });

  mlir_type_subclass(m, "ArrayType", hwTypeIsAArrayType)
      .def_classmethod("get",
                       [](py::object cls, MlirType elementType, intptr_t size) {
                         return cls(hwArrayTypeGet(elementType, size));
                       })
      .def_property_readonly(
          "element_type",
          [](MlirType self) { return hwArrayTypeGetElementType(self); })
      .def_property_readonly(
          "size", [](MlirType self) { return hwArrayTypeGetSize(self); });

  mlir_type_subclass(m, "ParamIntType", hwTypeIsAIntType)
      .def_classmethod(
          "get_from_param",
          [](py::object cls, MlirContext ctx, MlirAttribute param) {
            return cls(hwParamIntTypeGet(param));
          })
      .def_property_readonly("width", [](MlirType self) {
        return hwParamIntTypeGetWidthAttr(self);
      });

  mlir_type_subclass(m, "StructType", hwTypeIsAStructType)
      .def_classmethod(
          "get",
          [](py::object cls, py::list pyFieldInfos) {
            llvm::SmallVector<HWStructFieldInfo> mlirFieldInfos;
            MlirContext ctx;

            // Since we're just passing string refs to the type constructor,
            // copy them into a temporary vector to give them all new addresses.
            llvm::SmallVector<llvm::SmallString<8>> names;
            for (size_t i = 0, e = pyFieldInfos.size(); i < e; ++i) {
              auto tuple = pyFieldInfos[i].cast<py::tuple>();
              auto type = tuple[1].cast<MlirType>();
              ctx = mlirTypeGetContext(type);
              names.emplace_back(tuple[0].cast<std::string>());
              auto nameStringRef =
                  mlirStringRefCreate(names[i].data(), names[i].size());
              mlirFieldInfos.push_back(HWStructFieldInfo{
                  mlirIdentifierGet(ctx, nameStringRef), type});
            }
            return cls(hwStructTypeGet(ctx, mlirFieldInfos.size(),
                                       mlirFieldInfos.data()));
          })
      .def("get_field",
           [](MlirType self, std::string fieldName) {
             return hwStructTypeGetField(
                 self, mlirStringRefCreateFromCString(fieldName.c_str()));
           })
      .def("get_fields", [](MlirType self) {
        intptr_t num_fields = hwStructTypeGetNumFields(self);
        py::list fields;
        for (intptr_t i = 0; i < num_fields; ++i) {
          auto field = hwStructTypeGetFieldNum(self, i);
          auto fieldName = mlirIdentifierStr(field.name);
          std::string name(fieldName.data, fieldName.length);
          fields.append(py::make_tuple(name, field.type));
        }
        return fields;
      });

  mlir_type_subclass(m, "TypeAliasType", hwTypeIsATypeAliasType)
      .def_classmethod("get",
                       [](py::object cls, std::string scope, std::string name,
                          MlirType innerType) {
                         return cls(hwTypeAliasTypeGet(
                             mlirStringRefCreateFromCString(scope.c_str()),
                             mlirStringRefCreateFromCString(name.c_str()),
                             innerType));
                       })
      .def_property_readonly(
          "canonical_type",
          [](MlirType self) { return hwTypeAliasTypeGetCanonicalType(self); })
      .def_property_readonly(
          "inner_type",
          [](MlirType self) { return hwTypeAliasTypeGetInnerType(self); })
      .def_property_readonly("name",
                             [](MlirType self) {
                               MlirStringRef cStr =
                                   hwTypeAliasTypeGetName(self);
                               return std::string(cStr.data, cStr.length);
                             })
      .def_property_readonly("scope", [](MlirType self) {
        MlirStringRef cStr = hwTypeAliasTypeGetScope(self);
        return std::string(cStr.data, cStr.length);
      });

  mlir_attribute_subclass(m, "ParamDeclAttr", hwAttrIsAParamDeclAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::string name, MlirType type,
             MlirAttribute value) {
            return cls(hwParamDeclAttrGet(
                mlirStringRefCreateFromCString(name.c_str()), type, value));
          })
      .def_classmethod("get_nodefault",
                       [](py::object cls, std::string name, MlirType type) {
                         return cls(hwParamDeclAttrGet(
                             mlirStringRefCreateFromCString(name.c_str()), type,
                             MlirAttribute{nullptr}));
                       })
      .def_property_readonly(
          "value",
          [](MlirAttribute self) { return hwParamDeclAttrGetValue(self); })
      .def_property_readonly(
          "param_type",
          [](MlirAttribute self) { return hwParamDeclAttrGetType(self); })
      .def_property_readonly("name", [](MlirAttribute self) {
        MlirStringRef cStr = hwParamDeclAttrGetName(self);
        return std::string(cStr.data, cStr.length);
      });

  mlir_attribute_subclass(m, "ParamDeclRefAttr", hwAttrIsAParamDeclRefAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirContext ctx, std::string name) {
            return cls(hwParamDeclRefAttrGet(
                ctx, mlirStringRefCreateFromCString(name.c_str())));
          })
      .def_property_readonly(
          "param_type",
          [](MlirAttribute self) { return hwParamDeclRefAttrGetType(self); })
      .def_property_readonly("name", [](MlirAttribute self) {
        MlirStringRef cStr = hwParamDeclRefAttrGetName(self);
        return std::string(cStr.data, cStr.length);
      });

  mlir_attribute_subclass(m, "ParamVerbatimAttr", hwAttrIsAParamVerbatimAttr)
      .def_classmethod("get", [](py::object cls, MlirAttribute text) {
        return cls(hwParamVerbatimAttrGet(text));
      });

  mlir_attribute_subclass(m, "InnerSymAttr", hwAttrIsAInnerSymAttr)
      .def_classmethod("get",
                       [](py::object cls, MlirAttribute symName) {
                         return cls(hwInnerSymAttrGet(symName));
                       })
      .def_property_readonly("symName", [](MlirAttribute self) {
        return hwInnerSymAttrGetSymName(self);
      });

  mlir_attribute_subclass(m, "InnerRefAttr", hwAttrIsAInnerRefAttr)
      .def_classmethod(
          "get",
          [](py::object cls, MlirAttribute moduleName, MlirAttribute innerSym) {
            return cls(hwInnerRefAttrGet(moduleName, innerSym));
          })
      .def_property_readonly(
          "module",
          [](MlirAttribute self) { return hwInnerRefAttrGetModule(self); })
      .def_property_readonly("name", [](MlirAttribute self) {
        return hwInnerRefAttrGetName(self);
      });

  mlir_attribute_subclass(m, "GlobalRefAttr", hwAttrIsAGlobalRefAttr)
      .def_classmethod("get", [](py::object cls, MlirAttribute symName) {
        return cls(hwGlobalRefAttrGet(symName));
      });
}
