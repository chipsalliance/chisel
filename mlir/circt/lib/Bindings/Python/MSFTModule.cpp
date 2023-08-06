//===- MSFTModule.cpp - MSFT API pybind module ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DialectModules.h"

#include "circt-c/Dialect/MSFT.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "PybindUtils.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

using namespace circt;
using namespace circt::msft;
using namespace mlir::python::adaptors;

static py::handle getPhysLocationAttr(MlirAttribute attr) {
  return py::module::import("circt.dialects.msft")
      .attr("PhysLocationAttr")(attr)
      .release();
}

class PrimitiveDB {
public:
  PrimitiveDB(MlirContext ctxt) { db = circtMSFTCreatePrimitiveDB(ctxt); }
  ~PrimitiveDB() { circtMSFTDeletePrimitiveDB(db); }
  bool addPrimitive(MlirAttribute locAndPrim) {
    return mlirLogicalResultIsSuccess(
        circtMSFTPrimitiveDBAddPrimitive(db, locAndPrim));
  }
  bool isValidLocation(MlirAttribute loc) {
    return circtMSFTPrimitiveDBIsValidLocation(db, loc);
  }

  CirctMSFTPrimitiveDB db;
};

class PlacementDB {
public:
  PlacementDB(MlirModule top, PrimitiveDB *seed) {
    db = circtMSFTCreatePlacementDB(top, seed ? seed->db
                                              : CirctMSFTPrimitiveDB{nullptr});
  }
  ~PlacementDB() { circtMSFTDeletePlacementDB(db); }
  MlirOperation place(MlirOperation instOp, MlirAttribute loc,
                      std::string subpath, MlirLocation srcLoc) {
    auto cSubpath = mlirStringRefCreate(subpath.c_str(), subpath.size());
    return circtMSFTPlacementDBPlace(db, instOp, loc, cSubpath, srcLoc);
  }
  void removePlacement(MlirOperation locOp) {
    circtMSFTPlacementDBRemovePlacement(db, locOp);
  }
  bool movePlacement(MlirOperation locOp, MlirAttribute newLoc) {
    return mlirLogicalResultIsSuccess(
        circtMSFTPlacementDBMovePlacement(db, locOp, newLoc));
  }
  MlirOperation getInstanceAt(MlirAttribute loc) {
    return circtMSFTPlacementDBGetInstanceAt(db, loc);
  }
  py::handle getNearestFreeInColumn(CirctMSFTPrimitiveType prim,
                                    uint64_t column, uint64_t nearestToY) {
    MlirAttribute nearest = circtMSFTPlacementDBGetNearestFreeInColumn(
        db, prim, column, nearestToY);
    if (!nearest.ptr)
      return py::none();
    return getPhysLocationAttr(nearest);
  }
  void walkPlacements(
      py::function pycb,
      std::tuple<py::object, py::object, py::object, py::object> bounds,
      py::object prim, py::object walkOrder) {

    auto handleNone = [](py::object o) {
      return o.is_none() ? -1 : o.cast<int64_t>();
    };
    int64_t cBounds[4] = {
        handleNone(std::get<0>(bounds)), handleNone(std::get<1>(bounds)),
        handleNone(std::get<2>(bounds)), handleNone(std::get<3>(bounds))};
    CirctMSFTPrimitiveType cPrim;
    if (prim.is_none())
      cPrim = -1;
    else
      cPrim = prim.cast<CirctMSFTPrimitiveType>();

    CirctMSFTWalkOrder cWalkOrder;
    if (!walkOrder.is_none())
      cWalkOrder = walkOrder.cast<CirctMSFTWalkOrder>();
    else
      cWalkOrder = CirctMSFTWalkOrder{CirctMSFTDirection::NONE,
                                      CirctMSFTDirection::NONE};

    circtMSFTPlacementDBWalkPlacements(
        db,
        [](MlirAttribute loc, MlirOperation locOp, void *userData) {
          py::gil_scoped_acquire gil;
          py::function pycb = *((py::function *)(userData));
          pycb(loc, locOp);
        },
        cBounds, cPrim, cWalkOrder, &pycb);
  }

private:
  CirctMSFTPlacementDB db;
};

class PyLocationVecIterator {
public:
  /// Get item at the specified position, translating a nullptr to None.
  static py::handle getItem(MlirAttribute locVec, intptr_t pos) {
    MlirAttribute loc = circtMSFTLocationVectorAttrGetElement(locVec, pos);
    if (loc.ptr == nullptr)
      return py::none();
    return py::detail::type_caster<MlirAttribute>().cast(
        loc, py::return_value_policy::automatic, py::handle());
  }

  PyLocationVecIterator(MlirAttribute attr) : attr(attr) {}
  PyLocationVecIterator &dunderIter() { return *this; }

  py::handle dunderNext() {
    if (nextIndex >= circtMSFTLocationVectorAttrGetNumElements(attr)) {
      throw py::stop_iteration();
    }
    return getItem(attr, nextIndex++);
  }

  static void bind(py::module &m) {
    py::class_<PyLocationVecIterator>(m, "LocationVectorAttrIterator",
                                      py::module_local())
        .def("__iter__", &PyLocationVecIterator::dunderIter)
        .def("__next__", &PyLocationVecIterator::dunderNext);
  }

private:
  MlirAttribute attr;
  intptr_t nextIndex = 0;
};
/// Populate the msft python module.
void circt::python::populateDialectMSFTSubmodule(py::module &m) {
  mlirMSFTRegisterPasses();

  m.doc() = "MSFT dialect Python native extension";

  m.def("replaceAllUsesWith", &circtMSFTReplaceAllUsesWith);

  py::enum_<PrimitiveType>(m, "PrimitiveType")
      .value("M20K", PrimitiveType::M20K)
      .value("DSP", PrimitiveType::DSP)
      .value("FF", PrimitiveType::FF)
      .export_values();

  py::enum_<CirctMSFTDirection>(m, "Direction")
      .value("NONE", CirctMSFTDirection::NONE)
      .value("ASC", CirctMSFTDirection::ASC)
      .value("DESC", CirctMSFTDirection::DESC)
      .export_values();

  mlir_attribute_subclass(m, "PhysLocationAttr",
                          circtMSFTAttributeIsAPhysLocationAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, PrimitiveType devType, uint64_t x, uint64_t y,
             uint64_t num, MlirContext ctxt) {
            return cls(circtMSFTPhysLocationAttrGet(ctxt, (uint64_t)devType, x,
                                                    y, num));
          },
          "Create a physical location attribute", py::arg(),
          py::arg("dev_type"), py::arg("x"), py::arg("y"), py::arg("num"),
          py::arg("ctxt") = py::none())
      .def_property_readonly(
          "devtype",
          [](MlirAttribute self) {
            return (PrimitiveType)circtMSFTPhysLocationAttrGetPrimitiveType(
                self);
          })
      .def_property_readonly("x",
                             [](MlirAttribute self) {
                               return circtMSFTPhysLocationAttrGetX(self);
                             })
      .def_property_readonly("y",
                             [](MlirAttribute self) {
                               return circtMSFTPhysLocationAttrGetY(self);
                             })
      .def_property_readonly("num", [](MlirAttribute self) {
        return circtMSFTPhysLocationAttrGetNum(self);
      });

  mlir_attribute_subclass(m, "PhysicalBoundsAttr",
                          circtMSFTAttributeIsAPhysicalBoundsAttr)
      .def_classmethod(
          "get",
          [](py::object cls, uint64_t xMin, uint64_t xMax, uint64_t yMin,
             uint64_t yMax, MlirContext ctxt) {
            auto physicalBounds =
                circtMSFTPhysicalBoundsAttrGet(ctxt, xMin, xMax, yMin, yMax);
            return cls(physicalBounds);
          },
          "Create a PhysicalBounds attribute", py::arg("cls"), py::arg("xMin"),
          py::arg("xMax"), py::arg("yMin"), py::arg("yMax"),
          py::arg("context") = py::none());

  mlir_attribute_subclass(m, "LocationVectorAttr",
                          circtMSFTAttributeIsALocationVectorAttribute)
      .def_classmethod(
          "get",
          [](py::object cls, MlirType type, std::vector<py::handle> pylocs,
             MlirContext ctxt) {
            // Get a LocationVector being sensitive to None in the list of
            // locations.
            SmallVector<MlirAttribute> locs;
            for (auto attrHandle : pylocs)
              if (attrHandle.is_none())
                locs.push_back({nullptr});
              else
                locs.push_back(mlirPythonCapsuleToAttribute(
                    mlirApiObjectToCapsule(attrHandle).ptr()));
            return cls(circtMSFTLocationVectorAttrGet(ctxt, type, locs.size(),
                                                      locs.data()));
          },
          "Create a LocationVector attribute", py::arg("cls"), py::arg("type"),
          py::arg("locs"), py::arg("context") = py::none())
      .def("reg_type", &circtMSFTLocationVectorAttrGetType)
      .def("__len__", &circtMSFTLocationVectorAttrGetNumElements)
      .def("__getitem__", &PyLocationVecIterator::getItem,
           "Get the location at the specified position", py::arg("pos"))
      .def("__iter__",
           [](MlirAttribute arr) { return PyLocationVecIterator(arr); });
  PyLocationVecIterator::bind(m);

  py::class_<PrimitiveDB>(m, "PrimitiveDB")
      .def(py::init<MlirContext>(), py::arg("ctxt") = py::none())
      .def("add_primitive", &PrimitiveDB::addPrimitive,
           "Inform the DB about a new placement.", py::arg("loc_and_prim"))
      .def("is_valid_location", &PrimitiveDB::isValidLocation,
           "Query the DB as to whether or not a primitive exists.",
           py::arg("loc"));

  py::class_<PlacementDB>(m, "PlacementDB")
      .def(py::init<MlirModule, PrimitiveDB *>(), py::arg("top"),
           py::arg("seed") = nullptr)
      .def("place", &PlacementDB::place, "Place a dynamic instance.",
           py::arg("dyn_inst"), py::arg("location"), py::arg("subpath"),
           py::arg("src_location") = py::none())
      .def("remove_placement", &PlacementDB::removePlacement,
           "Remove a placement.", py::arg("location"))
      .def("move_placement", &PlacementDB::movePlacement,
           "Move a placement to another location.", py::arg("old_location"),
           py::arg("new_location"))
      .def("get_nearest_free_in_column", &PlacementDB::getNearestFreeInColumn,
           "Find the nearest free primitive location in column.",
           py::arg("prim_type"), py::arg("column"), py::arg("nearest_to_y"))
      .def("get_instance_at", &PlacementDB::getInstanceAt,
           "Get the instance at location. Returns None if nothing exists "
           "there. Otherwise, returns (path, subpath, op) of the instance "
           "there.")
      .def("walk_placements", &PlacementDB::walkPlacements,
           "Walk the placements, with possible bounds. Bounds are (xmin, xmax, "
           "ymin, ymax) with 'None' being unbounded.",
           py::arg("callback"),
           py::arg("bounds") =
               std::make_tuple(py::none(), py::none(), py::none(), py::none()),
           py::arg("prim_type") = py::none(),
           py::arg("walk_order") = py::none());

  py::class_<CirctMSFTWalkOrder>(m, "WalkOrder")
      .def(py::init<CirctMSFTDirection, CirctMSFTDirection>(),
           py::arg("columns") = CirctMSFTDirection::NONE,
           py::arg("rows") = CirctMSFTDirection::NONE);

  mlir_attribute_subclass(m, "AppIDAttr", circtMSFTAttributeIsAnAppIDAttr)
      .def_classmethod(
          "get",
          [](py::object cls, std::string name, uint64_t index,
             MlirContext ctxt) {
            return cls(circtMSFTAppIDAttrGet(ctxt, wrap(name), index));
          },
          "Create an AppID attribute", py::arg("cls"), py::arg("name"),
          py::arg("index"), py::arg("context") = py::none())
      .def_property_readonly("name",
                             [](MlirAttribute self) {
                               StringRef name =
                                   unwrap(circtMSFTAppIDAttrGetName(self));
                               return std::string(name.data(), name.size());
                             })
      .def_property_readonly("index", [](MlirAttribute self) {
        return circtMSFTAppIDAttrGetIndex(self);
      });
}
