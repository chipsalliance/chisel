#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union

from .support import get_user_loc

from .circt.dialects import msft
from .circt.support import attribute_to_var
from .circt.ir import Attribute, StringAttr, ArrayAttr, FlatSymbolRefAttr

from functools import singledispatchmethod

PrimitiveType = msft.PrimitiveType


class PhysLocation:
  __slots__ = ["_loc"]

  @singledispatchmethod
  def __init__(self,
               prim_type: Union[str, PrimitiveType],
               x: int,
               y: int,
               num: Optional[int] = 0):

    if isinstance(prim_type, str):
      prim_type = getattr(PrimitiveType, prim_type)
    # TODO: Once we get into non-zero num primitives, this needs to be updated.
    if num is None:
      num = 0

    assert isinstance(prim_type, PrimitiveType)
    assert isinstance(x, int)
    assert isinstance(y, int)
    assert isinstance(num, int)
    self._loc = msft.PhysLocationAttr.get(prim_type, x, y, num)

  @__init__.register(msft.PhysLocationAttr)
  def __from_loc(self, loc):
    self._loc = loc

  @__init__.register(Attribute)
  def __from_attr(self, loc):
    self._loc = msft.PhysLocationAttr(loc)

  def __str__(self) -> str:
    loc = self._loc
    return f"PhysLocation<{loc.devtype}, x:{loc.x}, y:{loc.y}, num:{loc.num}>"

  def __repr__(self) -> str:
    return self.__str__()


class LocationVector:
  from .types import Type

  __slots__ = ["type", "_loc"]

  @singledispatchmethod
  def __init__(self, type: Type, locs: List[Optional[Tuple[int, int, int]]]):
    assert len(locs) == type.bitwidth, \
      "List length must match reg bitwidth"
    from .circt.dialects import msft as circt_msft
    self.type = type
    phys_locs: List[circt_msft.PhysLocationAttr] = list()
    for loc in locs:
      if loc is None:
        phys_locs.append(None)
      else:
        phys_locs.append(
            circt_msft.PhysLocationAttr.get(PrimitiveType.FF, loc[0], loc[1],
                                            loc[2]))
    self._loc = circt_msft.LocationVectorAttr.get(type._type, phys_locs)

  @__init__.register(msft.LocationVectorAttr)
  def __from_loc(self, loc):
    self._loc = loc

  @property
  def locs(self) -> List[PhysLocation]:
    return [PhysLocation(loc) if loc is not None else None for loc in self._loc]

  def __str__(self) -> str:
    locs = [f"{loc}" for loc in self.locs]
    return f"LocationVector<{self.type}, [" + ", ".join(locs) + "]>"

  def __repr__(self) -> str:
    return self.__str__()


class PhysicalRegion:
  _counter = 0
  _used_names = set([])

  __slots__ = ["_physical_region"]

  def __init__(self, name: str = None, bounds: list = None):
    if name is None or name in PhysicalRegion._used_names:
      prefix = name if name is not None else "region"
      name = f"{prefix}_{PhysicalRegion._counter}"
      while name in PhysicalRegion._used_names:
        PhysicalRegion._counter += 1
        name = f"{prefix}_{PhysicalRegion._counter}"
    PhysicalRegion._used_names.add(name)

    if bounds is None:
      bounds = []

    name_attr = StringAttr.get(name)
    bounds_attr = ArrayAttr.get(bounds)
    self._physical_region = msft.PhysicalRegionOp(name_attr, bounds_attr)

  def add_bounds(self, x_bounds: tuple, y_bounds: tuple):
    """Add a new bounding box to the region."""
    if (len(x_bounds) != 2):
      raise ValueError(f"expected lower and upper x bounds, got: {x_bounds}")
    if (len(y_bounds) != 2):
      raise ValueError(f"expected lower and upper y bounds, got: {y_bounds}")

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    bounds = msft.PhysicalBoundsAttr.get(x_min, x_max, y_min, y_max)

    self._physical_region.add_bounds(bounds)

    return self

  def get_ref(self):
    """Get a pair suitable for add_attribute to attach to an operation."""
    name = self._physical_region.sym_name.value
    return ("loc", msft.PhysicalRegionRefAttr.get(name))


class PrimitiveDB:
  __slots__ = ["_db"]

  def __init__(self):
    self._db = msft.PrimitiveDB()

  def add_coords(self,
                 prim_type: Union[str, PrimitiveType],
                 x: int,
                 y: int,
                 num: Optional[int] = None):
    self.add(PhysLocation(prim_type, x, y, num))

  def add(self, physloc: PhysLocation):
    self._db.add_primitive(physloc._loc)


class PlacementDB:
  from .instance import Instance

  __slots__ = ["_db", "_sys"]

  def __init__(self, sys, _circt_mod, seed: Optional[PrimitiveDB]):
    self._db = msft.PlacementDB(_circt_mod, seed._db if seed else None)
    self._sys = sys

  def get_instance_at(self, loc: PhysLocation):
    """Get the instance placed at `loc`. Returns (Instance, subpath)."""
    loc_op = self._db.get_instance_at(loc._loc)
    if loc_op is None:
      return None
    inst = self._sys._op_cache.get_or_create_inst_from_op(loc_op.parent.opview)
    subpath = attribute_to_var(loc_op.opview.subPath)
    return (inst, subpath)

  def reserve_location(self, loc: PhysLocation, entity: EntityExtern):
    sym_name = entity._entity_extern.sym_name.value
    ref = FlatSymbolRefAttr.get(sym_name)
    path = ArrayAttr.get([ref])
    subpath = ""
    self._db.add_placement(loc._loc, path, subpath, entity._entity_extern)

  def place(self, inst: Instance, loc: PhysLocation, subpath: str = ""):
    self._db.place(inst._dyn_inst.operation, loc._loc, subpath, get_user_loc())


class EntityExtern:
  __slots__ = ["_entity_extern"]

  def __init__(self, tag: str, metadata: Any = ""):
    self._entity_extern = msft.EntityExternOp.create(tag, metadata)
