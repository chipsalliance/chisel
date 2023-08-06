#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from .devicedb import LocationVector
from .module import AppID

from .circt.dialects import hw, msft, seq
from .circt import ir

from typing import Dict, Iterator, List, Optional, Tuple, Union


class Instance:
  """Represents a _specific_ instance, unique in a design. This is in contrast
  to a module instantiation within another module."""
  from .module import Module

  __slots__ = ["parent", "inside_of", "root", "symbol", "_op_cache"]

  def __init__(self, parent: Instance, inside_of: Module,
               symbol: Optional[ir.Attribute]):
    """
    Construct a new instance. Since the terminology can be confusing:
    - inside_of: the module which contains this instance (e.g. the instantiation
      site).
    - tgt_mod: if the instance is an instantation, `tgt_mod` is the module being
      instantiated. Examples of things which aren't instantiations:
      `seq.compreg`s.
    """

    self.inside_of = inside_of
    self.parent = parent
    self.root = parent.root
    if symbol is not None:
      self.symbol = hw.InnerSymAttr(symbol)
    else:
      self.symbol = None
    self._op_cache = parent.root.system._op_cache

  def _get_ip(self) -> ir.InsertionPoint:
    return ir.InsertionPoint(self._dyn_inst.body.blocks[0])

  def walk(self, callback):
    """Since by default this instance is a leaf, just callback self."""
    callback(self)

  @property
  def _inside_of_symbol(self) -> str:
    """Return the string symbol of the module which contains this instance."""
    return self._op_cache.get_pyproxy_symbol(self.inside_of)

  def __repr__(self) -> str:
    path_names = [i.name for i in self.path]
    return "<instance: [" + ", ".join(path_names) + "]>"

  @property
  def path_names(self):
    """A list of instance names representing the instance path."""
    return [i.name for i in self.path]

  def add_named_attribute(self,
                          name: str,
                          value: str,
                          subpath: Union[str, list[str]] = None):
    """Add an arbitrary named attribute to this instance."""
    if isinstance(subpath, list):
      subpath = "|".join(subpath)
    if subpath:
      subpath = "|" + subpath
    with self._get_ip():
      msft.DynamicInstanceVerbatimAttrOp(
          name=ir.StringAttr.get(name),
          value=ir.StringAttr.get(value),
          subPath=None if subpath is None else ir.StringAttr.get(subpath),
          ref=None)

  @property
  def _dyn_inst(self) -> msft.DynamicInstanceOp:
    """Returns the raw CIRCT op backing this Instance."""
    op = self._op_cache.create_or_get_dyn_inst(self)
    if op is None:
      raise InstanceDoesNotExistError(str(self))
    return op

  @property
  def path(self) -> list[Instance]:
    return self.parent.path + [self]

  @property
  def name(self) -> str:
    assert self.symbol is not None, \
           "If symbol is None, name() needs to be overridden"
    return self.symbol.symName.value

  @property
  def appid(self) -> Optional[AppID]:
    """Get the appid assigned to this instance or None if one was not assigned
    during generation."""
    static_op = self._op_cache.get_inner_sym_ops_in_module(
        self.inside_of)[self.symbol]
    if AppID.AttributeName in static_op.attributes:
      return AppID(static_op.attributes[AppID.AttributeName])
    return None


class ModuleInstance(Instance):
  """Instance specialization for modules. Since they are the only thing which
  can contain operations (for now), put all of the children stuff in here."""

  from .module import Module

  def __init__(self, parent: Instance, instance_sym: Optional[ir.Attribute],
               inside_of: Module, tgt_mod: Module):
    super().__init__(parent, inside_of, instance_sym)
    self.tgt_mod = tgt_mod
    self._child_cache: Dict[ir.StringAttr, Instance] = None
    self._add_appids()

  def _add_appids(self):
    """Implicitly add members for each AppID which is contained by this
    instance."""
    with self.root.system:
      circt_mod = self.tgt_mod.circt_mod
    if not isinstance(circt_mod,
                      msft.MSFTModuleOp) or circt_mod.childAppIDBases is None:
      return
    for name in [n.value for n in circt_mod.childAppIDBases]:
      if hasattr(self, name):
        continue
      self.__slots__.append(name)
      setattr(self, name, _AppIDInstance(self, name))

  def _create_instance(self, static_op: ir.Operation) -> Instance:
    """Create a new `Instance` which is a child of `parent` in the instance
    hierarchy and corresponds to the given static operation. The static
    operation need not be a module instantiation."""

    inner_sym = static_op.attributes["inner_sym"]
    if isinstance(static_op, msft.InstanceOp):
      tgt_mod = self._op_cache.get_symbol_pyproxy(static_op.moduleName)
      return ModuleInstance(self,
                            instance_sym=inner_sym,
                            inside_of=self.tgt_mod,
                            tgt_mod=tgt_mod)
    if isinstance(static_op, seq.CompRegOp):
      return RegInstance(self, self.tgt_mod, inner_sym, static_op)

    return Instance(self, self.tgt_mod, inner_sym)

  def _children(self) -> Dict[hw.InnerSymAttr, Instance]:
    """Return a dict of MLIR StringAttr this instances' children. Cache said
    list."""
    if self._child_cache is not None:
      return self._child_cache
    symbols_in_mod = self._op_cache.get_inner_sym_ops_in_module(self.tgt_mod)
    children = {
        sym: self._create_instance(op) for (sym, op) in symbols_in_mod.items()
    }
    # TODO: make these weak refs
    self._child_cache = children
    return children

  @property
  def children(self) -> Dict[str, Instance]:
    """Return a dict of python strings to this instances' children."""
    return {ir.StringAttr(key).value: inst for (key, inst) in self._children()}

  def __getitem__(self, child_name: str) -> Instance:
    """Get a child instance."""
    return self._children()[hw.InnerSymAttr.get(ir.StringAttr.get(child_name))]

  def walk(self, callback):
    """Descend the instance hierarchy, calling back on each instance."""
    callback(self)
    for child in self._children().values():
      child.walk(callback)

  def place(self,
            devtype: msft.PrimitiveType,
            x: int,
            y: int,
            num: int = 0,
            subpath: Union[str, list[str]] = ""):
    import pycde.devicedb as devdb
    if isinstance(subpath, list):
      subpath = "|".join(subpath)
    if subpath:
      subpath = "|" + subpath
    loc = devdb.PhysLocation(devtype, x, y, num)
    self.root.system.placedb.place(self, loc, subpath)

  @property
  def locations(self) -> List[Tuple[object, str]]:
    """Returns a list of physical locations assigned to this instance in
    (PhysLocation, subpath) format."""

    def conv(op):
      import pycde.devicedb as devdb
      loc = devdb.PhysLocation(op.loc)
      subPath = op.subPath
      if subPath is not None:
        subPath = ir.StringAttr(subPath).value
      return (loc, subPath)

    dyn_inst_block = self._dyn_inst.operation.regions[0].blocks[0]
    return [
        conv(op)
        for op in dyn_inst_block
        if isinstance(op, msft.PDPhysLocationOp)
    ]


class _AppIDInstance:
  """Helper class to provide accessors to AppID'd instances."""

  __slots__ = ["owner_instance", "appid_name"]

  def __init__(self, owner_instance: ModuleInstance, appid_name: str):
    self.owner_instance = owner_instance
    self.appid_name = appid_name

  def _search(self) -> Iterator[AppID, Instance]:
    inner_sym_ops = self.owner_instance._op_cache.get_inner_sym_ops_in_module(
        self.owner_instance.tgt_mod)
    for child in self.owner_instance._children().values():
      # "Look through" instance hierarchy levels if the child has an AppID
      # accessor for our name.
      if hasattr(child, self.appid_name):
        child_appid_inst = getattr(child, self.appid_name)
        if not isinstance(child_appid_inst, _AppIDInstance):
          continue
        for c in child_appid_inst._search():
          yield c

      # Check if the AppID is local, then return the instance if it is.
      instance_op = inner_sym_ops[child.symbol]
      if AppID.AttributeName in instance_op.attributes:
        try:
          # This is the only way to test that a certain attribute is a certain
          # attribute type. *Sigh*.
          appid = msft.AppIDAttr(instance_op.attributes[AppID.AttributeName])
          if appid.name == self.appid_name:
            yield (appid, child)
        except ValueError:
          pass

  def __getitem__(self, index: int) -> Instance:
    for (appid, child) in self._search():
      if appid.index == index:
        return child

    raise IndexError(f"{self.appid_name}[{index}] not found")

  def __iter__(self) -> Iterator[Instance]:
    for (_, child) in self._search():
      yield child


class RegInstance(Instance):
  """Instance specialization for registers."""

  from .module import Module

  __slots__ = ["type"]

  def __init__(self, parent: Instance, inside_of: Module,
               symbol: Optional[ir.Attribute], static_op: seq.CompRegOp):
    super().__init__(parent, inside_of, symbol)

    from .types import _FromCirctType
    self.type = _FromCirctType(static_op.operation.operands[0].type)

  def place(self, locs: Union[LocationVector, List[Optional[Tuple[int, int,
                                                                  int]]]]):
    import pycde.devicedb as devdb
    if isinstance(locs, devdb.LocationVector):
      vec = locs
    else:
      vec = devdb.LocationVector(self.type, locs)
    self.root.system.placedb.place(self, vec)


class InstanceHierarchyRoot(ModuleInstance):
  """
  A root of an instance hierarchy starting at top-level 'module'. Different from
  an `Instance` since it doesn't have an instance symbol (since it addresses the
  'top' module). Plus, CIRCT models it this way.
  """
  import pycde.system as cdesys
  from .module import Module

  def __init__(self, module: Module, instance_name: str, sys: cdesys.System):
    self.instance_name = instance_name
    self.system = sys
    self.root = self
    super().__init__(parent=self,
                     instance_sym=None,
                     inside_of=module,
                     tgt_mod=module)
    sys._op_cache.create_instance_hier_op(self)

  @property
  def _dyn_inst(self) -> msft.InstanceHierarchyOp:
    """Returns the raw CIRCT op backing this Instance."""
    op = self._op_cache.get_instance_hier_op(self)
    if op is None:
      raise InstanceDoesNotExistError(self.inside_of.modcls.__name__)
    return op

  @property
  def _cache_key(self):
    mod_sym = ir.FlatSymbolRefAttr.get(
        self._op_cache.get_pyproxy_symbol(self.inside_of))
    inst_name = None if self.instance_name is None else ir.StringAttr.get(
        self.instance_name)
    return (mod_sym, inst_name)

  @property
  def name(self) -> str:
    return "<<root>>"

  @property
  def path(self) -> list:
    return []


class InstanceError(Exception):
  """An error related to dynamic instances."""

  def __init__(self, msg: str):
    super().__init__(msg)


class InstanceDoesNotExistError(Exception):
  """The instance which you are trying to reach does not exist (anymore)"""

  def __init__(self, inst_str: str):
    super().__init__(f"Instance {self} does not exist (anymore)")
