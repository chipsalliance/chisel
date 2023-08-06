#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import capnp
import os
from pathlib import Path
import time
import typing


class Type:

  def __init__(self, width, type_id: typing.Optional[int] = None):
    self.type_id = type_id
    self.width = width

  def is_valid(self, obj) -> bool:
    """Is a Python object compatible with HW type."""
    assert False, "unimplemented"


class VoidType(Type):

  def __init__(self, type_id: typing.Optional[int] = None):
    super().__init__(0, type_id)

  def is_valid(self, obj) -> bool:
    return obj is None


class IntType(Type):

  def __init__(self,
               width: int,
               signed: bool,
               type_id: typing.Optional[int] = None):
    super().__init__(width, type_id)
    self.signed = signed

  def is_valid(self, obj) -> bool:
    if self.width == 0:
      return obj is None
    if not isinstance(obj, int):
      return False
    if obj >= 2**self.width:
      return False
    return True

  def __str__(self):
    return ("" if self.signed else "u") + \
      f"int{self.width}"


class StructType(Type):

  def __init__(self,
               fields: typing.List[typing.Tuple[str, Type]],
               type_id: typing.Optional[int] = None):
    self.fields = fields
    width = sum([ftype.width for (_, ftype) in self.fields])
    super().__init__(width, type_id)

  def is_valid(self, obj) -> bool:
    fields_count = 0
    if isinstance(obj, dict):
      for (fname, ftype) in self.fields:
        if fname not in obj:
          return False
        if not ftype.is_valid(obj[fname]):
          return False
        fields_count += 1
      if fields_count != len(obj):
        return False
      return True
    return False


class Port:

  def __init__(self,
               client_path: typing.List[str],
               backend,
               impl_type: str,
               read_type: typing.Optional[Type] = None,
               write_type: typing.Optional[Type] = None):
    # If a backend doesn't support a particular implementation type, just skip
    # it. We don't want to error out on services which aren't being used.
    if backend.supports_impl(impl_type):
      self._backend = backend.get_port(client_path, read_type, write_type)
    else:
      self._backend = None
    self.client_path = client_path
    self.read_type = read_type
    self.write_type = write_type


class WritePort(Port):

  def write(self, msg=None) -> bool:
    assert self.write_type is not None, "Expected non-None write_type"
    if not self.write_type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.write_type}'")
    if self._backend is None:
      raise ValueError("Backend does not support implementation of port")
    return self._backend.write(msg)


class ReadPort(Port):

  def read(self, blocking_timeout: typing.Optional[float] = 1.0):
    if self._backend is None:
      raise ValueError("Backend does not support implementation of port")
    return self._backend.read(blocking_timeout)


class ReadWritePort(Port):

  def __call__(self,
               msg=None,
               blocking_timeout: typing.Optional[float] = 1.0) -> typing.Any:
    """Send a message and wait for a response. If 'timeout' is exceeded while
    waiting for a response, there may well be one coming. It is the caller's
    responsibility to clear the response channel before sending another request
    so as to ensure correlation between request and response.

    Intended for blocking or synchronous interfaces."""

    if not self.write(msg):
      raise RuntimeError(f"Could not send message '{msg}'")
    return self.read(blocking_timeout)

  def write(self, msg=None) -> bool:
    assert self.write_type is not None, "Expected non-None write_type"
    if not self.write_type.is_valid(msg):
      raise ValueError(f"'{msg}' cannot be converted to '{self.write_type}'")
    return self._backend.write(msg)

  def read(self, blocking_timeout: typing.Optional[float] = 1.0):
    return self._backend.read(blocking_timeout)


class _CosimNode:
  """Provides a capnp-based co-simulation backend."""

  def __init__(self, root, prefix: typing.List[str]):
    self._root: Cosim = root
    self._endpoint_prefix = prefix

  def supports_impl(self, impl_type: str) -> bool:
    """The cosim backend only supports cosim connectivity implementations."""
    return impl_type == "cosim"

  def get_child(self, child_name: str):
    """When instantiating a child instance, get the backend node with which it
    is associated."""
    child_path = self._endpoint_prefix + [child_name]
    return _CosimNode(self._root, child_path)

  def get_port(self,
               client_path: typing.List[str],
               read_type: typing.Optional[Type] = None,
               write_type: typing.Optional[Type] = None):
    """When building a service port, get the backend port which it should use
    for interactions."""
    path = ".".join(self._endpoint_prefix) + "." + "_".join(client_path)
    ep = self._root._open_endpoint(
        path,
        write_type=write_type.type_id if write_type is not None else None,
        read_type=read_type.type_id if read_type is not None else None)
    return _CosimPort(self, ep, read_type, write_type)


class Cosim(_CosimNode):
  """Connect to a Cap'N Proto RPC co-simulation and provide a cosim backend
  service."""

  def __init__(self, schemaPath, hostPort):
    """Load the schema and connect to the RPC server"""
    self._schema = capnp.load(schemaPath)
    self._rpc_client = capnp.TwoPartyClient(hostPort)
    self._cosim = self._rpc_client.bootstrap().cast_as(
        self._schema.CosimDpiServer)

    # Find the simulation prefix and use it in our parent constructor.
    ifaces = self.list()
    prefix = [] if len(ifaces) == 0 else ifaces[0].endpointID.split(".")[:1]
    super().__init__(self, prefix)

  def load_package(path: os.PathLike):
    """Load a cosim connection from something running out of 'path' package dir.
    Reads and parses 'cosim.cfg' from that directory to get the connection
    information. Loads the capnp schema from the 'runtime' directory in that
    package path."""
    path = Path(path)
    simcfg = path / "cosim.cfg"
    if not simcfg.exists():
      simcfg = Path.cwd() / "cosim.cfg"
      if not simcfg.exists():
        raise RuntimeError("Could not find simulation connection file")
    port_lines = filter(lambda x: x.startswith("port:"),
                        simcfg.open().readlines())
    port = int(list(port_lines)[0].split(":")[1])
    return Cosim(os.path.join(path, "runtime", "schema.capnp"),
                 f"{os.uname()[1]}:{port}")

  def list(self):
    """List the available interfaces"""
    return self._cosim.list().wait().ifaces

  def _open_endpoint(self, epid: str, write_type=None, read_type=None):
    """Open the endpoint, optionally checking the send and recieve types"""
    for iface in self.list():
      if iface.endpointID == epid:
        # Optionally check that the type IDs match.
        if write_type is not None:
          assert iface.sendTypeID == write_type.schema.node.id
        else:
          assert write_type is None
        if read_type is not None:
          assert iface.recvTypeID == read_type.schema.node.id
        else:
          assert read_type is None

        openResp = self._cosim.open(iface).wait()
        assert openResp.iface is not None
        return openResp.iface
    assert False, f"Could not find specified EndpointID: {epid}"


class _CosimPort:
  """Cosim backend for service ports. This is where the real meat is buried."""

  class _TypeConverter:
    """Parent class for Capnp type converters."""

    def __init__(self, schema, esi_type: Type):
      self.esi_type = esi_type
      assert hasattr(esi_type, "capnp_name")
      if not hasattr(schema, esi_type.capnp_name):
        raise ValueError("Cosim does not support non-capnp types.")
      self.capnp_type = getattr(schema, esi_type.capnp_name)

  class _VoidConverter(_TypeConverter):
    """Convert python ints to and from capnp messages."""

    def write(self, py_int: None):
      return self.capnp_type.new_message()

    def read(self, capnp_resp) -> None:
      return capnp_resp.as_struct(self.capnp_type)

  class _IntConverter(_TypeConverter):
    """Convert python ints to and from capnp messages."""

    def write(self, py_int: int):
      return self.capnp_type.new_message(i=py_int)

    def read(self, capnp_resp) -> int:
      return capnp_resp.as_struct(self.capnp_type).i

  class _StructConverter(_TypeConverter):
    """Convert python ints to and from capnp messages."""

    def write(self, py_dict: dict):
      return self.capnp_type.new_message(**py_dict)

    def read(self, capnp_resp) -> int:
      capnp_msg = capnp_resp.as_struct(self.capnp_type)
      ret = {}
      for (fname, _) in self.esi_type.fields:
        if hasattr(capnp_msg, fname):
          ret[fname] = getattr(capnp_msg, fname)
      return ret

  # Lookup table for getting the correct type converter for a given type.
  ConvertLookup = {
      VoidType: _VoidConverter,
      IntType: _IntConverter,
      StructType: _StructConverter
  }

  def __init__(self, node: _CosimNode, endpoint,
               read_type: typing.Optional[Type],
               write_type: typing.Optional[Type]):
    self._endpoint = endpoint
    schema = node._root._schema
    # For each type, lookup the type converter and store that instead of the
    # type itself.
    if read_type is not None:
      converter = _CosimPort.ConvertLookup[type(read_type)]
      self._read_convert = converter(schema, read_type)
    if write_type is not None:
      converter = _CosimPort.ConvertLookup[type(write_type)]
      self._write_convert = converter(schema, write_type)

  def write(self, msg) -> bool:
    """Write a message to this port."""
    self._endpoint.send(self._write_convert.write(msg)).wait()
    return True

  def read(self, blocking_time: typing.Optional[float]):
    """Read a message from this port. If 'blocking_timeout' is None, return
    immediately. Otherwise, wait up to 'blocking_timeout' for a message. Returns
    the message if found, None if no message was read."""

    if blocking_time is None:
      # Non-blocking.
      recvResp = self._endpoint.recv(False).wait()
    else:
      # Blocking. Since our cosim rpc server doesn't currently support blocking
      # reads, use polling instead.
      e = time.time() + blocking_time
      recvResp = None
      while recvResp is None or e > time.time():
        recvResp = self._endpoint.recv(False).wait()
        if recvResp.hasData:
          break
        else:
          time.sleep(0.001)
    if not recvResp.hasData:
      return None
    assert recvResp.resp is not None
    return self._read_convert.read(recvResp.resp)
