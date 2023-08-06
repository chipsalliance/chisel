# REQUIRES: esi-cosim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# ... can't glob *.sv because PyCDE always includes driver.sv, but that's not the
# top that we want to use. Just delete it.
# RUN: rm -f %t/hw/driver.sv
# RUN: esi-cosim-runner.py --no-aux-files --tmpdir %t --schema %t/hw/schema.capnp %s %t/hw/*.sv
# PY: from esi_ram import run_cosim
# PY: run_cosim(tmpdir, rpcschemapath, simhostport)

import pycde
from pycde import (Clock, Input, Module, generator, types)
from pycde.constructs import Wire
from pycde import esi
from pycde.bsp import cosim

import sys

RamI64x8 = esi.DeclareRandomAccessMemory(types.i64, 8)
WriteType = RamI64x8.write.to_server_type


@esi.ServiceDecl
class MemComms:
  write = esi.FromServer(WriteType)
  read = esi.ToFromServer(to_server_type=types.i64, to_client_type=types.i3)
  loopback = esi.ToFromServer(to_server_type=WriteType,
                              to_client_type=WriteType)


class Mid(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    (address_chan, ready) = RamI64x8.read.to_server_type.wrap(2, True)
    (read_data, read_valid) = RamI64x8.read(address_chan).unwrap(True)
    (write_data, _) = WriteType.wrap({
        'data': read_data,
        'address': 3
    }, read_valid)
    RamI64x8.write(write_data)


class Top(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    Mid(clk=ports.clk, rst=ports.rst)
    RamI64x8.write(MemComms.write("write"))
    read_address = Wire(RamI64x8.read.to_server_type)
    read_data = RamI64x8.read(read_address)
    read_address.assign(MemComms.read(read_data, "read"))

    loopback_wire = Wire(WriteType)
    loopback_wire.assign(MemComms.loopback(loopback_wire, "loopback"))

    RamI64x8.instantiate_builtin("sv_mem",
                                 result_types=[],
                                 inputs=[ports.clk, ports.rst])


def run_cosim(tmpdir=".", schema_path="schema.capnp", rpchostport=None):
  import os
  sys.path.append(os.path.join(tmpdir, "runtime"))
  import ESIMem as esi_sys
  from ESIMem.common import Cosim
  if rpchostport is None:
    port = open("cosim.cfg").read().split(':')[1].strip()
    rpchostport = f"localhost:{port}"

  cosim = Cosim(schema_path, rpchostport)
  print(cosim.list())
  top = esi_sys.top(cosim).bsp
  print(dir(top))

  write_cmd = {"address": 2, "data": 42}
  loopback_result = top.loopback[0](write_cmd)
  assert loopback_result == write_cmd

  read_result = top.read[0](2)
  assert read_result == 0
  read_result = top.read[0](3)
  assert read_result == 0

  top.write[0].write(write_cmd)
  read_result = top.read[0](2)
  assert read_result == 42
  read_result = top.read[0](3)
  assert read_result == 42


if __name__ == "__main__":
  s = pycde.System([cosim.CosimBSP(Top)],
                   name="ESIMem",
                   output_directory=sys.argv[1])
  s.compile()
  s.package()
