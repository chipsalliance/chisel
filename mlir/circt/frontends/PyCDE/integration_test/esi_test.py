# REQUIRES: esi-cosim, rtl-sim
# RUN: rm -rf %t
# RUN: %PYTHON% %s %t 2>&1
# RUN: esi-cosim-runner.py --no-aux-files --tmpdir %t --schema %t/runtime/schema.capnp %s `ls %t/hw/*.sv | grep -v driver.sv`
# PY: from esi_test import run_cosim
# PY: run_cosim(tmpdir, rpcschemapath, simhostport)

import pycde
from pycde import (Clock, Input, InputChannel, OutputChannel, Module, generator,
                   types)
from pycde import esi
from pycde.bsp import cosim
from pycde.constructs import Wire

import sys


@esi.ServiceDecl
class HostComms:
  to_host = esi.ToServer(types.any)
  from_host = esi.FromServer(types.any)
  req_resp = esi.ToFromServer(to_server_type=types.i16,
                              to_client_type=types.i32)


class Producer(Module):
  clk = Input(types.i1)
  int_out = OutputChannel(types.i32)

  @generator
  def construct(ports):
    chan = HostComms.from_host("loopback_in", types.i32)
    ports.int_out = chan


class Consumer(Module):
  clk = Input(types.i1)
  int_in = InputChannel(types.i32)

  @generator
  def construct(ports):
    HostComms.to_host(ports.int_in, "loopback_out")


class LoopbackInOutAdd7(Module):

  @generator
  def construct(ports):
    loopback = Wire(types.channel(types.i16))
    from_host = HostComms.req_resp(loopback, "loopback_inout")
    ready = Wire(types.i1)
    data, valid = from_host.unwrap(ready)
    plus7 = data.as_uint(15) + types.ui8(7)
    data_chan, data_ready = loopback.type.wrap(plus7.as_bits(), valid)
    ready.assign(data_ready)
    loopback.assign(data_chan)


class Mid(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)

    LoopbackInOutAdd7()


class Top(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    Mid(clk=ports.clk, rst=ports.rst)


if __name__ == "__main__":
  s = pycde.System(cosim.CosimBSP(Top),
                   name="ESILoopback",
                   output_directory=sys.argv[1],
                   sw_api_langs=["python"])
  s.compile()
  s.package()


def run_cosim(tmpdir, schema_path, rpchostport):
  import os
  import time
  sys.path.append(os.path.join(tmpdir, "runtime"))
  import ESILoopback as esi_sys
  from ESILoopback.common import Cosim

  top = esi_sys.top(Cosim(schema_path, rpchostport))

  assert top.bsp.req_resp_read_any() is None
  assert top.bsp.req_resp[0].read(blocking_timeout=None) is None
  assert top.bsp.to_host_read_any() is None
  assert top.bsp.to_host[0].read(blocking_timeout=None) is None

  assert top.bsp.req_resp[0].write(5) is True
  time.sleep(0.05)
  assert top.bsp.to_host_read_any() is None
  assert top.bsp.to_host[0].read(blocking_timeout=None) is None
  assert top.bsp.req_resp[0].read() == 12
  assert top.bsp.req_resp[0].read(blocking_timeout=None) is None

  assert top.bsp.req_resp[0].write(9) is True
  time.sleep(0.05)
  assert top.bsp.to_host_read_any() is None
  assert top.bsp.to_host[0].read(blocking_timeout=None) is None
  assert top.bsp.req_resp_read_any() == 16
  assert top.bsp.req_resp_read_any() is None
  assert top.bsp.req_resp[0].read(blocking_timeout=None) is None

  assert top.bsp.from_host[0].write(9) is True
  time.sleep(0.05)
  assert top.bsp.req_resp_read_any() is None
  assert top.bsp.req_resp[0].read(blocking_timeout=None) is None
  assert top.bsp.to_host_read_any() == 9
  assert top.bsp.to_host[0].read(blocking_timeout=None) is None

  assert top.bsp.from_host[0].write(9) is True
  time.sleep(0.05)
  assert top.bsp.req_resp_read_any() is None
  assert top.bsp.req_resp[0].read(blocking_timeout=None) is None
  assert top.bsp.to_host[0].read() == 9
  assert top.bsp.to_host_read_any() is None

  print("Success: all tests pass!")
