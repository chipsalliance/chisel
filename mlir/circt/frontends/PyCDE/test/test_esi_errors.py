# RUN: rm -rf %t
# RUN: %PYTHON% py-split-input-file.py %s 2>&1 | FileCheck %s

from pycde import (Clock, Input, InputChannel, OutputChannel, Module, generator,
                   types)
from pycde import esi
from pycde.testing import unittestmodule


@esi.ServiceDecl
class HostComms:
  to_host = esi.ToServer(types.any)
  from_host = esi.FromServer(types.any)


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


@unittestmodule(print=True)
class LoopbackTop(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    p = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)
    # Use Cosim to implement the standard 'HostComms' service.
    esi.Cosim(HostComms, ports.clk, ports.rst)


class MultiplexerService(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(types.i1)

  def __init__(self, **inputs):
    super().__init__(HostComms, **inputs)

  @generator
  def generate(ports, channels):

    c = types.i128(0)
    v = types.i1(0)
    chan, rdy = types.channel(types.i128).wrap(c, v)
    try:
      # CHECK: Channel type mismatch. Expected Channel<Bits<32>, ValidReady>, got Channel<Bits<128>, ValidReady>.
      channels.to_client_reqs[0].assign(chan)
    except Exception as e:
      print(e)
    try:
      input_req = channels.to_server_reqs[0]
      channels.to_client_reqs[1].assign(input_req)
      # CHECK: Producer_1.loopback_in has already been connected.
      channels.to_client_reqs[1].assign(input_req)
    except Exception as e:
      print(e)
    # CHECK: ValueError: Producer.loopback_in has not been connected.


@unittestmodule(run_passes=True, print_after_passes=True)
class MultiplexerTop(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    MultiplexerService(clk=ports.clk, rst=ports.rst)

    p = Producer(clk=ports.clk)
    p1 = Producer(clk=ports.clk)
    Consumer(clk=ports.clk, int_in=p.int_out)


# -----


class BrokenService(esi.ServiceImplementation):
  clk = Clock()
  rst = Input(types.i1)

  def __init__(self, **inputs):
    super().__init__(HostComms, **inputs)

  @generator
  def generate(ports, channels):
    return "asdf"
    # CHECK: ValueError: Generators must a return a bool or None


@unittestmodule(run_passes=True, print_after_passes=True)
class BrokenTop(Module):
  clk = Clock(types.i1)
  rst = Input(types.i1)

  @generator
  def construct(ports):
    BrokenService(clk=ports.clk, rst=ports.rst)
