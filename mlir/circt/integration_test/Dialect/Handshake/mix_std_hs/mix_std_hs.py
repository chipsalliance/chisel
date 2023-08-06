import cocotb
from helper import initDut, to_struct, from_struct
import random
import ctypes

random.seed(0)


def kernel(e):
  if (e >> 1) & 0x1:
    if (e >> 2) & 0x1:
      return e + 4
    return e * 10
  return e


def getOutput(t):
  return tuple(map(kernel, t))


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  inputs = [(8, 8, 4, 8, 5, 3, 1, 0)]
  outputs = [to_struct(getOutput(i), ctypes.c_uint64) for i in inputs]

  # Run checkOutputs using a converter that decodes the long binary value into
  # the expected struct.
  resCheck = cocotb.start_soon(
      out0.checkOutputs(
          outputs,
          converter=lambda x: from_struct(x, len(inputs[0]), ctypes.c_uint64)))

  in0Send = cocotb.start_soon(in0.send(to_struct(inputs[0], ctypes.c_uint64)))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck


def randomTuple():
  return tuple([random.randint(0, 100) for _ in range(8)])


@cocotb.test()
async def multipleInputs(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  N = 10
  inputs = [randomTuple() for _ in range(N)]

  outputs = [to_struct(getOutput(i), ctypes.c_uint64) for i in inputs]
  resCheck = cocotb.start_soon(
      out0.checkOutputs(
          outputs,
          converter=lambda x: from_struct(x, len(inputs[0]), ctypes.c_uint64)))

  for i in inputs:
    in0Send = cocotb.start_soon(in0.send(to_struct(i, ctypes.c_uint64)))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await resCheck
