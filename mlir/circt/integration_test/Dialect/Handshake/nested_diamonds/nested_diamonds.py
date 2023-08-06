import cocotb
from helper import initDut
import random

random.seed(0)


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                 ["out0", "out1"])
  out0Check = cocotb.start_soon(out0.checkOutputs([0]))

  in0Send = cocotb.start_soon(in0.send(10))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await out0Check


def getResInput(i):
  if (i <= 50):
    return 0
  if (i <= 100):
    return 1
  if (i <= 200):
    return 2
  return 3


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                 ["out0", "out1"])

  N = 20
  inputs = [random.randint(0, 300) for _ in range(N)]
  res = [getResInput(i) for i in inputs]

  out0Check = cocotb.start_soon(out0.checkOutputs(res))

  for i in inputs:
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await out0Check
