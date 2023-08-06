import cocotb
from helper import initDut
import math


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, out1, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                       ["out0", "out1", "out2"])
  out0Check = cocotb.start_soon(out0.checkOutputs([15]))
  out1Check = cocotb.start_soon(out1.checkOutputs([120]))

  in0Send = cocotb.start_soon(in0.send(5))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await out0Check
  await out1Check


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, out1, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                       ["out0", "out1", "out2"])

  N = 10
  res0 = [i * (i + 1) / 2 for i in range(N)]
  res1 = [math.factorial(i) for i in range(N)]

  out0Check = cocotb.start_soon(out0.checkOutputs(res0))
  out1Check = cocotb.start_soon(out1.checkOutputs(res1))

  for i in range(N):
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await out0Check
  await out1Check
