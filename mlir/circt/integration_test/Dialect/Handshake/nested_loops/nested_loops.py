import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                 ["out0", "out1"])
  out0Check = cocotb.start_soon(out0.checkOutputs([4]))

  in0Send = cocotb.start_soon(in0.send(2))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await out0Check


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                 ["out0", "out1"])

  N = 10
  # sum_{i = 0}^n (sum_{j=0}^i i) = 1/6 * (n^3 + 3n^2 + 2n)
  res = [(1 / 6) * (n**3 + 3 * n**2 + 2 * n) for n in range(N)]

  out0Check = cocotb.start_soon(out0.checkOutputs(res))

  for i in range(N):
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await out0Check
