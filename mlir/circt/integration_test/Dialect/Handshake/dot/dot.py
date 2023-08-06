import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [inCtrl], [out0, outCtrl] = await initDut(dut, ["in0"], ["out0", "out1"])

  out0Check = cocotb.start_soon(out0.checkOutputs([3589632]))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await inCtrlSend
  await out0Check
