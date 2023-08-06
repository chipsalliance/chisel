import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [inCtrl], [out0, outCtrl] = await initDut(dut, ["inCtrl"],
                                            ["out0", "outCtrl"])

  out0Check = cocotb.start_soon(out0.checkOutputs([579]))
  inCtrlSend = cocotb.start_soon(inCtrl.send())
  await inCtrlSend

  await out0Check
