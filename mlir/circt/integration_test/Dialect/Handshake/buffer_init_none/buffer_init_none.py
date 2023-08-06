import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [inCtrl], [outCtrl] = await initDut(dut, ["inCtrl"], ["outCtrl"])

  outCtrlCheck = cocotb.start_soon(outCtrl.awaitNOutputs(1))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await inCtrlSend
  await outCtrlCheck
