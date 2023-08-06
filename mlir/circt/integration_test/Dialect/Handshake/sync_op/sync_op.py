import cocotb
from helper import initDut


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])
  inputs = [1, 2, 3, 4]
  resCheck = cocotb.start_soon(out0.checkOutputs(inputs))
  ctrlCheck = cocotb.start_soon(outCtrl.awaitNOutputs(len(inputs)))

  for i in inputs:
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await resCheck
  await ctrlCheck
