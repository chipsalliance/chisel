import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  resCheck = cocotb.start_soon(out0.checkOutputs([24]))
  ctrlCheck = cocotb.start_soon(outCtrl.awaitNOutputs(1))

  in0Send = cocotb.start_soon(in0.send(24))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck
  await ctrlCheck


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  inputs = [24, 0, 42, 0]
  # This is an identity circuit susceptible to reorderings
  resCheck = cocotb.start_soon(out0.checkOutputs(inputs))
  ctrlCheck = cocotb.start_soon(outCtrl.awaitNOutputs(len(inputs)))

  for i in inputs:
    in0Send = cocotb.start_soon(in0.send(i))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await resCheck
  await ctrlCheck
