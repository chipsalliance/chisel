import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                 ["out0", "out1"])

  resCheck = cocotb.start_soon(out0.checkOutputs([100]))

  in0Send = cocotb.start_soon(in0.send(0))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck


@cocotb.test()
async def sendMultiple(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "in1"],
                                                 ["out0", "out1"])

  resCheck = cocotb.start_soon(
      out0.checkOutputs([100, 24, 100, 24, 100, 24, 100, 24]))

  for i in range(4):
    in0Send = cocotb.start_soon(in0.send(0))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

    in0Send = cocotb.start_soon(in0.send(24))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await inCtrlSend

  await resCheck
