import cocotb
from helper import initDut


@cocotb.test()
async def oneInput(dut):
  [in0, in1, inCtrl], [out0,
                       outCtrl] = await initDut(dut, ["in0", "in1", "in2"],
                                                ["out0", "out1"])

  resCheck = cocotb.start_soon(out0.checkOutputs([42]))

  in0Send = cocotb.start_soon(in0.send(42))
  in1Send = cocotb.start_soon(in1.send(1))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await in1Send
  await inCtrlSend

  await resCheck


@cocotb.test()
async def multipleInputs(dut):
  [in0, in1, inCtrl], [out0,
                       outCtrl] = await initDut(dut, ["in0", "in1", "in2"],
                                                ["out0", "out1"])

  resCheck = cocotb.start_soon(out0.checkOutputs([42, 42, 10, 10, 10]))

  inputs = [(42, 1), (0, 0), (10, 1), (42, 0), (99, 0)]
  for (data, w) in inputs:
    in0Send = cocotb.start_soon(in0.send(data))
    in1Send = cocotb.start_soon(in1.send(w))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await in1Send
    await inCtrlSend

  await resCheck
