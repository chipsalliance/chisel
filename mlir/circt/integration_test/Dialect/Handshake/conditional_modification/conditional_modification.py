import cocotb
from helper import initDut

inNames = ["in0", "in1", "in2", "in3"]
outNames = ["out0", "out1", "out2"]


@cocotb.test()
async def oneInput(dut):
  [in0, in1, in2, inCtrl], [out0, out1,
                            outCtrl] = await initDut(dut, inNames, outNames)

  out0Check = cocotb.start_soon(out0.checkOutputs([18]))
  out1Check = cocotb.start_soon(out1.checkOutputs([24]))

  in0Send = cocotb.start_soon(in0.send(18))
  in1Send = cocotb.start_soon(in1.send(24))
  in2Send = cocotb.start_soon(in2.send(0))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await in1Send
  await in2Send
  await inCtrlSend

  await out0Check
  await out1Check


@cocotb.test()
async def multiple(dut):
  [in0, in1, in2, inCtrl], [out0, out1,
                            outCtrl] = await initDut(dut, inNames, outNames)

  out0Check = cocotb.start_soon(out0.checkOutputs([18, 42, 42]))
  # COCOTB treats all integers as unsigned, thus we have to compare with the
  # two's complement representation
  out1Check = cocotb.start_soon(out1.checkOutputs([24, 2**32 - 6, 42]))

  inputs = [(18, 24, 0), (18, 24, 1), (42, 0, 1)]
  for (d0, d1, cond) in inputs:
    in0Send = cocotb.start_soon(in0.send(d0))
    in1Send = cocotb.start_soon(in1.send(d1))
    in2Send = cocotb.start_soon(in2.send(cond))
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    await in0Send
    await in1Send
    await in2Send
    await inCtrlSend

  await out0Check
  await out1Check
