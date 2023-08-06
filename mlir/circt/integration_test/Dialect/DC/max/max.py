import cocotb
from dc_cocotb import initDut
import random

inNames = [f"in{i}" for i in range(8)] + ["in8"]
random.seed(0)


def genInOutPair():
  inputs = [random.randint(0, 1000) for _ in range(8)]
  output = max(inputs)
  return inputs, output


@cocotb.test()
async def oneInput(dut):
  ins, [out0, outCtrl] = await initDut(dut, inNames, ["out0", "out1"])
  inCtrl = ins[-1]
  inPorts = ins[:-1]

  inputs, output = genInOutPair()
  resCheck = cocotb.start_soon(out0.checkOutputs([output]))

  sends = [
      cocotb.start_soon(inPort.send(data))
      for [inPort, data] in zip(inPorts, inputs)
  ]
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  for s in sends:
    await s
  await inCtrlSend

  await resCheck


@cocotb.test()
async def multipleInputs(dut):
  ins, [out0, outCtrl] = await initDut(dut, inNames, ["out0", "out1"])
  inCtrl = ins[-1]
  inPorts = ins[:-1]

  inOutPairs = [genInOutPair() for _ in range(8)]
  resCheck = cocotb.start_soon(
      out0.checkOutputs([out for (_, out) in inOutPairs]))

  for (inputs, _) in inOutPairs:
    sends = [
        cocotb.start_soon(inPort.send(data))
        for [inPort, data] in zip(inPorts, inputs)
    ]
    inCtrlSend = cocotb.start_soon(inCtrl.send())

    for s in sends:
      await s
    await inCtrlSend

  await resCheck
