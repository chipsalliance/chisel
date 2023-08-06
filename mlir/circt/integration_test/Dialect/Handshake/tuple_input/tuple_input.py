import cocotb
from helper import initDut, to_struct
import ctypes


@cocotb.test()
async def oneInput(dut):
  [in0, inCtrl], [out0, outCtrl] = await initDut(dut, ["in0", "inCtrl"],
                                                 ["out0", "outCtrl"])

  resCheck = cocotb.start_soon(out0.checkOutputs([42]))
  in0Send = cocotb.start_soon(in0.send(to_struct((24, 18), ctypes.c_uint32)))
  inCtrlSend = cocotb.start_soon(inCtrl.send())

  await in0Send
  await inCtrlSend

  await resCheck
