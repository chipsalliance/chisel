import cocotb
from cocotb.triggers import Timer
import cocotb.clock


async def clock(dut):
  dut.clock.value = 0
  await Timer(1, units='ns')
  dut.clock.value = 1
  await Timer(1, units='ns')


async def initDut(dut):
  """
  Initializes a dut by adding a clock, setting initial valid and ready flags,
  and performing a reset.
  """
  # Reset
  dut.reset.value = 1
  await clock(dut)
  dut.reset.value = 0
  await clock(dut)


@cocotb.test()
async def test1(dut):
  dut.go.value = 0
  await initDut(dut)

  dut.arg0.value = 42
  dut.arg1.value = 24
  dut.go.value = 1

  while dut.done != 1:
    await clock(dut)
    dut.go.value = 0
    dut.arg0.value = 0
    dut.arg1.value = 0

  assert dut.out == 174, f"Expected 174, got {dut.out}"
