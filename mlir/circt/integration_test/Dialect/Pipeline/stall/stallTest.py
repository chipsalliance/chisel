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


def ref(v1, v2):
  x1 = v1 + v2
  x2 = x1 + v1
  return x1 + x2


@cocotb.test()
async def test1(dut):
  dut.go.value = 0
  dut.stall.value = 0
  await initDut(dut)

  v1 = 42
  v2 = 24
  resref = ref(v1, v2)

  # Get values into the first stage
  dut.arg0.value = v1
  dut.arg1.value = v2
  dut.go.value = 1
  await clock(dut)

  # Stall for 2 cycles. Done should not be asserted nor the output correct.
  dut.go.value = 0
  dut.arg0.value = 0
  dut.arg1.value = 0
  dut.stall.value = 1
  for i in range(2):
    await clock(dut)
    assert dut.done != 1, "DUT should not be done when stalling"

  # Unstall and wait for 1 clock cycles - this should propagate the values through
  # the remaining stage.
  dut.stall.value = 0
  await clock(dut)
  assert dut.done == 1, "DUT should be done after unstalling"
  assert dut.out == resref, f"Expected {resref}, got {dut.out}"

  # Clock once more, done should be deasserted.
  await clock(dut)
  assert dut.done == 0, "DUT should not be done after pipeline has finished"
