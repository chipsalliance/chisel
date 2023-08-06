import cocotb
from cocotb.triggers import Timer
import cocotb.clock


async def clock(dut):
  dut.clk.value = 0
  await Timer(1, units='ns')
  dut.clk.value = 1
  await Timer(1, units='ns')


async def initDut(dut):
  # Reset
  dut.rst.value = 1
  await clock(dut)
  dut.rst.value = 0
  await clock(dut)


async def write(dut, value):
  dut.inp.value = value
  dut.wrEn.value = 1
  await clock(dut)
  dut.wrEn.value = 0
  await clock(dut)


async def combRead(dut, out):
  dut.rdEn.value = 1
  # Combinational reads, so let the model settle before reading
  await Timer(1, units='ns')
  return dut.out.value


async def read(dut):
  data = await combRead(dut, dut.out.value)
  await clock(dut)
  dut.rdEn.value = 0
  await clock(dut)
  return data


async def readWrite(dut, value):
  dut.inp.value = value
  dut.wrEn.value = 1
  data = await combRead(dut, dut.out.value)
  await clock(dut)
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  await clock(dut)
  return data


FIFO_DEPTH = 4
FIFO_ALMOST_FULL = 2
FIFO_ALMOST_EMPTY = 1


async def test_separate_read_write(dut):
  # Run a test where we incrementally write and read values from 1 to FIFO_DEPTH values
  for i in range(1, FIFO_DEPTH):
    for j in range(i):
      await write(dut, 42 + j)

    if i >= FIFO_ALMOST_FULL:
      assert dut.almost_full.value == 1

    if i <= FIFO_ALMOST_EMPTY:
      assert dut.almost_empty.value == 1

    if i == FIFO_DEPTH:
      assert dut.full.value == 1

    for j in range(i):
      assert await read(dut) == 42 + j

    assert dut.empty.value == 1


async def test_concurrent_read_write(dut):
  # Fill up the FIFO halfway and concurrently read and write. Should be able
  # to do this continuously.
  counter = 0
  for i in range(FIFO_DEPTH // 2):
    await write(dut, counter)
    counter += 1

  for i in range(FIFO_DEPTH * 2):
    expected_value = counter - FIFO_DEPTH // 2
    print("expected_value: ", expected_value)
    assert await readWrite(dut, counter) == expected_value
    assert dut.full.value == 0
    assert dut.empty.value == 0
    counter += 1


@cocotb.test()
async def test1(dut):
  dut.inp.value = 0
  dut.rdEn.value = 0
  dut.wrEn.value = 0
  await initDut(dut)

  await test_separate_read_write(dut)
  await test_concurrent_read_write(dut)
