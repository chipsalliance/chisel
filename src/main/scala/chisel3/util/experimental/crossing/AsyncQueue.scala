package chisel3.util.experimental.crossing

import chisel3._
import chisel3.util._
import chisel3.util.experimental.group

class AsyncEnqueueIO[T <: Data](gen: T) extends Bundle {
  val clock: Clock = Input(Clock())
  val reset: Bool = Input(Bool())
  val source: DecoupledIO[T] = Flipped(Decoupled(gen))
}

class AsyncDequeueIO[T <: Data](gen: T) extends Bundle {
  val clock: Clock = Input(Clock())
  val reset: Bool = Input(Bool())
  val sink: DecoupledIO[T] = Decoupled(gen)
}

/** Memory used for queue.
  * In ASIC, it will be synthesised to Flip-Flop
  *
  * if [[narrow]]: one write port, one read port.
  *
  * if not [[narrow]]: one write port, multiple read port.
  * ASIC will synthesis it to Flip-Flop
  * not recommend for FPGA
  */
class DataMemory[T <: Data](gen: T, depth: Int, narrow: Boolean) extends RawModule {
  val dataQueue: SyncReadMem[T] = SyncReadMem(depth, gen)

  // write IO
  val writeEnable: Bool = IO(Input(Bool()))
  val writeData: T = IO(Input(gen))
  val writeIndex: UInt = IO(Input(UInt(log2Ceil(depth).W)))
  when(writeEnable)(dataQueue.write(writeIndex, writeData))

  // read IO
  val readEnable: Bool = IO(Input(Bool()))

  // narrow read IO
  val readDataAndIndex: Option[(T, UInt)] = if (narrow) Some((
    IO(Output(gen)).suggestName("data"),
    IO(Input(UInt(log2Ceil(depth).W))).suggestName("index")
  )) else None
  readDataAndIndex.foreach { case (readData, readIndex) => readData := dataQueue.read(readIndex) }

  // broad read IO
  val fullReadData: Option[Vec[T]] = if (narrow) None else Some(IO(Output(Vec(depth, gen))))
  fullReadData match {
    case Some(fullData) => fullData.zipWithIndex.map { case (data, index) => data := dataQueue.read(index.U) }
    case _ =>
  }
}

/** Sink of [[AsyncQueue]] constructor.
  *
  * @tparam T Hardware type to be converted.
  * @note SRAM-based clock-domain-crossing source.
  */
class AsyncQueueSink[T <: Data](gen: T, depth: Int, sync: Int, narrow: Boolean = true) extends MultiIOModule {
  require(depth > 0 && isPow2(depth), "todo")
  require(sync >= 2, "todo")
  private val depthWidth: Int = log2Ceil(depth)
  /** Dequeue Decoupled IO. */
  val dequeue: DecoupledIO[T] = IO(Decoupled(gen))

  val readIndexGray: UInt = withReset(reset.asAsyncReset())(grayCounter(depthWidth + 1, dequeue.fire(), false.B, "readIndex"))
  val readIndexGrayReg: UInt = withReset(reset.asAsyncReset())(RegNext(next = readIndexGray, init = 0.U).suggestName("readIndexReg"))
  val writeIndexGray: UInt = IO(Input(UInt((depthWidth + 1).W)))

  /** ready signal to indicate [[DecoupledIO]] this queue is not empty, can still dequeue new data. */
  val empty: Bool = readIndexGray === writeIndexGray
  val valid: Bool = !empty
  val validReg: Bool = withReset(reset.asAsyncReset())(RegNext(next = valid, init = false.B).suggestName("validReg"))

  // dequeue to [[DecoupledIO]]
  dequeue.valid := validReg

  // port to access memory
  val readEnable: Bool = IO(Output(Bool()))
  readEnable := valid

  val readDataAndIndex: Option[(T, UInt)] = if (narrow) Some((
    IO(Input(gen)).suggestName("data"),
    IO(Output(UInt(log2Ceil(depth).W))).suggestName("index")
  )) else None
  readDataAndIndex.foreach {
    case (data, index) =>
      dequeue.bits := data
      index := readIndexGray(depthWidth, 0)
  }

  // This register does not NEED to be reset, as its contents will not
  // be considered unless the asynchronously reset deq valid, register is set.
  // It is possible that bits latches when the source domain is reset / has power cut
  // This is safe, because isolation gates brought mem low before the zeroed [[writeIndex]] reached us.

  val fullReadData: Option[Vec[T]] = if (narrow) None else Some(IO(Input(Vec(depth, gen))))
  fullReadData.foreach(fullData => dequeue.bits := fullData(readIndexGray(depthWidth, 0)))
}

/** Source of [[AsyncQueue]] constructor.
  *
  * @tparam T Hardware type to be converted.
  * @todo make sync optional, if None use async logic.
  *       add some verification codes.
  */
class AsyncQueueSource[T <: Data](gen: T, depth: Int, sync: Int) extends MultiIOModule {
  require(depth > 0 && isPow2(depth), "todo")
  require(sync >= 2, "todo")
  private val depthWidth: Int = log2Ceil(depth)
  /** Enqueue Decoupled IO. */
  val enqueue: DecoupledIO[T] = IO(Flipped(Decoupled(gen)))

  val writeIndexGray: UInt = withReset(reset.asAsyncReset())(grayCounter(depthWidth + 1, enqueue.fire(), false.B, "writeIndex"))
  val writeIndexGrayReg: UInt = withReset(reset.asAsyncReset())(RegNext(next = writeIndexGray, init = 0.U).suggestName("writeIndexReg"))
  val readIndexGray: UInt = IO(Input(UInt((depthWidth + 1).W)))

  /** ready signal to indicate [[DecoupledIO]] this queue is not full, can still enqueue new data. */
  val full: Bool = writeIndexGray === (readIndexGray ^ (depth | depth >> 1).U)
  val ready: Bool = !full
  val readyReg: Bool = withReset(reset.asAsyncReset())(RegNext(next = ready, init = false.B).suggestName("readyReg"))

  // enqueue from [[DecoupledIO]]
  enqueue.ready := readyReg

  // port to access memory
  val writeEnable: Bool = IO(Output(Bool()))
  writeEnable := enqueue.fire()
  val writeData: T = IO(Output(gen))
  writeData := enqueue.bits
  val writeIndex: UInt = IO(Output(UInt(log2Ceil(depth).W)))
  writeIndex := writeIndexGrayReg(depthWidth, 0)
}

/** cross-clock-domain syncing asynchronous queue.
  *
  * @note [[AsyncQueueSource.writeIndexGray]] and [[AsyncQueueSink.readIndexGray]] will be synced to each other.
  *       both of these use a dual-gray-counter for index and empty/full detecting.
  *       index has `depth + 1` size
  * {{{
  *       ramIndex := index(depth, 0)
  *       full := writeIndex === (readIndex ^ (depth | depth >> 1).U)
  *       empty :=  writeIndex === readIndex
  * }}}
  *
  */
class AsyncQueue[T <: Data](gen: T, depth: Int, sync: Int, narrow: Boolean) extends MultiIOModule {
  val enqueue: AsyncEnqueueIO[T] = IO(new AsyncEnqueueIO(gen))
  val sourceModule: AsyncQueueSource[T] =
    withClockAndReset(enqueue.clock, enqueue.reset)(Module(new AsyncQueueSource(gen, depth, sync)))
  sourceModule.enqueue <> enqueue.source

  val dequeue: AsyncDequeueIO[T] = IO(new AsyncDequeueIO(gen))
  val sinkModule: AsyncQueueSink[T] =
    withClockAndReset(enqueue.clock, enqueue.reset)(Module(new AsyncQueueSink(gen, depth, sync, narrow)))
  dequeue.sink <> sinkModule.dequeue

  // read/write index bidirectional sync
  sourceModule.readIndexGray :=
    withClockAndReset(enqueue.clock, enqueue.reset.asAsyncReset()) {
      val shiftRegisters = ShiftRegisters(sinkModule.readIndexGray, sync)
      group(shiftRegisters, s"AsyncResetSynchronizerShiftReg_w${sinkModule.readIndexGray.width}_d$sync", "syncReadIndexGray")
      shiftRegisters.last
    }
  sinkModule.writeIndexGray :=
    withClockAndReset(dequeue.clock, dequeue.reset.asAsyncReset()) {
      val shiftRegisters = ShiftRegisters(sourceModule.writeIndexGray, sync)
      group(shiftRegisters, s"AsyncResetSynchronizerShiftReg_w${sourceModule.readIndexGray.width}_d$sync", "syncReadIndexGray")
      shiftRegisters.last
    }

  val memoryModule: DataMemory[T] = withClock(dequeue.clock)(Module(new DataMemory(gen, depth, narrow)))
  memoryModule.writeEnable := sourceModule.writeEnable
  memoryModule.writeData := sourceModule.writeData
  memoryModule.writeIndex := sourceModule.writeIndex
  memoryModule.readEnable := sinkModule.readEnable
  (memoryModule.readDataAndIndex zip sinkModule.readDataAndIndex).foreach {
    case ((memoryData, memoryIndex), (sinkData, sinkIndex)) =>
      sinkData := withClock(dequeue.clock)(RegNext(memoryData))
      memoryIndex := sinkIndex
    case _ =>
  }

  (memoryModule.fullReadData zip sinkModule.fullReadData).foreach {
    case (memoryFullData, sinkFullData) =>
      sinkFullData := withClock(dequeue.clock)(RegNext(memoryFullData))
    case _ =>
  }
}
