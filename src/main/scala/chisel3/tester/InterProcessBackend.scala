// See LICENSE for license details.

package chisel3.tester

import java.util.concurrent.ConcurrentLinkedQueue
import scala.collection.mutable

import chisel3._
import chisel3.tester.TesterUtils.{getIOPorts, getPortNames}

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.immutable.ListMap
import scala.concurrent.{Await, ExecutionContext, Future, blocking}
import scala.concurrent.duration._
import scala.sys.process.{Process, ProcessLogger}
import java.io.{File, RandomAccessFile}
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

object InterProcessBackend {

  class InterProcessBackendException(message: String) extends RuntimeException(message)
  object InterProcessBackendException {
    def apply(message: String): InterProcessBackendException = new InterProcessBackendException(message: String)
  }

  trait ChannelProtocol {
    def aquire: Unit
    def release: Unit
    def ready: Boolean
    def valid: Boolean
    def produce: Unit
    def consume: Unit
  }
  object VerilatorChannelProtocol {
    val channel_data_offset_64bw = 4    // Offset from start of channel buffer to actual user data in 64bit words.
  }
  class VerilatorChannelProtocol(val buffer: java.nio.ByteBuffer) extends ChannelProtocol {
    import VerilatorChannelProtocol._
    implicit def intToByte(i: Int) = i.toByte
    buffer order java.nio.ByteOrder.nativeOrder
    buffer.clear()
    def aquire {
      buffer put (0, 1)
      buffer put (2, 0)
      while((buffer get 1) == 1 && (buffer get 2) == 0) {}
    }
    def release { buffer put (0, 0) }
    def ready = (buffer get 3) == 0
    def valid = (buffer get 3) == 1
    def produce { buffer put (3, 1) }
    def consume { buffer put (3, 0) }
    def update(idx: Int, data: Long) { buffer putLong (8 * idx + channel_data_offset_64bw, data) }
    def update(base: Int, data: String) {
      data.zipWithIndex foreach {case (c, i) => buffer put (base + i + channel_data_offset_64bw, c) }
      buffer put (base + data.size + channel_data_offset_64bw, 0)
    }
    def apply(idx: Int): Long = buffer getLong (8 * idx + channel_data_offset_64bw)
  }
}

class InterProcessBackend[T <: Module](
  dut: T,
  cmd: Seq[String],
  rnd: scala.util.Random)
  extends BackendInstance[T] with ThreadedBackend {
  import InterProcessBackend._
  val portNameDelimiter: String = "."
  val ioPortNameDelimiter: String = "."
  implicit val logger = new TestErrorLog

  def getModule: T = dut

  private val HARD_POKE = 1
//  def getPortNames(dut: Module) = (getDataNames("io", dut.io) ++ getDataNames("reset", dut.reset)).map{case (d: Data, s: String) => (d, (dut.name + "." + s))}.toMap
  // We need to add the dut name to the signal name for the backend.
  protected val portNames = getPortNames(dut) map { case (k: Data, v: String) => (k, v)}
  val (inputs, outputs) = getIOPorts(dut)
  protected def resolveName(signal: Data): String = {
    portNames.getOrElse(signal, dut.name + portNameDelimiter + signal.toString())
  }
  private[tester] case class TestApplicationException(exitVal: Int, lastMessage: String)
    extends RuntimeException(lastMessage)

  private[tester] object TesterProcess {
    def apply(cmd: Seq[String], logs: ArrayBuffer[String]): Process = {
      require(new java.io.File(cmd.head).exists, s"${cmd.head} doesn't exist")
      val processBuilder = Process(cmd mkString " ")
      val processLogger = ProcessLogger(println, logs += _) // don't log stdout
      processBuilder run processLogger
    }
    def kill() {
      while(!exitValue.isCompleted) process.destroy
      println("Exit Code: %d".format(process.exitValue))
    }
  }

  def getMappedBuffer(name: String): (MappedByteBuffer, RandomAccessFile) = {
    val file = new java.io.RandomAccessFile(name, "rw")
    val channel = file.getChannel
    /* @volatile */ val buffer = {
      /* We have seen runs where buffer.put(0,0) fails with:
  [info]   java.lang.IndexOutOfBoundsException:
  [info]   at java.nio.Buffer.checkIndex(Buffer.java:532)
  [info]   at java.nio.DirectByteBuffer.put(DirectByteBuffer.java:300)
  [info]   at Chisel.Tester$Channel.release(Tester.scala:148)
  [info]   at Chisel.Tester.start(Tester.scala:717)
  [info]   at Chisel.Tester.<init>(Tester.scala:743)
  [info]   at ArbiterSuite$ArbiterTests$8.<init>(ArbiterTest.scala:396)
  [info]   at ArbiterSuite$$anonfun$testStableRRArbiter$1.apply(ArbiterTest.scala:440)
  [info]   at ArbiterSuite$$anonfun$testStableRRArbiter$1.apply(ArbiterTest.scala:440)
  [info]   at Chisel.Driver$.apply(Driver.scala:65)
  [info]   at Chisel.chiselMain$.apply(hcl.scala:63)
  [info]   ...
       */
      val size = channel.size
      assert(size > 16, "channel.size is bogus: %d".format(size))
      channel map (FileChannel.MapMode.READ_WRITE, 0, size)
    }
    new File(name).delete
    (buffer, file)
  }

  private class Channel(t: Tuple2[MappedByteBuffer, RandomAccessFile]) extends VerilatorChannelProtocol(t._1) {
    def close {t._2.close}
  }

  val (inputsNameToChunkSizeMap, outputsNameToChunkSizeMap) = {
    def genChunk(args: (Data, String)) = args match {
      case (pin, name) => name -> ((pin.getWidth-1)/64 + 1)
    }
    (ListMap((inputs map genChunk): _*), ListMap((outputs map genChunk): _*))
  }
  private object SIM_CMD extends Enumeration {
    val RESET, STEP, UPDATE, POKE, PEEK, FORCE, GETID, GETCHK, FIN = Value }
  implicit def cmdToId(cmd: SIM_CMD.Value) = cmd.id
  implicit def int(x: Int):  BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  implicit def int(x: Long): BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  private var isStale = false
  private val _pokeMap = HashMap[String, BigInt]()
  private val _peekMap = HashMap[String, BigInt]()
  private val _signalMap = HashMap[String, Int]()
  private val _chunks = HashMap[String, Int]()
  private val _logs = ArrayBuffer[String]()

  //initialize simulator process
  private val process = TesterProcess(cmd, _logs)
  // Set up a Future to wait for (and signal) the test process exit.
  import ExecutionContext.Implicits.global
  private val exitValue = Future(blocking(process.exitValue))
  // memory mapped channels
  private val (inChannel, outChannel, cmdChannel) = {
    // Wait for the startup message
    // NOTE: There may be several messages before we see our startup message.
    val simStartupMessageStart = "sim start on "
    while (!_logs.exists(_ startsWith simStartupMessageStart) && !exitValue.isCompleted) { Thread.sleep(100) }
    // Remove the startup message (and any precursors).
    while (!_logs.isEmpty && !_logs.head.startsWith(simStartupMessageStart)) {
      println(_logs.remove(0))
    }
    println(if (!_logs.isEmpty) _logs.remove(0) else "<no startup message>")
    while (_logs.size < 3) {
      // If the test application died, throw a run-time error.
      throwExceptionIfDead(exitValue)
      Thread.sleep(100)
    }
    val in_channel_name = _logs.remove(0)
    val out_channel_name = _logs.remove(0)
    val cmd_channel_name = _logs.remove(0)
    val in_channel = new Channel(getMappedBuffer(in_channel_name))
    val out_channel = new Channel(getMappedBuffer(out_channel_name))
    val cmd_channel = new Channel(getMappedBuffer(cmd_channel_name))

    println(s"inChannelName: ${in_channel_name}")
    println(s"outChannelName: ${out_channel_name}")
    println(s"cmdChannelName: ${cmd_channel_name}")

    in_channel.consume
    cmd_channel.consume
    in_channel.release
    out_channel.release
    cmd_channel.release

    (in_channel, out_channel, cmd_channel)
  }

  private def dumpLogs() {
    _logs foreach logger.info
    _logs.clear
  }

  private def throwExceptionIfDead(exitValue: Future[Int]) {
    if (exitValue.isCompleted) {
      val exitCode = Await.result(exitValue, Duration(-1, SECONDS))
      // We assume the error string is the last log entry.
      val errorString = if (_logs.size > 0) {
        _logs.last
      } else {
        "test application exit"
      } + " - exit code %d".format(exitCode)
      dumpLogs()
      throw new TestApplicationException(exitCode, errorString)
    }
  }
  // A busy-wait loop that monitors exitValue so we don't loop forever if the test application exits for some reason.
  private def mwhile(block: => Boolean)(loop: => Unit) {
    while (!exitValue.isCompleted && block) {
      loop
    }
    // If the test application died, throw a run-time error.
    throwExceptionIfDead(exitValue)
  }

  private def sendCmd(data: Int) = {
    cmdChannel.aquire
    val ready = cmdChannel.ready
    if (ready) {
      cmdChannel(0) = data
      cmdChannel.produce
    }
    cmdChannel.release
    ready
  }

  private def sendCmd(data: String) = {
    cmdChannel.aquire
    val ready = cmdChannel.ready
    if (ready) {
      cmdChannel(0) = data
      cmdChannel.produce
    }
    cmdChannel.release
    ready
  }

  private def recvResp = {
    outChannel.aquire
    val valid = outChannel.valid
    val resp = if (!valid) None else {
      outChannel.consume
      Some(outChannel(0).toInt)
    }
    outChannel.release
    resp
  }

  private def sendValue(value: BigInt, chunk: Int) = {
    inChannel.aquire
    val ready = inChannel.ready
    if (ready) {
      (0 until chunk) foreach (i => inChannel(i) = (value >> (64*i)).toLong)
      inChannel.produce
    }
    inChannel.release
    ready
  }

  private def recvValue(chunk: Int) = {
    outChannel.aquire
    val valid = outChannel.valid
    val value = if (!valid) None else {
      outChannel.consume
      Some(((0 until chunk) foldLeft BigInt(0))(
        (res, i) => res | (int(outChannel(i)) << (64*i))))
    }
    outChannel.release
    value
  }

  private def recvOutputs = {
    _peekMap.clear
    outChannel.aquire
    val valid = outChannel.valid
    if (valid) {
      (outputsNameToChunkSizeMap.toList foldLeft 0){case (off, (out, chunk)) =>
        _peekMap(out) = ((0 until chunk) foldLeft BigInt(0))(
          (res, i) => res | (int(outChannel(off + i)) << (64 * i))
        )
        off + chunk
      }
      outChannel.consume
    }
    outChannel.release
    valid
  }

  private def sendInputs = {
    inChannel.aquire
    val ready = inChannel.ready
    if (ready) {
      (inputsNameToChunkSizeMap.toList foldLeft 0){case (off, (in, chunk)) =>
        val value = _pokeMap getOrElse (in, BigInt(0))
        (0 until chunk) foreach (i => inChannel(off + i) = (value >> (64 * i)).toLong)
        off + chunk
      }
      inChannel.produce
    }
    inChannel.release
    ready
  }

  private def update() {
    mwhile(!sendCmd(SIM_CMD.UPDATE)) { }
    mwhile(!sendInputs) { }
    mwhile(!recvOutputs) { }
    isStale = false
  }

  private def takeStep() {
    mwhile(!sendCmd(SIM_CMD.STEP)) { }
    mwhile(!sendInputs) { }
    mwhile(!recvOutputs) { }
    dumpLogs()
  }

  private def getId(path: String) = {
    mwhile(!sendCmd(SIM_CMD.GETID)) { }
    mwhile(!sendCmd(dut.name + portNameDelimiter + path)) { }
    if (exitValue.isCompleted) {
      0
    } else {
      (for {
        _ <- Stream.from(1)
        data = recvResp
        if data != None
      } yield data.get).head
    }
  }

  private def getChunk(id: Int) = {
    mwhile(!sendCmd(SIM_CMD.GETCHK)) { }
    mwhile(!sendCmd(id)) { }
    if (exitValue.isCompleted){
      0
    } else {
      (for {
        _ <- Stream.from(1)
        data = recvResp
        if data != None
      } yield data.get).head
    }
  }

  private def poke(id: Int, chunk: Int, v: BigInt, force: Boolean = false) {
    val cmd = if (!force) SIM_CMD.POKE else SIM_CMD.FORCE
    mwhile(!sendCmd(cmd)) { }
    mwhile(!sendCmd(id)) { }
    mwhile(!sendValue(v, chunk)) { }
  }

  private def peek(id: Int, chunk: Int): BigInt = {
    mwhile(!sendCmd(SIM_CMD.PEEK)) { }
    mwhile(!sendCmd(id)) { }
    if (exitValue.isCompleted) {
      BigInt(0)
    } else {
      (for {
        _ <- Stream.from(1)
        data = recvValue(chunk)
        if data != None
      } yield data.get).head
    }
  }

  private def start() {
    println(s"""STARTING ${cmd mkString " "}""")
    mwhile(!recvOutputs) { }
    // reset(5)
    for (i <- 0 until 5) {
      mwhile(!sendCmd(SIM_CMD.RESET)) { }
      mwhile(!recvOutputs) { }
    }
  }

  def pokeSignal(signalName: String, value: BigInt) {
    if (inputsNameToChunkSizeMap contains signalName) {
      _pokeMap(signalName) = value
      isStale = true
    } else {
      val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
      if (id >= 0) {
        poke(id, _chunks getOrElseUpdate (signalName, getChunk(id)), value)
        isStale = true
      } else {
        throw InterProcessBackendException(s"Can't find $signalName in the emulator...")
      }
    }
  }

  override def pokeBits(signal: Bits, value: BigInt, priority: Int): Unit = {
    if (threadingChecker.doPoke(currentThread.get, signal, value, priority, new Throwable)) {
//      println(s"${portNames(signal)} <- $value")
      pokeSignal(portNames(signal), value)
    }
  }

  def peekSignal(signalName: String): BigInt = {
    if (isStale) {
      update
    }
    val result: BigInt = {
      if (outputsNameToChunkSizeMap contains signalName) {
        _peekMap(signalName)
      } else if (inputsNameToChunkSizeMap contains signalName) {
        _pokeMap(signalName)
      } else {
        val id = _signalMap getOrElseUpdate(signalName, getId(signalName))
        if (id >= 0) {
          peek(id, _chunks getOrElse(signalName, getChunk(id)))
        } else {
          throw InterProcessBackendException(s"Can't find $signalName in the emulator...")
        }
      }
    }
    result
  }

  override def peekBits(signal: Bits, stale: Boolean): BigInt = {
    require(!stale, "Stale peek not yet implemented")

    // TODO: properly determine clock
    threadingChecker.doPeek(currentThread.get, signal, dut.clock, new Throwable)
    val a = peekSignal(portNames(signal))
//    println(s"${portNames(signal)} -> $a")
    a
  }

  def step(n: Int) {
    update
    (0 until n) foreach (_ => takeStep)
  }

  def reset(n: Int = 1) {
    for (i <- 0 until n) {
      mwhile(!sendCmd(SIM_CMD.RESET)) { }
      mwhile(!recvOutputs) { }
    }
  }

  def finish() {
    mwhile(!sendCmd(SIM_CMD.FIN)) { }
    println("Exit Code: %d".format(
      Await.result(exitValue, Duration.Inf)))
    dumpLogs
    inChannel.close
    outChannel.close
    cmdChannel.close
  }


  override def expectBits(signal: Bits, value: BigInt, stale: Boolean): Unit = {
    require(!stale, "Stale peek not yet implemented")

    println(s"${portNames(signal)} ?> $value")
    Context().env.testerExpect(value, peekBits(signal, stale), resolveName(signal), None)
  }

  protected val clockCounter : mutable.HashMap[Clock, Int] = mutable.HashMap()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }
  protected def getClock(clk: Clock): Boolean = peekSignal(portNames(clk)).toInt match {
    case 0 => false
    case 1 => true
  }

  protected val lastClockValue: mutable.HashMap[Clock, Boolean] = mutable.HashMap()
  protected val threadingChecker = new ThreadingChecker()

  override def timescope(contents: => Unit): Unit = {
    val newTimescope = threadingChecker.newTimescope(currentThread.get)
    contents
    threadingChecker.closeTimescope(newTimescope).foreach { case (data, valueOption) =>
      valueOption match {
        case Some(value) => pokeSignal(portNames(data), value)
          println(s"${portNames(data)} <- (revert) $value")
        case None => pokeSignal(portNames(data), 0)  // TODO: randomize or 4-state sim
          println(s"${portNames(data)} <- (revert) DC")
      }
    }
  }

  override def step(signal: Clock, cycles: Int): Unit = {
    // TODO: maybe a fast condition for when threading is not in use?
    for (_ <- 0 until cycles) {
      // If a new clock, record the current value so change detection is instantaneous
      if (signal != dut.clock && !lastClockValue.contains(signal)) {
        lastClockValue.put(signal, getClock(signal))
      }

      val thisThread = currentThread.get
      blockedThreads.put(signal, blockedThreads.getOrElseUpdate(signal, Seq()) :+ thisThread)
      scheduler()
      thisThread.waiting.acquire()
    }
  }

  protected val interruptedException = new ConcurrentLinkedQueue[Throwable]()

  protected def onException(e: Throwable) {
    interruptedException.offer(e)
  }

  override def run(testFn: T => Unit): Unit = {
    // Once everything has been prepared, we can start the communications.
    start()
    val mainThread = fork {
      val resetPath = "reset"
      pokeSignal(resetPath, 1)
      step(1)
      pokeSignal(resetPath, 0)

      testFn(dut)
    }
    // TODO: stop abstraction-breaking activeThreads
    require(activeThreads.length == 1)  // only thread should be main
    activeThreads.trimStart(1)  // delete active threads - TODO fix this
    blockedThreads.put(dut.clock, Seq(mainThread))  // TODO dehackify, this allows everything below to kick off

    while (!mainThread.done) {  // iterate timesteps
      val unblockedThreads = new mutable.ArrayBuffer[TesterThread]()

      // Unblock threads waiting on main clock
      unblockedThreads ++= blockedThreads.getOrElse(dut.clock, Seq())
      blockedThreads.remove(dut.clock)
      clockCounter.put(dut.clock, getClockCycle(dut.clock) + 1)

//      println(s"clock step")

      // TODO: allow dependent clocks to step based on test stimulus generator
      // Unblock threads waiting on dependent clock
      require((blockedThreads.keySet - dut.clock) subsetOf lastClockValue.keySet)
      val untrackClocks = lastClockValue.keySet -- blockedThreads.keySet
      for (untrackClock <- untrackClocks) {  // purge unused clocks
        lastClockValue.remove(untrackClock)
      }
      lastClockValue foreach { case (clock, lastValue) =>
        val currentValue = getClock(clock)
        if (currentValue != lastValue) {
          lastClockValue.put(clock, currentValue)
          if (currentValue) {  // rising edge
            unblockedThreads ++= blockedThreads.getOrElse(clock, Seq())
            blockedThreads.remove(clock)
            threadingChecker.advanceClock(clock)

            clockCounter.put(clock, getClockCycle(clock) + 1)
          }
        }
      }

      // Actually run things
      runThreads(unblockedThreads)

      // Propagate exceptions
      if (!interruptedException.isEmpty()) {
        throw interruptedException.poll()
      }

      threadingChecker.timestep()
      Context().env.checkpoint()
      step(1)
    }

    for (thread <- allThreads.clone()) {
      // Kill the threads using an InterruptedException
      if (thread.thread.isAlive) {
        thread.thread.interrupt()
      }
    }
  }
}
