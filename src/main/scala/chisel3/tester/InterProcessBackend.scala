// See LICENSE for license details.

package chisel3.tester

import java.io.File
import java.nio.channels.FileChannel
import java.util.concurrent.{Semaphore, SynchronousQueue, TimeUnit}

import chisel3._
import chisel3.tester.TesterUtils.{getIOPorts, getPortNames}

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.immutable.ListMap
import scala.concurrent.{Await, ExecutionContext, Future, blocking}
import scala.concurrent.duration._
import scala.sys.process.{Process, ProcessLogger}
import java.io.{File, PrintStream}
import java.nio.channels.FileChannel
import java.util.concurrent.TimeUnit

class InterProcessBackend[T <: Module](
  dut: T,
  cmd: Seq[String],
  rnd: scala.util.Random)
  extends BackendInstance[T] with ThreadedBackend {
  val portNameDelimiter: String = "."
  val ioPortNameDelimiter: String = "."
  implicit val logger = new TestErrorLog

  def getModule() = dut

  private val HARD_POKE = 1
//  def getPortNames(dut: Module) = (getDataNames("io", dut.io) ++ getDataNames("reset", dut.reset)).map{case (d: Data, s: String) => (d, (dut.name + "." + s))}.toMap
  // We need to add the dut name to the signal name for the backend.
  protected val portNames = getPortNames(dut) map { case (k: Data, v: String) => (k, (dut.name + portNameDelimiter + v))}
  val (inputs, outputs) = getIOPorts(dut)
  protected def resolveName(signal: Data) =
    portNames.getOrElse(signal, dut.name + portNameDelimiter + signal.toString())

  protected val threadingChecker = new ThreadingChecker()

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

  private class Channel(name: String) {
    private lazy val file = new java.io.RandomAccessFile(name, "rw")
    private lazy val channel = file.getChannel
    @volatile private lazy val buffer = {
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
    implicit def intToByte(i: Int) = i.toByte
    val channel_data_offset_64bw = 4    // Offset from start of channel buffer to actual user data in 64bit words.
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
    def close { file.close }
    buffer order java.nio.ByteOrder.nativeOrder
    new File(name).delete
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
    val in_channel = new Channel(in_channel_name)
    val out_channel = new Channel(out_channel_name)
    val cmd_channel = new Channel(cmd_channel_name)

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
    mwhile(!sendCmd(path)) { }
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

  def pokeSignal(signalName: String, value: BigInt, priority: Int) {
    if (inputsNameToChunkSizeMap contains signalName) {
      _pokeMap(signalName) = value
      isStale = true
    } else {
      val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
      if (id >= 0) {
        poke(id, _chunks getOrElseUpdate (signalName, getChunk(id)), value)
        isStale = true
      } else {
        logger info s"Can't find $signalName in the emulator..."
      }
    }
  }

  override def pokeBits(signal: Bits, value: BigInt, priority: Int) {
    pokeSignal(portNames(signal), value, priority)
  }

  def peekSignal(signalName: String, stale: Boolean): Option[BigInt] = {
    if (isStale) update
    val result =
      if (outputsNameToChunkSizeMap contains signalName) _peekMap get signalName
      else if (inputsNameToChunkSizeMap contains signalName) _pokeMap get signalName
      else {
        val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
        if (id >= 0) {
          Some(peek(id, _chunks getOrElse (signalName, getChunk(id))))
        } else {
          logger info s"Can't find $signalName in the emulator..."
          None
        }
      }
    result
  }

  override def peekBits(signal: Bits, stale: Boolean): BigInt = {
    peekSignal(portNames(signal), stale).getOrElse(BigInt(rnd.nextInt()))
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

    Context().env.testerExpect(value, peekBits(signal, stale), resolveName(signal), None)
  }

  protected val clockCounter = HashMap[Clock, Int]()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }

  protected val lastClockValue = HashMap[Clock, Boolean]()

  protected def scheduler() {
    var testDone: Boolean = false  // set at the end of the clock cycle that the main thread dies on

    while (activeThreads.isEmpty && !testDone) {
      threadingChecker.finishTimestep()
      Context().env.checkpoint()
      step(1)
      clockCounter.put(dut.clock, getClockCycle(dut.clock) + 1)

      if (mainTesterThread.get.done) {
        testDone = true
      }

      threadingChecker.newTimestep(dut.clock)

      // Unblock threads waiting on main clock
      activeThreads ++= blockedThreads.getOrElse(dut.clock, Seq())
      blockedThreads.remove(dut.clock)

      // Unblock threads waiting on dependent clocks
      // TODO: purge unused clocks instead of still continuing to track them
      val waitingClocks = blockedThreads.keySet ++ lastClockValue.keySet - dut.clock
      for (waitingClock <- waitingClocks) {
        val currentClockVal = peekSignal(portNames(waitingClock), false).get.toInt match {
          case 0 => false
          case 1 => true
        }
        if (lastClockValue.getOrElseUpdate(waitingClock, currentClockVal) != currentClockVal) {
          lastClockValue.put(waitingClock, currentClockVal)
          if (currentClockVal == true) {
            activeThreads ++= blockedThreads.getOrElse(waitingClock, Seq())
            blockedThreads.remove(waitingClock)
            threadingChecker.newTimestep(waitingClock)

            clockCounter.put(waitingClock, getClockCycle(waitingClock) + 1)
          }
        }
      }
    }

    if (!testDone) {  // if test isn't over, run next thread
      val nextThread = activeThreads.head
      currentThread = Some(nextThread)
      activeThreads.trimStart(1)
      nextThread.waiting.release()
    } else {  // if test is done, return to the main scalatest thread
      finish()
      scalatestWaiting.release()
    }

  }

  override def step(signal: Clock, cycles: Int): Unit = {
    // TODO: clock-dependence
    // TODO: maybe a fast condition for when threading is not in use?
    for (_ <- 0 until cycles) {
      val thisThread = currentThread.get
      threadingChecker.finishThread(thisThread, signal)
      // TODO this also needs to be called on thread death

      blockedThreads.put(signal, blockedThreads.getOrElseUpdate(signal, Seq()) :+ thisThread)
      scheduler()
      thisThread.waiting.acquire()
    }
  }

  protected var scalatestThread: Option[Thread] = None
  protected var mainTesterThread: Option[TesterThread] = None
  protected val scalatestWaiting = new Semaphore(0)
  protected val interruptedException = new SynchronousQueue[Throwable]()

  protected def onException(e: Throwable) {
    scalatestThread.get.interrupt()
    interruptedException.offer(e, 10, TimeUnit.SECONDS)
  }

  override def run(testFn: T => Unit): Unit = {
    // Once everything has been prepared, we can start the communications.
    start()
    val resetPath = dut.name + ".reset"
    pokeSignal(resetPath, 1, HARD_POKE)
    step(1)
    pokeSignal(resetPath, 0, HARD_POKE)

    val mainThread = fork(
      testFn(dut)
    )

    require(activeThreads.length == 1)  // only thread should be main
    activeThreads.trimStart(1)
    currentThread = Some(mainThread)
    scalatestThread = Some(Thread.currentThread())
    mainTesterThread = Some(mainThread)

    mainThread.waiting.release()
    try {
      scalatestWaiting.acquire()
    } catch {
      case e: InterruptedException =>
        throw interruptedException.poll(10, TimeUnit.SECONDS)
    }

    mainTesterThread = None
    scalatestThread = None
    currentThread = None

    for (thread <- allThreads.clone()) {
      // Kill the threads using an InterruptedException
      if (thread.thread.isAlive) {
        thread.thread.interrupt()
      }
    }
  }
}
