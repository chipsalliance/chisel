// See LICENSE for license details.

package chisel3.tester

import java.io.File
import java.nio.file.Paths
import java.util.concurrent.{Semaphore, SynchronousQueue, TimeUnit}

import chisel3._
import chisel3.tester.TesterUtils.{getIOPorts, getPortNames}

import scala.collection.mutable.{ArrayBuffer, HashMap}
import firrtl.util.BackendCompilationUtilities._

import JNILibraryBackend._

object JNILibraryBackend {
  var shimLoaded = false
  def loadJNITestShim(fullPath: String): Unit = {
    if (!shimLoaded) {
      System.load(fullPath)
    }
    shimLoaded = true
  }
}

class JNILibraryBackend[T <: Module](
  dut: T,
  libraryFullPath: String,
  rnd: scala.util.Random)
  extends BackendInstance[T] with ThreadedBackend {
  val portNameDelimiter: String = "."
  val ioPortNameDelimiter: String = "."
  implicit val logger = new TestErrorLog

  def getModule() = dut

  private val HARD_POKE = 1
//  def getPortNames(dut: Module) = (getDataNames("io", dut.io) ++ getDataNames("reset", dut.reset)).map{case (d: Data, s: String) => (d, (dut.name + "." + s))}.toMap
  // We need to add the dut name to the signal name for the backend.
  protected val portNames = getPortNames(dut) map { case (k: Data, v: String) => (k, v)}
  val (inputs, outputs) = getIOPorts(dut)
  protected def resolveName(signal: Data) =
    portNames.getOrElse(signal, dut.name + portNameDelimiter + signal.toString())

  protected val threadingChecker = new ThreadingChecker()

  private[tester] case class TestApplicationException(exitVal: Int, lastMessage: String)
    extends RuntimeException(lastMessage)

  val (inputsNameToChunkSizeMap, outputsNameToChunkSizeMap) = {
    def genChunk(args: (Data, String)) = args match {
      case (pin, name) => name -> ((pin.getWidth-1)/64 + 1)
    }
    (HashMap((inputs map genChunk): _*), HashMap((outputs map genChunk): _*))
  }
  implicit def int(x: Int):  BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  implicit def int(x: Long): BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  private var isStale = false
  private val _pokeMap = HashMap[String, BigInt]()
  private val _peekMap = HashMap[String, BigInt]()
  private val _signalMap = HashMap[String, Int]()
  private val _chunks = HashMap[String, Int]()
  private val _logs = ArrayBuffer[String]()
  private val bpw = 8;
  private val inputSize = inputsNameToChunkSizeMap.values.sum * bpw
  private val outputSize = outputsNameToChunkSizeMap.values.sum * bpw

  private def dumpLogs() {
    _logs foreach logger.info
    _logs.clear
  }

  val jniTest = new JNITestAPI
  // Load the shared library containing our test implementation.
  jniTest.init(libraryFullPath, "")
  private val inChannel = new InterProcessBackend.VerilatorChannelProtocol(jniTest.inputBuffer)
  private val outChannel = new InterProcessBackend.VerilatorChannelProtocol(jniTest.outputBuffer)
  inChannel.consume
  inChannel.release
  outChannel.release

  private def recvOutputs() = {
    _peekMap.clear
//    outChannel.aquire
    jniTest.getOutputSignals(outChannel.buffer)
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
//    inChannel.aquire
    val ready = inChannel.ready
    if (ready) {
      (inputsNameToChunkSizeMap.toList foldLeft 0){case (off, (in, chunk)) =>
        val value = _pokeMap getOrElse (in, BigInt(0))
        (0 until chunk) foreach (i => inChannel(off + i) = (value >> (64 * i)).toLong)
        off + chunk
      }
      inChannel.produce
      jniTest.setInputSignals(inChannel.buffer)
    }
    inChannel.release
    ready
  }

  private def update() {
    sendInputs
    jniTest.update()
    recvOutputs()
    isStale = false
  }

  private def takeStep() {
    sendInputs
    jniTest.step(1)
    sendInputs
    recvOutputs()
    dumpLogs()
  }

  private def getId(path: String) = {
    val fullPath = dut.name + portNameDelimiter + path
    jniTest.getID(fullPath)
  }

  private def getSignalWordSize(id: Int) = {
    jniTest.getSignalWordSize(id)
  }

  private def poke(id: Int, chunk: Int, value: BigInt, force: Boolean = false): Unit = {
    val longData = Array[Long](chunk)
    if (chunk <= 1) {
      longData(0) = value.toLong
    } else {
      (0 until chunk) foreach (i => longData(i) = (value >> (64*i)).toLong)
    }
    jniTest.poken(id, chunk, longData)
  }

  private def peek(id: Int, chunk: Int): BigInt = {
    val longData = Array[Long](chunk)
    jniTest.peekn(id, chunk, longData)
    if (chunk <= 1) {
      BigInt(longData(0))
    } else {
      ((0 until chunk) foldLeft BigInt(0))(
        (res, i) => res | longData(i) << (64*i))
    }
  }

  private def start() {
    reset(5)
    recvOutputs()
  }

  def pokeSignal(signalName: String, value: BigInt, priority: Int) {
    if (inputsNameToChunkSizeMap contains signalName) {
      _pokeMap(signalName) = value
      isStale = true
    } else {
      val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
      if (id >= 0) {
        poke(id, _chunks getOrElseUpdate (signalName, getSignalWordSize(id)), value)
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
    if (isStale) {
      update
    }
    val result =
      if (outputsNameToChunkSizeMap contains signalName) _peekMap get signalName
      else if (inputsNameToChunkSizeMap contains signalName) _pokeMap get signalName
      else {
        val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
        if (id >= 0) {
          Some(peek(id, _chunks getOrElse (signalName, getSignalWordSize(id))))
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
    jniTest.reset(n)
    recvOutputs()
    jniTest.start()
  }

  def finish() {
    jniTest.finish()
    dumpLogs
  }


  override def expectBits(signal: Bits, value: BigInt, stale: Boolean): Unit = {
    require(!stale, "Stale peek not yet implemented")

    try {
      Context().env.testerExpect(value, peekBits(signal, stale), resolveName(signal), None)
    } catch {
      case e: Throwable =>
        jniTest.abort()
        throw e
    }
  }

  protected val clockCounter = HashMap[Clock, Int]()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }

  protected val lastClockValue = HashMap[Clock, Boolean]()

  protected def scheduler() {
    var testDone: Boolean = false  // set at the end of the clock cycle that the main thread dies on
    var exception: Option[Throwable] = None

    try {
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
    } catch {
      case e: Throwable =>
        exception = Some(e)
        testDone = true
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
    exception match {
      case Some(e: Throwable) =>
        throw e
      case None =>
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
    scalatestThread match {
      case Some(t: Thread) =>
        t.interrupt()
      case None =>
    }
    interruptedException.offer(e, 10, TimeUnit.SECONDS)
  }

  override def run(testFn: T => Unit): Unit = {
    // Once everything has been prepared, we can start the communications.
    start()
    val resetPath = "reset"
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

object JNILibraryTesterBackend {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule
//  import JNITest

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  def start[T <: Module](dutGen: => T, options: TesterOptionsManager): BackendInstance[T] = {
    val optionsManager = options
    val (result, updatedOptions) = CSimulator.generateVerilog(dutGen, optionsManager)
    result match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val dut = getTopModule(circuit).asInstanceOf[T]
        val dir = new File(updatedOptions.targetDirName)
        val shimPieces = Seq("chisel3_tester_JNITestAPI")
        val extraObjects = Seq.empty
        val libName = updatedOptions.testerOptions.backendName match {
          case "verilator" | "jni" =>
            val cppHarnessFile = VerilatorTesterBackend.generateHarness(
              dut, circuit.name, dir, updatedOptions
            )
            VerilatorTesterBackend.buildSharedLibrary(circuit.name, dir, cppHarnessFile, updatedOptions, shimPieces, extraObjects)

          case "vcs" =>
            val vcsHarnessFile = VCSTesterBackend.generateHarness(
              dut, circuit.name, dir, updatedOptions
            )
            VCSTesterBackend.buildSharedLibrary(circuit.name, dir, vcsHarnessFile, updatedOptions)
        }
        // We need to load our C++ shim at least once.
        loadJNITestShim(Paths.get(dir.getAbsolutePath, s"${shimPieces.head}.${sharedLibraryExtension}").toString)
        val seed = optionsManager.testerOptions.testerSeed
        val rnd = new scala.util.Random(seed)
        new JNILibraryBackend[T](dut, Paths.get(dir.getAbsolutePath, libName).toString, rnd)
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
