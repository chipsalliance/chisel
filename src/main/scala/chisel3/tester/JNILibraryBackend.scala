// See LICENSE for license details.

package chisel3.tester

import java.io.File
import java.nio.file.Paths
import java.util.concurrent.ConcurrentLinkedQueue
import scala.collection.mutable

import chisel3._
import chisel3.tester.TesterUtils.{getIOPorts, getPortNames}

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.collection.immutable.ListMap
import firrtl.util.BackendCompilationUtilities._
import JNILibraryBackend._

object JNILibraryBackend {
  val singletonName = "chisel3.tester.SingletonLoaderShim"
  var classLoader: ClassLoader = ClassLoader.getSystemClassLoader()
  if (classLoader == null)
    classLoader = classOf[chisel3.tester.SingletonLoaderShim].getClassLoader
  if (classLoader == null)
    throw new Exception("Can't find classloader.")
  // Load and initialize the singelton.
  val singletonLoaderShim: Option[SingletonLoaderShimInterface] = Some(SingletonLoaderShim.getInstance())

//  println(s"loading ${singletonName} using ${classLoader.toString}")
  //  val singletonLoaderShim: Option[SingletonLoaderShim] = try {
//    val clazz = Class.forName(singletonName, true, classLoader)
//    //  val singletonLoaderShim: SingletonLoaderShim = instance.INSTANCE
//    val method = clazz.getDeclaredMethod("getInstance")
//    val singletonLoaderShim: SingletonLoaderShim = method.invoke(null).asInstanceOf[SingletonLoaderShim]
//    //  val singletonLoaderShim: SingletonLoaderShim = instance.asInstanceOf[SingletonLoaderShim].getInstance()
//    //  val singletonLoaderShim: SingletonLoaderShim = SingletonLoaderShim.getInstance
//    Some(singletonLoaderShim)
//  } catch {
//    case t: Throwable =>
//      println("Exception: " + t)
//      None
//  }
  def loadJNITestShim(fullPath: String): Unit = {
    println (s"loadJNITestShim: shim $singletonLoaderShim path $fullPath")
    singletonLoaderShim.get.loadJNITestShim(fullPath)
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

  val (inputsNameToChunkSizeMap, outputsNameToChunkSizeMap) = {
    def genChunk(args: (Data, String)) = args match {
      case (pin, name) => name -> ((pin.getWidth-1)/64 + 1)
    }
    (ListMap((inputs map genChunk): _*), ListMap((outputs map genChunk): _*))
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

  def pokeSignal(signalName: String, value: BigInt) {
    if (inputsNameToChunkSizeMap contains signalName) {
      _pokeMap(signalName) = value
      isStale = true
    } else {
      val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
      if (id >= 0) {
        poke(id, _chunks getOrElseUpdate (signalName, getSignalWordSize(id)), value)
        isStale = true
      } else {
        throw InterProcessBackend.InterProcessBackendException(s"Can't find $signalName in the emulator...")
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
    val result =
      if (outputsNameToChunkSizeMap contains signalName) {
        _peekMap(signalName)
      } else if (inputsNameToChunkSizeMap contains signalName) {
        _pokeMap(signalName)
      } else {
        val id = _signalMap getOrElseUpdate (signalName, getId(signalName))
        if (id >= 0) {
          peek(id, _chunks getOrElse (signalName, getSignalWordSize(id)))
        } else {
          throw InterProcessBackend.InterProcessBackendException(s"Can't find $signalName in the emulator...")
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
    finish()
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
