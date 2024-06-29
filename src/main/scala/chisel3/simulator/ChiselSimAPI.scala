// SPDX-License-Identifier: Apache-2.0

package chisel3
package simulator

import chisel3.util._
import chisel3.reflect.DataMirror
import chisel3.experimental.SourceInfo
import svsim._

import scala.util.{Failure, Success, Try}

trait TraceStyle {
  def isEnabled: Boolean = true
}

object TraceStyle {
  case object NoTrace extends TraceStyle {
    override def isEnabled: Boolean = false
  }

  case class Vcd(filename: String = "trace.vcd", traceUnderscore: Boolean = false) extends TraceStyle

  case class Vpd(filename: String = "trace.vpd", traceUnderscore: Boolean = false) extends TraceStyle

  case class Fst(filename: String = "trace.fst", traceUnderscore: Boolean = false) extends TraceStyle
}

trait ChiselSimAPI extends PeekPokeAPI {
  import ChiselSimAPI._

  def testName: Option[String]

  def testClassName: Option[String] = Some(this.getClass.getName)

  private def peekHierValueRec[B <: Data](signal: B)(implicit sourceInfo: SourceInfo): SimValue = {
    signal match {
      case v: Vec[_] =>
        VecValue(v.map(peekHierValueRec))

      case b: Record =>
        BundleValue(b.elements.map { case (name, field) => name -> peekHierValueRec(field) }.toMap)

      case sig: Element =>
        LeafValue.fromSimulationValue(sig.peekValue())
    }
  }

  trait testableAggregate[T <: Aggregate] {
    protected val sig: T
  }

  implicit final class testableRecord[T <: Record](protected val sig: T)(implicit val sourceInfo: SourceInfo)
      extends testableAggregate[T]

  implicit final class testableVec[U <: Data, T <: Vec[U]](protected val sig: T)(implicit val sourceInfo: SourceInfo)
      extends testableAggregate[T]

  implicit final class testableChiselEnum[T <: ChiselEnum](protected val sig: T)(implicit val sourceInfo: SourceInfo) {}

  protected def currentClock: Option[testableClock] =
    DutContext.current.clock.map(testableClock)

  sealed trait clockedInterface {
    protected val maxWaitCycles: Int = DutContext.current.maxWaitCycles
    protected val clock:         testableClock = currentClock.get
    protected def stepClock():   Unit = clock.step()

    protected def waitForSignal[D <: Data](
      signal:        D,
      expectedValue: BigInt = 1,
      maxCycles:     Option[Int] = None
    )(
      implicit sourceInfo: SourceInfo
    ) = {
      val maximumWaitCycles = maxCycles.getOrElse(maxWaitCycles)
      val isSigned = signal.isInstanceOf[SInt]
      val module = AnySimulatedModule.current
      val simulationPort = module.port(signal)

      module.willPeek()
      def getValue = {
        println(s"--- Waiting for signal ${signal.pathName} to be $expectedValue")
        simulationPort.get(isSigned = isSigned).asBigInt
      }

      // clock.stepUntil(signal, expectedValue, maximumWaitCycles)
      // if (signal.peekValue().asBigInt != expectedValue)
      //   throw TimedOutWaiting(maximumWaitCycles, signal.pathName)
      var cycles = 0
      while (getValue != expectedValue) {
        println(s"---    signal ${signal.pathName} is $getValue")
        if (cycles == maximumWaitCycles)
          throw TimedOutWaiting(cycles, signal.pathName)
        stepClock()
        cycles += 1
      }
      println(s"---    signal ${signal.pathName} is $expectedValue")

    }

    protected def pokeRec[B <: Data](signal: B, data: B)(implicit sourceInfo: SourceInfo): Unit = {
      (signal, data) match {
        case (s: Vec[_], d: Vec[_]) =>
          require(s.length == d.length, s"input is of length ${s.length} while data ia of size ${d.length}")
          d.zip(s).foreach {
            case (di, si) =>
              pokeRec(si, di)
          }
        case (s: Record, d: Record) =>
          d.elements.foreach {
            case (name, value) =>
              pokeRec(s.elements(name), value)
          }
        case (s, d) =>
          s.poke(d.litValue)
      }
    }

    def expectLeaf[B <: chisel3.Element](signal: B, data: B, message: String)(implicit sourceInfo: SourceInfo): Unit = {
      signal.expect(
        data.litValue,
        _.asBigInt,
        (observed: BigInt, expected: BigInt) => message,
        sourceInfo
      )
    }

    def expectLeaf[B <: chisel3.Element](signal: B, data: B)(implicit sourceInfo: SourceInfo): Unit = {
      signal.expect(
        data.litValue,
        _.asBigInt,
        (observed: BigInt, expected: BigInt) => s"Expectation failed: observed value $observed != $expected",
        sourceInfo
      )
    }

    def expectRec[B <: Data](
      signal:   B,
      expected: B,
      message:  Option[String] = None
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      (signal, expected) match {
        case (s: VecLike[_], d: VecLike[_]) =>
          require(s.length == d.length, s"Signal $s and expected $d have different lengths!")
          s.zip(d).zipWithIndex.foreach {
            case ((si, ei), i) =>
              expectRec(si, ei, message.orElse(Some(s"Expected ${d}. Element $i differs.")))
          }
        case (s: Record, d: Record) =>
          d.elements.foreach {
            case (name, value) =>
              val sigEl = s.elements(name)
              expectRec(
                sigEl,
                value,
                message.orElse(
                  Some(
                    s"\n${DataMirror.queryNameGuess(sigEl)} (=${peekHierValueRec(sigEl)}) =/= ${DataMirror
                      .queryNameGuess(value)} (=${value})"
                  )
                )
              )
          }
        case (s: chisel3.Element, d: chisel3.Element) =>
          message match {
            case Some(message) =>
              expectLeaf(s, d, message)(sourceInfo)
            case _ =>
              expectLeaf(s, d)(sourceInfo)
          }
        case _ =>
          throw new Exception("type not supported!")
      }
    }
  }

  implicit final class testableValidIO[T <: Data](sig: ValidIO[T])(implicit sourceInfo: SourceInfo)
      extends clockedInterface {

    private def valid = sig.valid

    def enqueue(
      data: T
    )(
      implicit sourceInfo: SourceInfo
    ) = {
      require(data.isLit, "enqueued data must be literal!")
      pokeRec(sig.bits, data)
      valid.poke(1)
      stepClock()
      valid.poke(0)
    }

    def enqueueSeq(
      dataSeq: Seq[T]
    )(
      implicit sourceInfo: SourceInfo
    ) = {
      for (data <- dataSeq) {
        enqueue(data)
      }
    }

    def waitForValid(
    )(
      implicit sourceInfo: SourceInfo
    ) = waitForSignal(valid)

    def dequeue(
    )(
      implicit sourceInfo: SourceInfo
    ): SimValue = {
      waitForValid()
      val value = peekHierValueRec(sig)
      stepClock()
      value
    }

    def expectDequeue(
      expected: T,
      message:  String
    )(
      implicit sourceInfo: SourceInfo
    ): Unit =
      expectDequeue(expected, Some(message))

    def expectDequeue(
      expected: T,
      message:  Option[String] = None
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(expected.isLit, "expected value must be a literal!")
      waitForValid()
      expectRec(sig.bits, expected, message)
      stepClock()
    }

    def expectDequeueSeq(
      dataSeq: Seq[T],
      message: String
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      for ((data, i) <- dataSeq.zipWithIndex) {
        expectDequeue(data, message)
      }
    }

    def expectDequeueSeq(
      dataSeq: Seq[T]
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      for ((exp, i) <- dataSeq.zipWithIndex) {
        expectDequeue(exp, s"Element $i was different from expected value: $exp!")
      }
    }

  }

  implicit final class testableDecoupledIO[T <: Data](sig: DecoupledIO[T])(implicit sourceInfo: SourceInfo)
      extends clockedInterface {
    private def valid = sig.valid
    private def ready = sig.ready

    def enqueue(
      data: T
    )(
      implicit sourceInfo: SourceInfo
    ) = {
      require(data.isLit, "enqueued data must be literal!")
      println(s">>> [enqueue] ${data.litValue}")
      pokeRec(sig.bits, data)
      println(s">>> [enqueue] valid.poke(1)")
      valid.poke(1)
      println(s">>> [enqueue] waitForReady")
      waitForReady()
      println(s">>> [enqueue] step")
      stepClock()
      println(s">>> [enqueue] valid.poke(0)")
      valid.poke(0)
    }

    def enqueueSeq(
      dataSeq: Seq[T]
    )(
      implicit sourceInfo: SourceInfo
    ) = {
      for (data <- dataSeq) {
        enqueue(data)
      }
    }

    def waitForValid(
    )(
      implicit sourceInfo: SourceInfo
    ) = waitForSignal(valid)

    def waitForReady(
    )(
      implicit sourceInfo: SourceInfo
    ) = waitForSignal(ready)

    def dequeue(
    )(
      implicit sourceInfo: SourceInfo
    ): SimValue = {
      ready.poke(1)
      waitForValid()
      val value = peekHierValueRec(sig)
      stepClock()
      ready.poke(0)
      value
    }

    def expectDequeue(
      expected: T,
      message:  String
    )(
      implicit sourceInfo: SourceInfo
    ): Unit =
      expectDequeue(expected, Some(message))

    def expectDequeue(
      expected: T,
      message:  Option[String] = None
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(expected.isLit, "expected value must be a literal!")
      ready.poke(1)
      waitForValid()
      expectRec(sig.bits, expected, message)
      stepClock()
      ready.poke(0)
    }

    def expectDequeueSeq(
      dataSeq: Seq[T],
      message: String
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      for ((data, i) <- dataSeq.zipWithIndex) {
        expectDequeue(data, message)
      }
    }

    def expectDequeueSeq(
      dataSeq: Seq[T]
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      for ((exp, i) <- dataSeq.zipWithIndex) {
        expectDequeue(exp, s"Element $i was different from expected value: $exp!")
      }
    }
  }

  /**
    * @param rootTestRunDir
    * @return workspace
    */
  protected def workspacePath(rootTestRunDir: Option[String]): os.Path = {
    val rootPath = rootTestRunDir.map(os.FilePath(_).resolveFrom(os.pwd)).getOrElse(os.pwd)
    (testClassName.toSeq ++ testName)
      .map(sanitizeFileName)
      .foldLeft(rootPath)(_ / _)
  }

  // TODO: Not possible know the actual workingDirectoryPath as assembled in [[Workspace]] or [[Simulator]]
  //   _without_ a workspace/simlator instance! This is just a best guess!
  def guessWorkDir[B <: Backend](settings: ChiselSimSettings[B]): os.Path =
    workspacePath(settings.testRunDir) / s"workdir-${settings.backendName}"

  val simulationLogFileName = "simulation-log.txt"

  def simulate[T <: RawModule, B <: Backend, U](modGen: => T, settings: ChiselSimSettings[B])(body: T => U): String =
    _simulate(() => modGen, settings)(body)

  private def _simulate[T <: RawModule, B <: Backend, U](
    modGen:   () => T,
    settings: ChiselSimSettings[B]
  )(body:     T => U
  ): String = {
    val workspace = new svsim.Workspace(
      path = workspacePath(settings.testRunDir).toString,
      workingDirectoryPrefix = settings.workingDirectoryPrefix
    )

    if (settings.resetWorkspace) {
      workspace.reset()
    } else {
      prunePath(workspace.primarySourcesPath)
      prunePath(workspace.generatedSourcesPath)
    }

    val elaboratedModule = workspace.elaborateGeneratedModule(modGen, settings.chiselArgs, settings.firtoolArgs)
    workspace.generateAdditionalSources()

    val supportArtifactsPath = os.FilePath(workspace.supportArtifactsPath).resolveFrom(os.pwd)
    val primarySourcesPath = os.FilePath(workspace.primarySourcesPath).resolveFrom(os.pwd)

    val layerSvFiles = os
      .list(supportArtifactsPath)
      .filter(p => os.isFile(p) && p.baseName.startsWith("layers_") && p.ext == "sv")
    layerSvFiles.foreach(p => os.move(p, primarySourcesPath / p.last, replaceExisting = true))

    val simulation = workspace.compile(settings.backend)(
      settings.backendName,
      settings.commonSettings,
      settings.backendSettings,
      customSimulationWorkingDirectory = settings.customSimulationWorkingDirectory,
      verbose = settings.verboseCompile
    )

    val simulationOutcome = Try {
      simulation
        .runElaboratedModule(
          elaboratedModule = elaboratedModule,
          conservativeCommandResolution = settings.conservativeCommandResolution,
          verbose = settings.verboseRun,
          traceEnabled = settings.traceStyle.isEnabled,
          executionScriptLimit = settings.executionScriptLimit,
          executionScriptEnabled = settings.executionScriptEnabled
        ) { module =>
          val dut = module.wrapped
          val clock = dut match {
            case m: Module =>
              val clock = module.port(m.clock)
              val reset = module.port(m.reset)
              reset.set(1)
              clock.tick(
                timestepsPerPhase = 1,
                cycles = 1,
                inPhaseValue = 0,
                outOfPhaseValue = 1
              )
              reset.set(0)
              Some(m.clock)
            case _ => None
          }
          DutContext.withValue(DutContext(clock, Seq.empty)) { body(module.wrapped) }
        }
      simulation.workingDirectoryPath
    }.recoverWith {
      // TODO: a more robust way for detecting assertions
      // FIXME: Ensure graceful simulation ending. Currently a test failure leaves a malformed FST trace in Verilator.
      //   VCD files seem to be still parsable by waveform viewers, but that's probably only due to their simpler textual structure.
      case svsim.Simulation.UnexpectedEndOfMessages =>
        val logFile = os.FilePath(simulation.workingDirectoryPath).resolveFrom(os.pwd) / simulationLogFileName
        val logLines = os.read.lines(logFile)
        // FIXME only implemented for Verilator
        logLines.zipWithIndex.collectFirst {
          case (line, _) if line.contains("Verilog $finish") =>
            // Simulation does not immediately exit on $finish, so we need to ignore assertions that happen after a call to $finish
            Success(simulation.workingDirectoryPath)
          case (line, lineNum) if line.contains("Assertion failed") =>
            val message = (logLines.drop(lineNum).filterNot(_.contains("Verilog $stop"))).mkString("\n")
            Failure(SVAssertionFailure(message))
        }
          .getOrElse(Failure(svsim.Simulation.UnexpectedEndOfMessages))
    }
    simulationOutcome.get

  }

  case class TestBuilder[T <: RawModule, B <: Backend](
    modGen:   () => T,
    settings: ChiselSimSettings[B]) {

    def withTrace(traceStyle: TraceStyle): TestBuilder[T, B] = copy(
      settings = settings.copy(traceStyle = traceStyle)
    )

    def apply[U](body: T => U): String = {
      _simulate(modGen, settings)(body)
    }
  }

  def test[T <: Module, B <: Backend](
    module:  => T,
    backend: B = verilator.Backend.initializeFromProcessEnvironment()
  ): TestBuilder[T, B] =
    new TestBuilder[T, B](() => module, ChiselSimSettings(backend))

  def test[T <: Module, B <: Backend](
    module:   => T,
    settings: ChiselSimSettings[B]
  ): TestBuilder[T, B] =
    new TestBuilder[T, B](() => module, settings)
}

object ChiselSimAPI {
  private def sanitizeFileName(name: String): String = {
    name.replaceAll(" ", "_").replaceAll("[^\\w\\.\\-]+", "")
  }

  private def prunePath(dirPath: String): Unit = pruneDir(os.FilePath(dirPath))

  private def pruneDir(dirPath: os.FilePath): Unit = {
    val dir = dirPath.resolveFrom(os.pwd)
    if (!os.exists(dir)) {
      os.makeDir.all(dir)
    } else {
      for (fileOrDir <- os.list(dir))
        os.remove.all(fileOrDir)
    }
  }
}

object ChiselSim extends ThreadedChiselSimAPI {
  val testName = None
}
