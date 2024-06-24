package chisel3
package simulator

import chisel3.util._
import chisel3.reflect.DataMirror
import chisel3.experimental.SourceInfo
import chisel3.simulator.Simulator._
import svsim._

import scala.util.DynamicVariable
import scala.util.{Failure, Success, Try}

trait SimFailureException extends Exception {
  val message:    String
  val sourceInfo: SourceInfo
}

case class SVAssertionFailure(message: String)(implicit val sourceInfo: SourceInfo) extends SimFailureException

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

case class DutContext(clock: Option[Clock], ports: Seq[(Data, ModuleInfo.Port)], maxWaitCycles: Int = 1000)

object DutContext {
  private val dynamicVariable = new scala.util.DynamicVariable[Option[DutContext]](None)
  def withValue[T](dut: DutContext)(body: => T): T = {
    require(dynamicVariable.value.isEmpty, "Nested simulations are not supported.")
    dynamicVariable.withValue(Some(dut))(body)
  }
  def current: DutContext = dynamicVariable.value.get
}

trait ChiselSimAPI extends PeekPokeAPI {
  import ChiselSimAPI._

  protected var dutContext = new DynamicVariable[Option[DutContext]](None)

  def testName: Option[String]

  def testClassName: Option[String] = Some(this.getClass.getName)

  private def peekHierValueRec[B <: Data](signal: B)(implicit sourceInfo: SourceInfo): HierarchicalValue = {
    signal match {
      case v: Vec[_] =>
        VecValue(v, v.map(peekHierValueRec))

      case b: Record =>
        BundleValue(b, b.elements.map { case (name, field) => name -> peekHierValueRec(field) }.toMap)

      case sig: Element =>
        LeafValue(sig, sig.peekValue())
    }
  }

  trait testableAggregate[T <: Aggregate] {
    protected val sig: T
    def peekHierValue()(implicit sourceInfo: SourceInfo): HierarchicalValue = peekHierValueRec(sig)
  }

  implicit final class testableRecord[T <: Record](protected val sig: T)(implicit sourceInfo: SourceInfo)
      extends testableAggregate[T]

  implicit final class testableVec[U <: Data, T <: Vec[U]](protected val sig: T)(implicit sourceInfo: SourceInfo)
      extends testableAggregate[T]

  implicit final class testableChiselEnum[T <: ChiselEnum](protected val sig: T)(implicit sourceInfo: SourceInfo) {

  }

  sealed trait clockedInterface {
    protected def clock = DutContext.current.clock.get // TODO: handle clock not being present
    protected def maxWaitCycles = DutContext.current.maxWaitCycles

    def pokeRec[B <: Data](signal: B, data: B)(implicit sourceInfo: SourceInfo): Unit = {
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
                      .queryNameGuess(value)} (=${HierarchicalValue(value)})"
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

    def enqueue(data: T) = {
      require(data.isLit, "enqueued data must be literal!")
      sig.valid.poke(true)
      pokeRec(sig.bits, data)
      clock.step()
      sig.valid.poke(false)
    }

    def enqueueSeq(dataSeq: Seq[T]) = {
      for (data <- dataSeq) {
        enqueue(data)
      }
    }

    def waitForValid() = {
      clock.stepUntil(sig.valid, 1, maxWaitCycles)
      val timeout = sig.valid.peekValue().asBigInt == 0
      chisel3.assert(!timeout, s"Timeout after $maxWaitCycles cycles waiting for valid")
    }

    def dequeue(): HierarchicalValue = {
      waitForValid()
      val value = peekHierValueRec(sig)
      clock.step()
      value
    }

    def expectDequeue(expected: T, message: String): Unit =
      expectDequeue(expected, Some(message))

    def expectDequeue(expected: T, message: Option[String] = None): Unit = {
      require(expected.isLit, "expected value must be a literal!")
      waitForValid()
      expectRec(sig.bits, expected, message)
      clock.step()
    }

    def expectDequeueSeq(dataSeq: Seq[T], message: String): Unit = {
      for ((data, i) <- dataSeq.zipWithIndex) {
        expectDequeue(data, message)
      }
    }

    def expectDequeueSeq(dataSeq: Seq[T]): Unit = {
      for ((exp, i) <- dataSeq.zipWithIndex) {
        expectDequeue(exp, s"Element $i was different from expected value: $exp!")
      }
    }

  }

  implicit final class testableDecoupledIO[T <: Data](sig: DecoupledIO[T])(implicit sourceInfo: SourceInfo)
      extends clockedInterface {

    def enqueue(data: T) = {
      require(data.isLit, "enqueued data must be literal!")
      sig.valid.poke(true)
      pokeRec(sig.bits, data)
      clock.stepUntil(sig.ready, 1, maxWaitCycles)
      assert(sig.ready.peekValue().asBigInt == 1, s"Timeout after $maxWaitCycles cycles waiting for ready")
      clock.step()
      sig.valid.poke(false)
    }

    def enqueueSeq(dataSeq: Seq[T]) = {
      for (data <- dataSeq) {
        enqueue(data)
      }
    }

    def waitForValid() = {
      clock.stepUntil(sig.valid, 1, maxWaitCycles)
      val timeout = sig.valid.peekValue().asBigInt == 0
      assert(!timeout, s"Timeout after $maxWaitCycles cycles waiting for valid")
    }

    def dequeue(): HierarchicalValue = {
      sig.ready.poke(true)
      waitForValid()
      val value = peekHierValueRec(sig)
      clock.step()
      sig.ready.poke(false)
      value
    }

    def expectDequeue(expected: T, message: String): Unit =
      expectDequeue(expected, Some(message))

    def expectDequeue(expected: T, message: Option[String] = None): Unit = {
      require(expected.isLit, "expected value must be a literal!")
      sig.ready.poke(true)
      waitForValid()
      expectRec(sig.bits, expected, message)
      clock.step()
      sig.ready.poke(false)
    }

    def expectDequeueSeq(dataSeq: Seq[T], message: String): Unit = {
      for ((data, i) <- dataSeq.zipWithIndex) {
        expectDequeue(data, message)
      }
    }

    def expectDequeueSeq(dataSeq: Seq[T]): Unit = {
      for ((exp, i) <- dataSeq.zipWithIndex) {
        expectDequeue(exp, s"Element $i was different from expected value: $exp!")
      }
    }
  }

  /**
    * @param rootTestRunDir
    * @return workspace
    */
  def workspacePath(rootTestRunDir: Option[String]): os.Path = {
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

  case class TestBuilder[T <: RawModule, B <: Backend](
    modGen:   () => T,
    backend:  B,
    settings: ChiselSimSettings[B]) {

    def withTrace(traceStyle: TraceStyle): TestBuilder[T, B] = copy(
      settings = settings.copy(traceStyle = traceStyle)
    )

    def apply[U](body: (T) => U): String = {
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
                  maxCycles = 1,
                  inPhaseValue = 0,
                  outOfPhaseValue = 1,
                  sentinel = None
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
  }
}

object ChiselSimAPI {
  def sanitizeFileName(name: String): String = {
    name.replaceAll(" ", "_").replaceAll("[^\\w\\.\\-]+", "")
  }

  def prunePath(dirPath: String): Unit = pruneDir(os.FilePath(dirPath))

  def pruneDir(dirPath: os.FilePath): Unit = {
    val dir = dirPath.resolveFrom(os.pwd)
    if (!os.exists(dir)) {
      os.makeDir.all(dir)
    } else {
      for (fileOrDir <- os.list(dir))
        os.remove.all(fileOrDir)
    }
  }
}
