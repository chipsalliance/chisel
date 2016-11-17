// See LICENSE for license details.

package chisel3.iotesters

import chisel3._
import java.io.File
import firrtl.{ExecutionOptionsManager, HasFirrtlOptions}
import firrtl_interpreter.{FirrtlRepl, ReplConfig, HasReplConfig, HasInterpreterOptions}

import scala.util.DynamicVariable

object Driver {
  private val backendVar = new DynamicVariable[Option[Backend]](None)
  private[iotesters] def backend = backendVar.value

  private val optionsManagerVar = new DynamicVariable[Option[TesterOptionsManager]](None)
  private[iotesters] def optionsManager = optionsManagerVar.value.getOrElse(new TesterOptionsManager)


  def execute[T <: Module](
                            dutGenerator: () => T,
                            optionsManager: TesterOptionsManager
                          )
                          (
                            testerGen: T => PeekPokeTester[T]
                          ): Boolean = {
    val testerOptions = optionsManager.testerOptions

    val (dut, backend) = testerOptions.backendName match {
      case "firrtl"    =>
        setupFirrtlTerpBackend(dutGenerator, optionsManager)
      case "verilator" =>
        setupVerilatorBackend(dutGenerator, optionsManager)
      case "vcs"       =>
        setupVCSBackend(dutGenerator, optionsManager)
      case _ =>
        throw new Exception(s"Unrecognized backend name ${testerOptions.backendName}")
    }

    if(optionsManager.topName.isEmpty) {
      optionsManager.setTargetDirName(s"${optionsManager.targetDirName}/${testerGen.getClass.getName}")
    }
    optionsManagerVar.withValue(Some(optionsManager)) {
      backendVar.withValue(Some(backend)) {
        try {
          testerGen(dut).finish
        } catch {
          case e: Throwable =>
            e.printStackTrace()
            backend match {
              case b: VCSBackend => TesterProcess.kill(b)
              case b: VerilatorBackend => TesterProcess.kill(b)
              case _ =>
            }
            throw e
        }
      }
    }
  }

  def execute[T <: Module](args: Array[String], dut: () => T)(
    testerGen: T => PeekPokeTester[T]
  ): Boolean = {
    val optionsManager = new TesterOptionsManager

    optionsManager.parse(args) match {
      case true =>
        execute(dut, optionsManager)(testerGen)
      case _ =>
        false
    }
  }

  /**
    * Start up the interpreter repl with the given circuit
    * To test a `class X extends Module {}`, add the following code to the end
    * of the file that defines
    * @example {{{
    *           object XRepl {
    *             def main(args: Array[String]) {
    *               val optionsManager = new ReplOptionsManager
    *               if(optionsManager.parse(args)) {
    *                 iotesters.Driver.executeFirrtlRepl(() => new X, optionsManager)
    *               }
    *             }
    * }}}
    * running main will place users in the repl with the circuit X loaded into the repl
    *
    * @param dutGenerator   Module to run in interpreter
    * @param optionsManager options
    * @return
    */
  def executeFirrtlRepl[T <: Module](
                                      dutGenerator: () => T,
                                      optionsManager: ReplOptionsManager = new ReplOptionsManager): Boolean = {

    optionsManager.chiselOptions = optionsManager.chiselOptions.copy(runFirrtlCompiler = false)
    optionsManager.firrtlOptions = optionsManager.firrtlOptions.copy(compilerName = "low")

    val chiselResult: ChiselExecutionResult = chisel3.Driver.execute(optionsManager, dutGenerator)
    chiselResult match {
      case ChiselExecutionSucccess(_, emitted, _) =>
        optionsManager.replConfig = ReplConfig(firrtlSource = emitted)
        FirrtlRepl.execute(optionsManager)
        true
      case ChiselExecutionFailure(message) =>
        println("Failed to compile circuit")
        false
    }
  }
  /**
    * This is just here as command line way to see what the options are
    * It will not successfully run
    * TODO: Look into dynamic class loading as way to make this main useful
 *
    * @param args unused args
    */
  def main(args: Array[String]) {
    execute(Array("--help"), null)(null)
  }
  /**
    * Runs the ClassicTester and returns a Boolean indicating test success or failure
    * @@backendType determines whether the ClassicTester uses verilator or the firrtl interpreter to simulate the circuit
    * Will do intermediate compliation steps to setup the backend specified, including cpp compilation for the verilator backend and firrtl IR compilation for the firrlt backend
    */
  def apply[T <: Module](dutGen: () => T, backendType: String = "firrtl")(
      testerGen: T => PeekPokeTester[T]): Boolean = {
    val optionsManager = new TesterOptionsManager

    val (dut, backend) = backendType match {
      case "firrtl"    => setupFirrtlTerpBackend(dutGen, optionsManager)
      case "verilator" => setupVerilatorBackend(dutGen, optionsManager)
      case "vcs"       => setupVCSBackend(dutGen, optionsManager)
      case _ => throw new Exception("Unrecongnized backend type $backendType")
    }
    backendVar.withValue(Some(backend)) {
      try {
        testerGen(dut).finish
      } catch { case e: Throwable =>
        e.printStackTrace()
        backend match {
          case b: VCSBackend => TesterProcess.kill(b)
          case b: VerilatorBackend => TesterProcess.kill(b)
          case _ =>
        }
        throw e
      }
    }
  }

  /**
    * Runs the ClassicTester using the verilator backend without doing Verilator compilation and returns a Boolean indicating success or failure
    * Requires the caller to supply path the already compile Verilator binary
    */
  def run[T <: Module](dutGen: () => T, cmd: Seq[String])
                      (testerGen: T => PeekPokeTester[T]): Boolean = {
    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = getTopModule(circuit).asInstanceOf[T]
    backendVar.withValue(Some(new VerilatorBackend(dut, cmd))) {
      try {
        testerGen(dut).finish
      } catch { case e: Throwable =>
        e.printStackTrace()
        backend match {
          case Some(b: VCSBackend) =>
            TesterProcess kill b
          case Some(b: VerilatorBackend) =>
            TesterProcess kill b
          case _ =>
        }
        throw e
      }
    }
  }

  def run[T <: Module](dutGen: () => T, binary: String, args: String*)
                      (testerGen: T => PeekPokeTester[T]): Boolean =
    run(dutGen, binary +: args.toSeq)(testerGen)

  def run[T <: Module](dutGen: () => T, binary: File, waveform: Option[File] = None)
                      (testerGen: T => PeekPokeTester[T]): Boolean = {
    val args = waveform match {
      case None => Nil
      case Some(f) => Seq(s"+waveform=$f")
    }
    run(dutGen, binary.toString +: args.toSeq)(testerGen)
  }
}

class ReplOptionsManager
  extends ExecutionOptionsManager("chisel-testers")
    with HasInterpreterOptions
    with HasChiselExecutionOptions
    with HasFirrtlOptions
    with HasReplConfig

