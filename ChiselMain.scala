// See LICENSE for license details.

package chisel3.iotesters

import chisel3._

import scala.collection.mutable.{ArrayBuffer}
import scala.util.{DynamicVariable}
import java.nio.file.{FileAlreadyExistsException, Files, Paths}
import java.io.{File, FileWriter, IOException}

private[iotesters] class TesterContext {
  var isGenVerilog = false
  var isGenHarness = false
  var isCompiling = false
  var isRunTest = false
  var isGateLevel = false
  var testerSeed = System.currentTimeMillis
  val testCmd = ArrayBuffer[String]()
  var backend = "verilator"
  var targetDir = new File("test_run_dir")
  var logFile: Option[String] = None
  var waveform: Option[String] = None
  val graph = new CircuitGraph
}

object chiselMain {
  private val contextVar = new DynamicVariable[Option[TesterContext]](None)
  private[iotesters] def context = contextVar.value getOrElse (new TesterContext)

  private def parseArgs(args: List[String]): Unit = args match {
    case "--firrtl" :: tail => context.backend = "firrtl" ; parseArgs(tail)
    case "--verilator" :: tail => context.backend = "verilator" ; parseArgs(tail)
    case "--vcs" :: tail => context.backend = "vcs" ; parseArgs(tail)
    case "--v" :: tail  => context.isGenVerilog = true ; parseArgs(tail)
    case "--backend" :: value :: tail => context.backend = value ; parseArgs(tail)
    case "--genHarness" :: tail => context.isGenHarness = true ; parseArgs(tail)
    case "--compile" :: tail => context.isCompiling = true ; parseArgs(tail)
    case "--test" :: tail => context.isRunTest = true ; parseArgs(tail)
    case "--testCommand" :: value :: tail => context.testCmd ++= value split ' ' ; parseArgs(tail)
    case "--testerSeed" :: value :: tail => context.testerSeed = value.toLong ; parseArgs(tail)
    case "--targetDir" :: value :: tail => context.targetDir = new File(value) ; parseArgs(tail)
    case "--gateLevel" :: tail => context.isGateLevel = false ; parseArgs(tail)
    case "--logFile" :: value :: tail => context.logFile = Some(value) ; parseArgs(tail)
    case "--waveform" :: value :: tail => context.waveform = Some(value) ; parseArgs(tail)
    case flag :: tail => parseArgs(tail) // skip unknown flag
    case Nil => // finish
  }

  private def genHarness[T <: Module](dut: Module, graph: CircuitGraph, chirrtl: firrtl.ir.Circuit) {
    val dir = context.targetDir
    context.backend match {
      case "firrtl" => // skip
      case "verilator" =>
        val harness = new FileWriter(new File(dir, s"${chirrtl.main}-harness.cpp"))
        val waveform = (new File(dir, s"${chirrtl.main}.vcd")).toString
        val annotation = new firrtl.Annotations.AnnotationMap(Nil)
        (new VerilatorCppHarnessCompiler(dut, graph, waveform)).compile(chirrtl, annotation, harness)
        harness.close
      case "vcs" =>
        val harness = new FileWriter(new File(dir, s"${chirrtl.main}-harness.v"))
        val waveform = (new File(dir, s"${chirrtl.main}.vpd")).toString
        genVCSVerilogHarness(dut, harness, waveform.toString, context.isGateLevel)
      case b => throw BackendException(b)
    }
  }

  private def compile(dutName: String) {
    val dir = context.targetDir
    context.backend match {
      case "firrtl" => // skip
      case "verilator" =>
        // Copy API files
        copyVerilatorHeaderFiles(context.targetDir.toString)
        // Generate Verilator
        chisel3.Driver.verilogToCpp(dutName, dutName, dir, Seq(), new File(s"$dutName-harness.cpp")).!
        // Compile Verilator
        chisel3.Driver.cppToExe(dutName, dir).!
      case "vcs" =>
        // Copy API files
        copyVpiFiles(context.targetDir.toString)
        // Compile VCS
        verilogToVCS(dutName, dir, new File(s"$dutName-harness.v")).!
      case b => throw BackendException(b)
    }
  }

  private def elaborate[T <: Module](args: Array[String], dutGen: () => T): T = {
    parseArgs(args.toList)
    try {
      Files.createDirectory(Paths.get(context.targetDir.toString))
    } catch {
      case x: FileAlreadyExistsException =>
      case x: IOException =>
        System.err.format("createFile error: %s%n", x)
    }
    val graph = context.graph
    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = (graph construct circuit).asInstanceOf[T]
    val dir = context.targetDir
    val name = circuit.name

    val chirrtl = firrtl.Parser.parse(chisel3.Driver.emit(dutGen))
    val chirrtlFile = new File(dir, s"${name}.ir")
    val verilogFile = new File(dir, s"${name}.v")
    if (context.backend == "firrtl") {
       val writer = new FileWriter(chirrtlFile)
      firrtl.FIRRTLEmitter run (chirrtl, writer)
      writer.close
    } else if (context.isGenVerilog) {
      val annotation = new firrtl.Annotations.AnnotationMap(Nil)
      val writer = new FileWriter(verilogFile)
      new firrtl.VerilogCompiler compile (chirrtl, annotation, writer)
      writer.close
    } 

    if (context.isGenHarness) genHarness(dut, graph, chirrtl)

    if (context.isCompiling) compile(name)

    if (context.testCmd.isEmpty) {
      context.backend match {
        case "firrtl" => // skip
        case "verilator" =>
          context.testCmd += (new File(context.targetDir, s"V$name")).toString
        case "vcs" =>
          context.testCmd += (new File(context.targetDir, name)).toString
        case b => throw BackendException(b)
      }
    }
    dut
  }

  def apply[T <: Module](args: Array[String], dutGen: () => T): T = {
    val ctx = Some(new TesterContext)
    val dut = contextVar.withValue(ctx) {
      elaborate(args, dutGen)
    }
    contextVar.value = ctx // TODO: is it ok?
    dut
  }

  def apply[T <: Module](args: Array[String], dutGen: () => T, testerGen: T => PeekPokeTester[T]) = {
    contextVar.withValue(Some(new TesterContext)) {
      val dut = elaborate(args, dutGen)
      if (context.isRunTest) {
        assert(try {
          testerGen(dut).finish
        } catch { case e: Throwable =>
          TesterProcess.killall
          false
        }, "Test failed")
      }
      dut
    }
  }
}

object chiselMainTest {
  def apply[T <: Module](args: Array[String], dutGen: () => T)(testerGen: T => PeekPokeTester[T]) = {
    chiselMain(args, dutGen, testerGen)
  }
}
