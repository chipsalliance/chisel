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
  var testerSeed = System.currentTimeMillis
  val testCmd = ArrayBuffer[String]()
  var backendType = "verilator"
  var backend: Option[Backend] = None
  var targetDir = new File("test_run_dir")
  var logFile: Option[File] = None
  var waveform: Option[File] = None
}

object chiselMain {
  private val contextVar = new DynamicVariable[Option[TesterContext]](None)
  private[iotesters] def context = contextVar.value getOrElse (new TesterContext)

  private def parseArgs(args: List[String]): Unit = args match {
    case "--firrtl" :: tail => context.backendType = "firrtl" ; parseArgs(tail)
    case "--verilator" :: tail => context.backendType = "verilator" ; parseArgs(tail)
    case "--vcs" :: tail => context.backendType = "vcs" ; parseArgs(tail)
    case "--glsim" :: tail => context.backendType = "glsim" ; parseArgs(tail)
    case "--v" :: tail  => context.isGenVerilog = true ; parseArgs(tail)
    case "--backend" :: value :: tail => context.backendType = value ; parseArgs(tail)
    case "--genHarness" :: tail => context.isGenHarness = true ; parseArgs(tail)
    case "--compile" :: tail => context.isCompiling = true ; parseArgs(tail)
    case "--test" :: tail => context.isRunTest = true ; parseArgs(tail)
    case "--testCommand" :: value :: tail => context.testCmd ++= value split ' ' ; parseArgs(tail)
    case "--testerSeed" :: value :: tail => context.testerSeed = value.toLong ; parseArgs(tail)
    case "--targetDir" :: value :: tail => context.targetDir = new File(value) ; parseArgs(tail)
    case "--logFile" :: value :: tail => context.logFile = Some(new File(value)) ; parseArgs(tail)
    case "--waveform" :: value :: tail => context.waveform = Some(new File(value)) ; parseArgs(tail)
    case flag :: tail => parseArgs(tail) // skip unknown flag
    case Nil => // finish
  }

  private def genHarness[T <: Module](dut: Module, nodes: Seq[internal.InstanceId], chirrtl: firrtl.ir.Circuit) {
    import firrtl.{ChirrtlForm, CircuitState}
    val dir = context.targetDir
    context.backendType match {
      case "firrtl" => // skip
      case "verilator" =>
        val harness = new FileWriter(new File(dir, s"${chirrtl.main}-harness.cpp"))
        val waveform = (new File(dir, s"${chirrtl.main}.vcd")).toString
        (new VerilatorCppHarnessCompiler(dut, nodes, waveform)).compile(CircuitState(chirrtl, ChirrtlForm), harness)
        harness.close()
      case "vcs" | "glsim" =>
        val harness = new FileWriter(new File(dir, s"${chirrtl.main}-harness.v"))
        val waveform = (new File(dir, s"${chirrtl.main}.vpd")).toString
        genVCSVerilogHarness(dut, harness, waveform.toString, context.backendType == "glsim")
      case b => throw BackendException(b)
    }
  }

  private def compile(dutName: String) {
    val dir = context.targetDir
    context.backendType match {
      case "firrtl" => // skip
      case "verilator" =>
        // Copy API files
        copyVerilatorHeaderFiles(context.targetDir.toString)
        // Generate Verilator
        assert(chisel3.Driver.verilogToCpp(
          dutName,
          dir,
          Seq(),
          new File(dir, s"$dutName-harness.cpp")).! == 0)
        // Compile Verilator
        assert(chisel3.Driver.cppToExe(dutName, dir).! == 0)
      case "vcs" | "glsim" =>
        // Copy API files
        copyVpiFiles(context.targetDir.toString)
        // Compile VCS
        assert(verilogToVCS(dutName, dir, new File(s"$dutName-harness.v")).! == 0)
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
    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = getTopModule(circuit).asInstanceOf[T]
    val nodes = getChiselNodes(circuit)
    val dir = context.targetDir
    val name = circuit.name

    val chirrtl = firrtl.Parser.parse(chisel3.Driver.emit(circuit))
    val chirrtlFile = new File(dir, s"${name}.ir")
    val verilogFile = new File(dir, s"${name}.v")
    context.backendType match {
      case "firrtl" =>
        val writer = new FileWriter(chirrtlFile)
        (new firrtl.FirrtlEmitter).emit(firrtl.CircuitState(chirrtl, firrtl.ChirrtlForm), writer)
        writer.close()
      case _ if (context.isGenVerilog) =>
        val annotations = firrtl.AnnotationMap(Seq(
          firrtl.passes.memlib.InferReadWriteAnnotation(name)))
        val writer = new FileWriter(verilogFile)
        (new firrtl.VerilogCompiler).compile(
          firrtl.CircuitState(chirrtl, firrtl.ChirrtlForm, Some(annotations)),
          writer,
          List(new firrtl.passes.memlib.InferReadWrite)
        )
        writer.close
      case _ =>
    } 

    if (context.isGenHarness) genHarness(dut, nodes, chirrtl)

    if (context.isCompiling) compile(name)

    dut
  }

  private def setupBackend[T <: Module](dut: T) {
    val name = dut.name

    if (context.testCmd.isEmpty) {
      context.backendType match {
        case "firrtl" => // skip
        case "verilator" =>
          context.testCmd += (new File(context.targetDir, s"V$name")).toString
        case "vcs" | "glsim" =>
          context.testCmd += (new File(context.targetDir, name)).toString
        case b => throw BackendException(b)
      }
    }

    context.waveform match {
      case None =>
      case Some(f) => context.testCmd += s"+waveform=$f"
    }

    context.backend = Some(context.backendType match {
      case "firrtl" =>
        val file = new java.io.File(context.targetDir, s"${dut.name}.ir")
        val ir = io.Source.fromFile(file).getLines mkString "\n"
        new FirrtlTerpBackend(dut, ir)
      case "verilator" =>
        new VerilatorBackend(dut, context.testCmd.toList, context.testerSeed)
      case "vcs" | "glsim" =>
        new VCSBackend(dut, context.testCmd.toList, context.testerSeed)
      case b => throw BackendException(b)
    })
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
        setupBackend(dut)
        assert(try {
          testerGen(dut).finish
        } catch { case e: Throwable =>
          e.printStackTrace
          context.backend match {
            case Some(b: VCSBackend) =>
              TesterProcess kill b
            case Some(b: VerilatorBackend) =>
              TesterProcess kill b
            case _ =>
          }
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
