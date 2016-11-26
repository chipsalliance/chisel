// See LICENSE for license details.
package chisel3.iotesters

import scala.collection.mutable.HashMap
import scala.util.Random
import java.io.{File, Writer, FileWriter, PrintStream, IOException}
import java.nio.file.{FileAlreadyExistsException, Files, Paths}
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

/**
  * Copies the necessary header files used for verilator compilation to the specified destination folder
  */
object copyVpiFiles {
  def apply(destinationDirPath: String): Unit = {
    new File(destinationDirPath).mkdirs()
    val simApiHFilePath = Paths.get(destinationDirPath + "/sim_api.h")
    val vpiHFilePath = Paths.get(destinationDirPath + "/vpi.h")
    val vpiCppFilePath = Paths.get(destinationDirPath + "/vpi.cpp")
    val vpiTabFilePath = Paths.get(destinationDirPath + "/vpi.tab")
    try {
      Files.createFile(simApiHFilePath)
      Files.createFile(vpiHFilePath)
      Files.createFile(vpiCppFilePath)
      Files.createFile(vpiTabFilePath)
    } catch {
      case x: FileAlreadyExistsException =>
        System.out.format("")
      case x: IOException => {
        System.err.format("createFile error: %s%n", x)
      }
    }

    Files.copy(getClass.getResourceAsStream("/sim_api.h"), simApiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/vpi.h"), vpiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/vpi.cpp"), vpiCppFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/vpi.tab"), vpiTabFilePath, REPLACE_EXISTING)
  }
}

/**
  * Generates the Module specific verilator harness cpp file for verilator compilation
  */
object genVCSVerilogHarness {
  def apply(dut: chisel3.Module, writer: Writer, vpdFilePath: String, isGateLevel: Boolean = false) {
    val dutName = dut.name
    val (inputs, outputs) = getDataNames("io", dut.io) partition (_._1.dir == chisel3.INPUT)

    writer write "module test;\n"
    writer write "  reg clock = 1;\n"
    writer write "  reg reset = 1;\n"
    val delay = if (isGateLevel) "#0.1" else ""
    inputs foreach { case (node, name) =>
      writer write s"  reg[${node.getWidth-1}:0] $name = 0;\n"
      writer write s"  wire[${node.getWidth-1}:0] ${name}_delay;\n"
      writer write s"  assign $delay ${name}_delay = $name;\n"
    }
    outputs foreach { case (node, name) =>
      writer write s"  wire[${node.getWidth-1}:0] ${name}_delay;\n"
      writer write s"  wire[${node.getWidth-1}:0] $name;\n"
      writer write s"  assign $delay $name = ${name}_delay;\n"
    }

    writer write "  always #`CLOCK_PERIOD clock = ~clock;\n"
    writer write "  reg vcdon = 0;\n"
    writer write "  reg [1023:0] vcdfile = 0;\n"
    writer write "  reg [1023:0] vpdfile = 0;\n"

    writer write "\n  /*** DUT instantiation ***/\n"
    writer write s"  ${dutName} ${dutName}(\n"
    writer write "    .clock(clock),\n"
    writer write "    .reset(reset),\n"
    writer write ((inputs ++ outputs).unzip._2 map (name => s"    .${name}(${name}_delay)") mkString ",\n")
    writer write "  );\n\n"

    writer write "  initial begin\n"
    writer write "    $init_rsts(reset);\n"
    writer write "    $init_ins(%s);\n".format(inputs.unzip._2 mkString ", ")
    writer write "    $init_outs(%s);\n".format(outputs.unzip._2 mkString ", ")
    writer write "    $init_sigs(%s);\n".format(dutName)
    writer write "    /*** VCD & VPD dump ***/\n"
    writer write "    if ($value$plusargs(\"vcdfile=%s\", vcdfile)) begin\n"
    writer write "      $dumpfile(vcdfile);\n"
    writer write "      $dumpvars(0, %s);\n".format(dutName)
    writer write "      $dumpoff;\n"
    writer write "      vcdon = 0;\n"
    writer write "    end\n"
    writer write "    if ($value$plusargs(\"waveform=%s\", vpdfile)) begin\n"
    writer write "      $vcdplusfile(vpdfile);\n"
    writer write "    end else begin\n"
    writer write "      $vcdplusfile(\"%s\");\n".format(vpdFilePath)
    writer write "    end\n"
    writer write "    if ($test$plusargs(\"vpdmem\")) begin\n"
    writer write "      $vcdplusmemon;\n"
    writer write "    end\n"
    writer write "    $vcdpluson(0);\n"
    writer write "  end\n\n"

    writer write "  always @(%s clock) begin\n".format(if (isGateLevel) "posedge" else "negedge")
    writer write "    if (vcdfile && reset) begin\n"
    writer write "      $dumpoff;\n"
    writer write "      vcdon = 0;\n"
    writer write "    end\n"
    writer write "    else if (vcdfile && !vcdon) begin\n"
    writer write "      $dumpon;\n"
    writer write "      vcdon = 1;\n"
    writer write "    end\n"
    writer write "    %s $tick();\n".format(if (isGateLevel) "#0.05" else "")
    writer write "    $vcdplusflush;\n"
    writer write "  end\n\n"
    writer write "endmodule\n"
    writer.close
  }
}

private[iotesters] object setupVCSBackend {
  def apply[T <: chisel3.Module](dutGen: () => T, optionsManager: TesterOptionsManager): (T, Backend) = {
    optionsManager.makeTargetDir()
    val dir = new File(optionsManager.targetDirName)

    val circuit = chisel3.Driver.elaborate(dutGen)
    val dut = getTopModule(circuit).asInstanceOf[T]

    // Generate CHIRRTL
    val chirrtl = firrtl.Parser.parse(chisel3.Driver.emit(dutGen))

    // Generate Verilog
    val verilogFile = new File(dir, s"${circuit.name}.v")
    val verilogWriter = new FileWriter(verilogFile)
    val annotations = firrtl.AnnotationMap(Seq(
      firrtl.passes.memlib.InferReadWriteAnnotation(circuit.name)))
    (new firrtl.VerilogCompiler).compile(
      firrtl.CircuitState(chirrtl, firrtl.ChirrtlForm, Some(annotations)),
      verilogWriter,
      List(new firrtl.passes.memlib.InferReadWrite)
    )
    verilogWriter.close()

    // Generate Harness
    val vcsHarnessFileName = s"${circuit.name}-harness.v"
    val vcsHarnessFile = new File(dir, vcsHarnessFileName)
    val vpdFile = new File(dir, s"${circuit.name}.vpd")
    copyVpiFiles(dir.toString)
    genVCSVerilogHarness(dut, new FileWriter(vcsHarnessFile), vpdFile.toString)
    assert(verilogToVCS(circuit.name, dir, new File(vcsHarnessFileName)).! == 0)

    (dut, new VCSBackend(dut, Seq((new File(dir, circuit.name)).toString)))
  }
}

private[iotesters] class VCSBackend(dut: chisel3.Module,
                                    cmd: Seq[String],
                                    _seed: Long = System.currentTimeMillis)
           extends VerilatorBackend(dut, cmd, _seed)
