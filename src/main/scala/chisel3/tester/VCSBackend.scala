// See LICENSE for license details.

package chisel3.tester

import java.io.{File, FileWriter, IOException}
import java.nio.file.{FileAlreadyExistsException, Files, Paths}
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

import chisel3.{assert => chisel3Assert, _}
import chisel3.tester.TesterUtils.getIOPorts
import firrtl.transforms.BlackBoxTargetDirAnno

import scala.sys.process.ProcessBuilder

class VCSProcessBackend[T <: Module](
  dut: T,
  cmd: Seq[String],
  rnd: scala.util.Random
  ) extends InterProcessBackend[T](dut, cmd, rnd) {
}

object VCSTesterShim {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  /**
    * Copies the necessary header files used for VCS compilation to the specified destination folder
    */
  def copyVpiFiles(destinationDirPath: String): Unit = {
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
      case _: FileAlreadyExistsException =>
        System.out.format("")
      case x: IOException =>
        System.err.format("createFile error: %s%n", x)
    }

    Files.copy(getClass.getResourceAsStream("/sim_api.h"), simApiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/vpi.h"), vpiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/vpi.cpp"), vpiCppFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/vpi.tab"), vpiTabFilePath, REPLACE_EXISTING)
  }

  /**
    * Generates the Module specific VCS harness cpp file for verilator compilation
    */
  def generateHarness(dut: chisel3.Module, vcdFilePath: String, isGateLevel: Boolean = false): String = {
    val codeBuffer = new StringBuilder
    val dutName = dut.name
    val (inputs, outputs) = getIOPorts(dut)

    codeBuffer.append("module test;\n")
    codeBuffer.append("  reg clock = 1;\n")
    codeBuffer.append("  reg reset = 1;\n")
    val delay = if (isGateLevel) "#0.1" else ""
    inputs foreach { case (node, name) =>
      codeBuffer.append(s"  reg[${node.getWidth-1}:0] $name = 0;\n")
      codeBuffer.append(s"  wire[${node.getWidth-1}:0] ${name}_delay;\n")
      codeBuffer.append(s"  assign $delay ${name}_delay = $name;\n")
    }
    outputs foreach { case (node, name) =>
      codeBuffer.append(s"  wire[${node.getWidth-1}:0] ${name}_delay;\n")
      codeBuffer.append(s"  wire[${node.getWidth-1}:0] $name;\n")
      codeBuffer.append(s"  assign $delay $name = ${name}_delay;\n")
    }

    codeBuffer.append("  always #`CLOCK_PERIOD clock = ~clock;\n")
    codeBuffer.append("  reg vcdon = 0;\n")
    codeBuffer.append("  reg [1023:0] vcdfile = 0;\n")
    codeBuffer.append("  reg [1023:0] vpdfile = 0;\n")

    codeBuffer.append("\n  /*** DUT instantiation ***/\n")
    codeBuffer.append(s"  $dutName $dutName(\n")
    codeBuffer.append("    .clock(clock),\n")
    codeBuffer.append("    .reset(reset),\n")
    codeBuffer.append(((inputs ++ outputs).unzip._2 map (name => s"    .$name(${name}_delay)") mkString ",\n"))
    codeBuffer.append("  );\n\n")

    codeBuffer.append("  initial begin\n")
    codeBuffer.append("    $init_rsts(reset);\n")
    codeBuffer.append("    $init_ins(%s);\n".format(inputs.unzip._2 mkString ", "))
    codeBuffer.append("    $init_outs(%s);\n".format(outputs.unzip._2 mkString ", "))
    codeBuffer.append("    $init_sigs(%s);\n".format(dutName))
    codeBuffer.append("    /*** VCD & VPD dump ***/\n")
    codeBuffer.append("    if ($value$plusargs(\"vcdfile=%s\", vcdfile)) begin\n")
    codeBuffer.append("      $dumpfile(vcdfile);\n")
    codeBuffer.append("      $dumpvars(0, %s);\n".format(dutName))
    codeBuffer.append("      $dumpoff;\n")
    codeBuffer.append("      vcdon = 0;\n")
    codeBuffer.append("    end\n")
    codeBuffer.append("    if ($value$plusargs(\"waveform=%s\", vpdfile)) begin\n")
    codeBuffer.append("      $vcdplusfile(vpdfile);\n")
    codeBuffer.append("    end else begin\n")
    codeBuffer.append("      $vcdplusfile(\"%s\");\n".format(vcdFilePath))
    codeBuffer.append("    end\n")
    codeBuffer.append("    if ($test$plusargs(\"vpdmem\")) begin\n")
    codeBuffer.append("      $vcdplusmemon;\n")
    codeBuffer.append("    end\n")
    codeBuffer.append("    $vcdpluson(0);\n")
    codeBuffer.append("  end\n\n")

    codeBuffer.append("  always @(%s clock) begin\n".format(if (isGateLevel) "posedge" else "negedge"))
    codeBuffer.append("    if (vcdfile && reset) begin\n")
    codeBuffer.append("      $dumpoff;\n")
    codeBuffer.append("      vcdon = 0;\n")
    codeBuffer.append("    end\n")
    codeBuffer.append("    else if (vcdfile && !vcdon) begin\n")
    codeBuffer.append("      $dumpon;\n")
    codeBuffer.append("      vcdon = 1;\n")
    codeBuffer.append("    end\n")
    codeBuffer.append("    %s $tick();\n".format(if (isGateLevel) "#0.05" else ""))
    codeBuffer.append("    $vcdplusflush;\n")
    codeBuffer.append("  end\n\n")
    codeBuffer.append("endmodule\n")
    codeBuffer.toString()
  }

  def constructVcsFlags(
                           topModule: String,
                           dir: java.io.File,
                           moreVcsFlags: Seq[String] = Seq.empty[String],
                           moreVcsCFlags: Seq[String] = Seq.empty[String]): Seq[String] = {

    val DefaultCcFlags = Seq("-I$VCS_HOME/include", "-I$dir", "-fPIC", "-std=c++11")
    val ccFlags = DefaultCcFlags ++ moreVcsCFlags

    val blackBoxVerilogList = {
      val list_file = new File(dir, firrtl.transforms.BlackBoxSourceHelper.FileListName)
      if(list_file.exists()) {
        Seq("-f", list_file.getAbsolutePath)
      }
      else {
        Seq.empty[String]
      }
    }

    val vcsFlags = Seq("-full64",
      "-quiet",
      "-timescale=1ns/1ps",
      "-debug_pp",
      s"-Mdir=$topModule.csrc",
      "+v2k", "+vpi",
      "+vcs+lic+wait",
      "+vcs+initreg+random",
      "+define+CLOCK_PERIOD=1",
      "-P", "vpi.tab",
      "-cpp", "g++", "-O2", "-LDFLAGS", "-lstdc++",
      "-CFLAGS", "\"%s\"".format(ccFlags mkString " ")) ++
      moreVcsFlags ++
      blackBoxVerilogList

    vcsFlags
  }

  def verilogToVCSExe(
                         topModule: String,
                         dir: java.io.File,
                         vcsHarness: java.io.File,
                         moreVcsFlags: Seq[String] = Seq.empty[String],
                         moreVcsCFlags: Seq[String] = Seq.empty[String],
                         editCommands: String = ""): ProcessBuilder = {

    val vcsFlags = constructVcsFlags(topModule, dir, moreVcsFlags, moreVcsCFlags)

    val cmd = Seq("cd", dir.toString, "&&", "vcs") ++ vcsFlags ++ Seq(
      "-o", topModule, s"$topModule.v", vcsHarness.toString, "vpi.cpp") mkString " "

    val commandEditor = CommandEditor(editCommands, "vcs-command-edit")
    val finalCommand = commandEditor(cmd)
    println(s"$finalCommand")

    Seq("bash", "-c", finalCommand)
  }

  def start[T <: Module](dutGen: => T, options: TesterOptionsManager): BackendInstance[T] = {
    val optionsManager = options
    // We need to intercept the CHIRRTL output and tweak it.
    optionsManager.chiselOptions = optionsManager.chiselOptions.copy(runFirrtlCompiler = false)
    optionsManager.makeTargetDir()
    val dir = new File(optionsManager.targetDirName)

    chisel3.Driver.execute(optionsManager, () => dutGen) match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val chirrtl = firrtl.Parser.parse(emitted)
        val dut = getTopModule(circuit).asInstanceOf[T]

        // This makes sure annotations for command line options get created
        firrtl.Driver.loadAnnotations(optionsManager)

        /*
        The following block adds an annotation that tells the black box helper where the
        current build directory is, so that it can copy verilog resource files into the right place
         */
        val annotations = optionsManager.firrtlOptions.annotations ++
          List(BlackBoxTargetDirAnno(optionsManager.targetDirName))

        val transforms = optionsManager.firrtlOptions.customTransforms

        VerilatorTesterShim.copyHeaderFiles(optionsManager.targetDirName)

        // Generate Verilog
        val verilogFile = new File(dir, s"${circuit.name}.v")
        val verilogWriter = new FileWriter(verilogFile)

        val compileResult = (new firrtl.VerilogCompiler).compileAndEmit(
          CircuitState(chirrtl, ChirrtlForm, annotations),
          customTransforms = transforms
        )
        val compiledStuff = compileResult.getEmittedCircuit
        verilogWriter.write(compiledStuff.value)
        verilogWriter.close()

        // Generate Harness
        val vcsHarnessFileName = s"${circuit.name}-harness.v"
        val vcsHarnessFile = new File(dir, vcsHarnessFileName)
        val vpdFile = new File(dir, s"${circuit.name}.vpd")
        copyVpiFiles(dir.toString)
        generateHarness(dut, vpdFile.toString)

        assert(
          verilogToVCSExe(circuit.name, dir, new File(vcsHarnessFileName),
            moreVcsFlags = optionsManager.testerOptions.moreVcsFlags,
            moreVcsCFlags = optionsManager.testerOptions.moreVcsCFlags,
            editCommands = optionsManager.testerOptions.vcsCommandEdits
          ).! == 0
        )

        val command = if(optionsManager.testerOptions.testCmd.nonEmpty) {
          optionsManager.testerOptions.testCmd
        }
        else {
          Seq(new File(dir, s"V${circuit.name}").toString)
        }
        val seed = optionsManager.testerOptions.testerSeed
        val rnd = new scala.util.Random(seed)
        new InterProcessBackend(dut, command, rnd)
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
