// See LICENSE for license details.

package chisel3.tester

import java.io.{File, FileWriter, IOException}
import java.nio.file.{FileAlreadyExistsException, Files, Paths}
import java.nio.file.StandardCopyOption.REPLACE_EXISTING

import chisel3.{assert => chisel3Assert, _}
import chisel3.tester.TesterUtils.getIOPorts
import firrtl.transforms.BlackBoxTargetDirAnno
import firrtl.util.BackendCompilationUtilities._

import scala.sys.process.ProcessBuilder

class VCSProcessBackend[T <: Module](
  val dut: T,
  val cmd: Seq[String],
  val rnd: scala.util.Random
  ) extends InterProcessBackend[T](dut, cmd, rnd) {
}

object VCSTesterBackend {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  /**
    * Copies the necessary header files used for VCS compilation to the specified destination folder
    */
  def copyVpiFiles(destinationDir: File): Unit = {
    val destinationDirPath = destinationDir.toString
    destinationDir.mkdirs()
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

    Files.copy(getClass.getResourceAsStream("/chisel3/tester/sim_api.h"), simApiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/chisel3/tester/vpi.h"), vpiHFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/chisel3/tester/vpi.cpp"), vpiCppFilePath, REPLACE_EXISTING)
    Files.copy(getClass.getResourceAsStream("/chisel3/tester/vpi.tab"), vpiTabFilePath, REPLACE_EXISTING)
  }

  /**
    * Generates the Module specific VCS harness cpp file for verilator compilation
    */
  def generateHarness(dut: chisel3.Module, circuitName: String, dir: File, options: TesterOptionsManager, isGateLevel: Boolean = false): File = {
    val vcsHarnessFileName = s"${circuitName}-harness.v"
    val vcsHarnessFile = new File(dir, vcsHarnessFileName)
    val vcsHarnessWriter = new FileWriter(vcsHarnessFile)
    val vcdFile = new File(dir, s"${circuitName}.vpd")
    val vcdFilePath = vcdFile.toString
    copyVpiFiles(dir)
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
    val emittedStuff = codeBuffer.toString()
    vcsHarnessWriter.append(emittedStuff)
    vcsHarnessWriter.close()
    vcsHarnessFile
  }

  def constructVcsFlags(
                           topModule: String,
                           dir: java.io.File,
                           moreVcsFlags: Seq[String] = Seq.empty[String],
                           moreVcsCFlags: Seq[String] = Seq.empty[String]): Seq[String] = {

    val DefaultCcFlags = Seq("-I$VCS_HOME/include", "-I$dir", "-fPIC", "-std=c++11")
    val ccFlags = DefaultCcFlags ++ moreVcsCFlags

    val blackBoxVerilogList = {
      val list_file = new File(dir, firrtl.transforms.BlackBoxSourceHelper.fileListName)
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
                         vcsToExeCmdEditor: Option[(String) => String] = None
                     ): ProcessBuilder = {

    val vcsFlags = constructVcsFlags(topModule, dir, moreVcsFlags, moreVcsCFlags)

    val cmd = Seq("cd", dir.toString, "&&", "vcs") ++ vcsFlags ++ Seq(
      "-o", topModule, s"$topModule.v", vcsHarness.toString, "vpi.cpp") mkString " "

    val finalCommand = vcsToExeCmdEditor match {
      case Some(f: ((String) => String)) => f(cmd)
      case None => cmd
    }
    println(s"$finalCommand")

    Seq("bash", "-c", finalCommand)
  }

  def verilogToLib(
                         topModule: String,
                         dir: java.io.File,
                         vcsHarness: java.io.File,
                         moreVcsFlags: Seq[String] = Seq.empty[String],
                         moreVcsCFlags: Seq[String] = Seq.empty[String],
                         vcsToExeCmdEditor: Option[(String) => String] = None
                     ): ProcessBuilder = {

    val vcsFlags = constructVcsFlags(topModule, dir, moreVcsFlags, moreVcsCFlags)

    val cmd = Seq("cd", dir.toString, "&&", "vcs") ++ vcsFlags ++ Seq(
      "-o", topModule, s"$topModule.v", vcsHarness.toString, "vpi.cpp") mkString " "

    val finalCommand = vcsToExeCmdEditor match {
      case Some(f: ((String) => String)) => f(cmd)
      case None => cmd
    }
    println(s"$finalCommand")

    Seq("bash", "-c", finalCommand)
  }

  def buildSharedLibrary(circuitName: String, dir: File, cppHarnessFile: File, options: TesterOptionsManager): String = {
//    assert(
//      verilogToLib(circuitName, dir, cppHarnessFile,
//        moreVcsFlags = options.testerOptions.moreVcsFlags,
//        moreVcsCFlags = options.testerOptions.moreVcsCFlags
//      ).! == 0
    throw new RuntimeException("Can't build VCS shared library")
  }

  /** Start an external VCS simulation process.
    *
    * @param dutGen - the curcuit to be simulated
    * @param options - TesterOptionsManager options
    * @param vcsToExeCmdEditor - an optional function taking two argument lists: (TesterOptionsManager)(String) and returning a String
    * @tparam T
    * @return - a BackendInstance suitable for driving the simulation.
    */
  def start[T <: Module](dutGen: => T, options: TesterOptionsManager, vcsToExeCmdEditor: Option[TesterOptionsManager => String => String] = None): BackendInstance[T] = {
    val (result, updatedOptions) = CSimulator.generateVerilog(dutGen, options)
    result match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val dut = getTopModule(circuit).asInstanceOf[T]
        val dir = new File(updatedOptions.targetDirName)
        // Generate Harness
        val vcsHarnessFile = generateHarness(
          dut, circuit.name, dir, updatedOptions
        )

        val cmdEditor = vcsToExeCmdEditor match {
          case Some(f: (TesterOptionsManager => String => String)) => Some(f(updatedOptions))
          case None => None
        }
        assert(
          verilogToVCSExe(circuit.name, dir, vcsHarnessFile,
            moreVcsFlags = updatedOptions.testerOptions.moreVcsFlags,
            moreVcsCFlags = updatedOptions.testerOptions.moreVcsCFlags,
            vcsToExeCmdEditor = cmdEditor
          ).! == 0
        )

        val command = if(updatedOptions.testerOptions.testCmd.nonEmpty) {
          updatedOptions.testerOptions.testCmd
        }
        else {
	  val exe = if (osVersion == OSVersion.Windows) ".exe" else ""
          Seq(new File(dir, s"V${circuit.name}${exe}").toString)
        }
        val seed = updatedOptions.testerOptions.testerSeed
        val rnd = new scala.util.Random(seed)
        new InterProcessBackend(dut, command, rnd)
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
