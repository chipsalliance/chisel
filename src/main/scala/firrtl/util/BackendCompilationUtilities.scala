// See LICENSE for license details.

package firrtl.util

import java.io._
import java.nio.file.Files
import java.text.SimpleDateFormat
import java.util.Calendar

import logger.LazyLogging

import firrtl.FileUtils

import scala.sys.process.{ProcessBuilder, ProcessLogger, _}

trait BackendCompilationUtilities extends LazyLogging {
  /** Parent directory for tests */
  lazy val TestDirectory = new File("test_run_dir")

  def timeStamp: String = {
    val format = new SimpleDateFormat("yyyyMMddHHmmss")
    val now = Calendar.getInstance.getTime
    format.format(now)
  }

  def loggingProcessLogger: ProcessLogger =
    ProcessLogger(logger.info(_), logger.warn(_))

  /**
    * Copy the contents of a resource to a destination file.
    * @param name the name of the resource
    * @param file the file to write it into
    */
  def copyResourceToFile(name: String, file: File): Unit = {
    val in = getClass.getResourceAsStream(name)
    if (in == null) {
      throw new FileNotFoundException(s"Resource '$name'")
    }
    val out = new FileOutputStream(file)
    Iterator.continually(in.read).takeWhile(-1 != _).foreach(out.write)
    out.close()
  }

  /** Create a test directory
    *
    * Will create outer directory called testName then inner directory based on
    * the current time
    */
  def createTestDirectory(testName: String): File = {
    val outer = new File(TestDirectory, testName)
    outer.mkdirs()
    Files.createTempDirectory(outer.toPath, timeStamp).toFile
  }

  def makeHarness(template: String => String, post: String)(f: File): File = {
    val prefix = f.toString.split("/").last
    val vf = new File(f.toString + post)
    val w = new FileWriter(vf)
    w.write(template(prefix))
    w.close()
    vf
  }

  /**
    * compule chirrtl to verilog by using a separate process
    *
    * @param prefix basename of the file
    * @param dir    directory where file lives
    * @return       true if compiler completed successfully
    */
  def firrtlToVerilog(prefix: String, dir: File): ProcessBuilder = {
    Process(
      Seq("firrtl",
        "-i", s"$prefix.fir",
        "-o", s"$prefix.v",
        "-X", "verilog"),
      dir)
  }

  /** Generates a Verilator invocation to convert Verilog sources to C++
    * simulation sources.
    *
    * The Verilator prefix will be V\$dutFile, and running this will generate
    * C++ sources and headers as well as a makefile to compile them.
    *
    * Verilator will automatically locate the top-level module as the one among
    * all the files which are not included elsewhere. If multiple ones exist,
    * the compilation will fail.
    *
    * If the file BlackBoxSourceHelper.fileListName (or an overridden .f resource filename that is
    * specified with the optional resourceFileName parameter) exists in the output directory,
    * it contains a list of source files to be included. Filter out any files in the vSources
    * sequence that are in this file so we don't include the same file multiple times.
    * This complication is an attempt to work-around the fact that clients used to have to
    * explicitly include additional Verilog sources. Now, more of that is automatic.
    *
    * @param dutFile name of the DUT .v without the .v extension
    * @param dir output directory
    * @param vSources list of additional Verilog sources to compile
    * @param cppHarness C++ testharness to compile/link against
    * @param suppressVcd specifies if VCD tracing should be suppressed
    * @param resourceFileName specifies what filename to look for to find a .f file
    */
  def verilogToCpp(
    dutFile: String,
    dir: File,
    vSources: Seq[File],
    cppHarness: File,
    suppressVcd: Boolean = false,
    resourceFileName: String = firrtl.transforms.BlackBoxSourceHelper.defaultFileListName
  ): ProcessBuilder = {

    val topModule = dutFile

    val list_file = new File(dir, resourceFileName)
    val blackBoxVerilogList = {
      if(list_file.exists()) {
        Seq("-f", list_file.getAbsolutePath)
      }
      else {
        Seq.empty[String]
      }
    }

    // Don't include the same file multiple times.
    // If it's in the main .f resource file, don't explicitly include it on the command line.
    // Build a set of canonical file paths to use as a filter to exclude already included additional Verilog sources.
    val blackBoxHelperFiles: Set[String] = {
      if(list_file.exists()) {
        FileUtils.getLines(list_file).toSet
      }
      else {
        Set.empty
      }
    }
    val vSourcesFiltered = vSources.filterNot(f => blackBoxHelperFiles.contains(f.getCanonicalPath))
    val command = Seq(
      "verilator",
      "--cc", s"${dir.getAbsolutePath}/$dutFile.v"
    ) ++
      blackBoxVerilogList ++
      vSourcesFiltered.flatMap(file => Seq("-v", file.getCanonicalPath)) ++
      Seq("--assert",
        "-Wno-fatal",
        "-Wno-WIDTH",
        "-Wno-STMTDLY"
      ) ++
      { if(suppressVcd) { Seq.empty } else { Seq("--trace")} } ++
      Seq(
        "-O1",
        "--top-module", topModule,
        "+define+TOP_TYPE=V" + dutFile,
        s"+define+PRINTF_COND=!$topModule.reset",
        s"+define+STOP_COND=!$topModule.reset",
        "-CFLAGS",
        s"""-Wno-undefined-bool-conversion -O1 -DTOP_TYPE=V$dutFile -DVL_USER_FINISH -include V$dutFile.h""",
        "-Mdir", dir.getAbsolutePath,
        "--exe", cppHarness.getAbsolutePath)
    logger.info(s"${command.mkString(" ")}") // scalastyle:ignore regex
    command
  }

  def cppToExe(prefix: String, dir: File): ProcessBuilder =
    Seq("make", "-C", dir.toString, "-j", "-f", s"V$prefix.mk", s"V$prefix")

  def executeExpectingFailure(
                               prefix: String,
                               dir: File,
                               assertionMsg: String = ""): Boolean = {
    var triggered = false
    val assertionMessageSupplied = assertionMsg != ""
    val e = Process(s"./V$prefix", dir) !
      ProcessLogger(line => {
        triggered = triggered || (assertionMessageSupplied && line.contains(assertionMsg))
        logger.info(line) // scalastyle:ignore regex
      },
      logger.warn(_))
    // Fail if a line contained an assertion or if we get a non-zero exit code
    //  or, we get a SIGABRT (assertion failure) and we didn't provide a specific assertion message
    triggered || (e != 0 && (e != 134 || !assertionMessageSupplied))
  }

  def executeExpectingSuccess(prefix: String, dir: File): Boolean = {
    !executeExpectingFailure(prefix, dir)
  }

  /** Creates and runs a Yosys script that creates and runs SAT on a miter
    * circuit. Returns true if SAT succeeds, false otherwise
    *
    * The custom and reference Verilog files must not contain any modules with
    * the same name otherwise Yosys will not be able to create a miter circuit
    *
    * @param customTop    name of the DUT with custom transforms without the .v
    *                     extension
    * @param referenceTop name of the DUT without custom transforms without the
    *                     .v extension
    * @param testDir      directory containing verilog files
    * @param resets       signals to set for SAT, format is
    *                     (timestep, signal, value)
    */
  def yosysExpectSuccess(customTop: String,
                         referenceTop: String,
                         testDir: File,
                         resets: Seq[(Int, String, Int)] = Seq.empty): Boolean = {
    !yosysExpectFailure(customTop, referenceTop, testDir, resets)
  }

  /** Creates and runs a Yosys script that creates and runs SAT on a miter
    * circuit. Returns false if SAT succeeds, true otherwise
    *
    * The custom and reference Verilog files must not contain any modules with
    * the same name otherwise Yosys will not be able to create a miter circuit
    *
    * @param customTop    name of the DUT with custom transforms without the .v
    *                     extension
    * @param referenceTop name of the DUT without custom transforms without the
    *                     .v extension
    * @param testDir      directory containing verilog files
    * @param resets       signals to set for SAT, format is
    *                     (timestep, signal, value)
    */
  def yosysExpectFailure(customTop: String,
                         referenceTop: String,
                         testDir: File,
                         resets: Seq[(Int, String, Int)] = Seq.empty): Boolean = {

    val setSignals = resets.map(_._2).toSet[String].map(s => s"-set in_$s 0").mkString(" ")
    val setAtSignals = resets.map {
      case (timestep, signal, value) => s"-set-at $timestep in_$signal $value"
    }.mkString(" ")
    val scriptFileName = s"${testDir.getAbsolutePath}/yosys_script"
    val yosysScriptWriter = new PrintWriter(scriptFileName)
    yosysScriptWriter.write(
      s"""read_verilog ${testDir.getAbsolutePath}/$customTop.v
         |read_verilog ${testDir.getAbsolutePath}/$referenceTop.v
         |prep; proc; opt; memory
         |miter -equiv -flatten $customTop $referenceTop miter
         |hierarchy -top miter
         |sat -verify -tempinduct -prove trigger 0 $setSignals $setAtSignals -seq 1 miter"""
        .stripMargin)
    yosysScriptWriter.close()

    val resultFileName = testDir.getAbsolutePath + "/yosys_results"
    val command = s"yosys -s $scriptFileName" #> new File(resultFileName)
    command.! != 0
  }
}
