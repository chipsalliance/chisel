package chisel3

import chisel3.test.DiscoverUnitTests
import circt.stage.ChiselStage
import java.io.File
import java.io.PrintStream
import scala.util.matching.Regex
import scopt.OptionParser

/** Utility to discover and generate all unit tests in the classpath. */
object UnitTests {

  /** Command line configuration options. */
  private case class Config(
    runpath:    List[String] = List(),
    outputFile: Option[File] = None,
    list:       Boolean = false,
    verbose:    Boolean = false,
    filters:    List[Regex] = List(),
    excludes:   List[Regex] = List()
  )

  def main(args: Array[String]): Unit = {
    var shouldExit = false
    val parser = new OptionParser[Config]("chisel3.UnitTests") {
      head("Chisel Unit Test Utility")
      help("help").abbr("h")

      opt[Seq[String]]('R', "runpath")
        .text("Where test classes are discovered and loaded from")
        .unbounded()
        .action((x, c) => c.copy(runpath = c.runpath ++ x))

      opt[File]('o', "output")
        .text("Output file name (\"-\" for stdout)")
        .action((x, c) => c.copy(outputFile = if (!x.getPath.isEmpty && x.getPath != "-") Some(x) else None))

      opt[Unit]('l', "list")
        .text("List tests instead of building them")
        .action((_, c) => c.copy(list = true))

      opt[Unit]('v', "verbose")
        .text("Print verbose information to stderr")
        .action((_, c) => c.copy(verbose = true))

      opt[Seq[String]]('f', "filter")
        .text("Only consider tests which match at least one filter regex")
        .unbounded()
        .action((x, c) => c.copy(filters = c.filters ++ x.map(_.r)))

      opt[Seq[String]]('x', "exclude")
        .text("Ignore tests which match at least one exclusion regex")
        .unbounded()
        .action((x, c) => c.copy(excludes = c.excludes ++ x.map(_.r)))

      // Do not `sys.exit` on `help` to facilitate testing.
      override def terminate(exitState: Either[String, Unit]): Unit = {
        shouldExit = true
      }
    }

    // Parse the command line options.
    val config = parser.parse(args, Config()) match {
      case Some(config) => config
      case None         => return
    }
    if (shouldExit)
      return

    // Define the handler that will be called for each discovered unit test, and
    // which will decide whether the test is generated or not.
    def handler(className: String, gen: () => Unit): Unit = {
      // If none of the inclusion filters match, skip this test.
      if (!config.filters.isEmpty && !config.filters.exists(_.findFirstMatchIn(className).isDefined)) {
        Console.err.println(f"Skipping ${className} (does not match filter)")
        return
      }

      // If any of the exclusion filters match, skip this test.
      if (config.excludes.exists(_.findFirstMatchIn(className).isDefined)) {
        Console.err.println(f"Skipping ${className} (matches exclude filter)")
        return
      }

      // If we are just listing tests, print the class name and skip this test.
      if (config.list) {
        println(className)
        return
      }

      // Otherwise generate the test.
      if (config.verbose)
        Console.err.println(f"Building ${className}")
      gen()
    }

    // If the user only asked for a list of tests, run test discovery without
    // setting up any of the Chisel builder stuff in the background. The handler
    // will never actually call the Chisel generators in this mode.
    if (config.list) {
      DiscoverUnitTests(handler, config.runpath)
      return
    }

    // Generate the unit tests.
    class AllUnitTests extends RawModule {
      DiscoverUnitTests(handler, config.runpath)
    }
    val chirrtl = ChiselStage.emitCHIRRTL(new AllUnitTests)

    // Write the result to the output.
    val output: PrintStream = config.outputFile match {
      case Some(file) => new PrintStream(file)
      case None       => Console.out
    }
    try {
      output.print(chirrtl)
    } finally {
      output.close()
    }
  }
}
