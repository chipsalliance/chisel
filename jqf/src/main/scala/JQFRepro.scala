package firrtl.jqf

import collection.JavaConverters._

import java.io.{File, FileNotFoundException, IOException, PrintWriter}
import java.net.MalformedURLException

import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.fuzz.repro.ReproGuidance
import edu.berkeley.cs.jqf.instrument.InstrumentingClassLoader

case class JQFReproOptions(
  classpath: Seq[String] = null,
  testClassName: String = null,
  testMethod: String = null,
  input: File = null,

  logCoverage: Option[File] = None,
  excludes: Seq[String] = Seq.empty,
  includes: Seq[String] = Seq.empty,
  printArgs: Boolean = false
)

object JQFRepro {
  final def main(args: Array[String]): Unit = {
    val parser = new scopt.OptionParser[JQFReproOptions]("JQF-Repro") {
        opt[String]("classpath")
          .required()
          .unbounded()
          .action((x, c) => c.copy(classpath = x.split(":")))
          .text("the classpath to instrument and load the test class from")
        opt[String]("testClassName")
          .required()
          .unbounded()
          .action((x, c) => c.copy(testClassName = x))
          .text("the full class path of the test class")
        opt[String]("testMethod")
          .required()
          .unbounded()
          .action((x, c) => c.copy(testMethod = x))
          .text("the method of the test class to run")
        opt[File]("input")
          .required()
          .unbounded()
          .action((x, c) => c.copy(input = x))
          .text("input file or directory to reproduce test case(s)")

        opt[File]("logCoverage")
          .unbounded()
          .action((x, c) => c.copy(logCoverage = Some(x)))
          .text("output file to dump coverage info")
        opt[Seq[String]]("excludes")
          .unbounded()
          .action((x, c) => c.copy(excludes = x))
          .text("comma-separated list of FQN prefixes to exclude from coverage instrumentation")
        opt[Seq[String]]("includes")
          .unbounded()
          .action((x, c) => c.copy(includes = x))
          .text("comma-separated list of FQN prefixes to forcibly include, even if they match an exclude")
        opt[Unit]("printArgs")
          .unbounded()
          .action((_, c) => c.copy(printArgs = true))
          .text("whether to print the args to each test case")
    }

    parser.parse(args, JQFReproOptions()) match {
      case Some(opts) => execute(opts)
      case _ => System.exit(1)
    }
  }

  def execute(opts: JQFReproOptions): Unit = {
    // Configure classes to instrument
    if (opts.excludes.nonEmpty) {
      System.setProperty("janala.excludes", opts.excludes.mkString(","))
    }
    if (opts.includes.nonEmpty) {
      System.setProperty("janala.includes", opts.includes.mkString(","))
    }

    val loader = try {
      new InstrumentingClassLoader(
        opts.classpath.toArray,
        getClass().getClassLoader())
    } catch  {
      case e: MalformedURLException =>
        throw new JQFException("Could not get project classpath", e)
    }

    // If a coverage dump file was provided, enable logging via system property
    if (opts.logCoverage.isDefined) {
      System.setProperty("jqf.repro.logUniqueBranches", "true")
    }

    // If args should be printed, set system property
    if (opts.printArgs) {
      System.setProperty("jqf.repro.printArgs", "true")
    }

    if (!opts.input.exists() || !opts.input.canRead()) {
      throw new JQFException("Cannot find or open file " + opts.input)
    }


    val (guidance, result) = try {
      val guidance = new ReproGuidance(opts.input, null)
      val result = GuidedFuzzing.run(opts.testClassName, opts.testMethod, loader, guidance, System.out)
      guidance -> result
    } catch {
      case e: ClassNotFoundException => throw new JQFException("Could not load test class", e);
      case e: IllegalArgumentException => throw new JQFException("Bad request", e);
      case e: FileNotFoundException => throw new JQFException("File not found", e);
      case e: RuntimeException => throw new JQFException("Internal error", e);
    }

    // If a coverage dump file was provided, then dump coverage
    if (opts.logCoverage.isDefined) {
      val coverageSet = guidance.getBranchesCovered()
      assert(coverageSet != null) // Should not happen if we set the system property above
      val sortedCoverage = coverageSet.asScala.toSeq.sorted
      try {
        val covOut = new PrintWriter(opts.logCoverage.get)
        sortedCoverage.foreach(covOut.println(_))
      } catch {
        case e: IOException =>
          throw new JQFException("Could not dump coverage info.", e)
      }
    }

    if (!result.wasSuccessful()) {
      throw new JQFException("Test case produces a failure.")
    }
  }
}
