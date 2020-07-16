package firrtl.jqf

import java.io.{File, FileNotFoundException, IOException}
import java.net.{MalformedURLException, URLClassLoader}
import java.time.Duration
import java.time.format.DateTimeParseException

import edu.berkeley.cs.jqf.fuzz.ei.ExecutionIndexingGuidance
import edu.berkeley.cs.jqf.fuzz.ei.ZestGuidance
import edu.berkeley.cs.jqf.fuzz.junit.GuidedFuzzing
import edu.berkeley.cs.jqf.instrument.InstrumentingClassLoader


case class JQFException(message: String, e: Throwable = null) extends Exception(message)

sealed trait JQFEngine
case object Zeal extends JQFEngine
case object Zest extends JQFEngine

case class JQFFuzzOptions(
  // required
  classpath: Seq[String] = null,
  outputDirectory: File = null,
  testClassName: String = null,
  testMethod: String = null,

  excludes: Seq[String] = Seq.empty,
  includes: Seq[String] = Seq.empty,
  time: Option[String] = None,
  blind: Boolean = false,
  engine: JQFEngine = Zest,
  disableCoverage: Boolean = false,
  inputDirectory: Option[File] = None,
  saveAll: Boolean = false,
  libFuzzerCompatOutput: Boolean = false,
  quiet: Boolean = false,
  exitOnCrash: Boolean = false,
  runTimeout: Option[Int] = None
)

object JQFFuzz {
  final def main(args: Array[String]): Unit = {
    val parser = new scopt.OptionParser[JQFFuzzOptions]("JQF-Fuzz") {
      opt[String]("classpath")
        .required()
        .unbounded()
        .action((x, c) => c.copy(classpath = x.split(":")))
        .text("the classpath to instrument and load the test class from")
      opt[File]("outputDirectory")
        .required()
        .unbounded()
        .action((x, c) => c.copy(outputDirectory = x))
        .text("the directory to output test results")
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

      opt[Seq[String]]("excludes")
        .unbounded()
        .action((x, c) => c.copy(excludes = x))
        .text("comma-separated list of FQN prefixes to exclude from coverage instrumentation")
      opt[Seq[String]]("includes")
        .unbounded()
        .action((x, c) => c.copy(includes = x))
        .text("comma-separated list of FQN prefixes to forcibly include, even if they match an exclude")
      opt[String]("time")
        .unbounded()
        .action((x, c) => c.copy(time = Some(x)))
        .text("the duration of time for which to run fuzzing")
      opt[Unit]("blind")
        .unbounded()
        .action((_, c) => c.copy(blind = true))
        .text("whether to generate inputs blindly without taking into account coverage feedback")
      opt[String]("engine")
        .unbounded()
        .action((x, c) => x match {
          case "zest" => c.copy(engine = Zest)
          case "zeal" => c.copy(engine = Zeal)
          case _ =>
            throw new JQFException(s"bad a value '$x' for --engine, must be zest|zeal")
        })
        .text("the fuzzing engine, valid choices are zest|zeal")
      opt[Unit]("disableCoverage")
        .unbounded()
        .action((_, c) => c.copy(disableCoverage = true))
        .text("disable code-coverage instrumentation")
      opt[File]("inputDirectory")
        .unbounded()
        .action((x, c) => c.copy(inputDirectory = Some(x)))
        .text("the name of the input directory containing seed files")
      opt[Unit]("saveAll")
        .unbounded()
        .action((_, c) => c.copy(saveAll = true))
        .text("save ALL inputs generated during fuzzing, even the ones that do not have any unique code coverage")
      opt[Unit]("libFuzzerCompatOutput")
        .unbounded()
        .action((_, c) => c.copy(libFuzzerCompatOutput = true))
        .text("use libFuzzer like output instead of AFL like stats screen")
      opt[Unit]("quiet")
        .unbounded()
        .action((_, c) => c.copy(quiet = true))
        .text("avoid printing fuzzing statistics progress in the console")
      opt[Unit]("exitOnCrash")
        .unbounded()
        .action((_, c) => c.copy(exitOnCrash = true))
        .text("stop fuzzing once a crash is found.")
      opt[Int]("runTimeout")
        .unbounded()
        .action((x, c) => c.copy(runTimeout = Some(x)))
        .text("the timeout for each individual trial, in milliseconds")
    }

    try {
      parser.parse(args, JQFFuzzOptions()) match {
        case Some(opts) => execute(opts)
        case _ => System.exit(1)
      }
      System.gc();
    } catch {
      case e: Throwable =>
        System.gc();
        throw e
    }
  }

  def execute(opts: JQFFuzzOptions): Unit = {
    // Configure classes to instrument
    if (opts.excludes.nonEmpty) {
      System.setProperty("janala.excludes", opts.excludes.mkString(","))
    }
    if (opts.includes.nonEmpty) {
      System.setProperty("janala.includes", opts.includes.mkString(","))
    }

    // Configure Zest Guidance
    if (opts.saveAll) {
      System.setProperty("jqf.ei.SAVE_ALL_INPUTS", "true")
    }
    if (opts.libFuzzerCompatOutput) {
      System.setProperty("jqf.ei.LIBFUZZER_COMPAT_OUTPUT", "true")
    }
    if (opts.quiet) {
      System.setProperty("jqf.ei.QUIET_MODE", "true")
    }
    if (opts.exitOnCrash) {
      System.setProperty("jqf.ei.EXIT_ON_CRASH", "true")
    }
    if (opts.runTimeout.isDefined) {
      System.setProperty("jqf.ei.TIMEOUT", opts.runTimeout.get.toString)
    }

    val duration = opts.time.map { time =>
      try {
        Duration.parse("PT" + time);
      } catch {
        case e: DateTimeParseException =>
          throw new JQFException("Invalid time duration: " + time, e)
      }
    }.getOrElse(null)

    val loader = try {
      val classpath = opts.classpath.toArray
      if (opts.disableCoverage) {
        new URLClassLoader(
          classpath.map(cpe => new File(cpe).toURI().toURL()),
          getClass().getClassLoader())
      } else {
        new InstrumentingClassLoader(
          classpath,
          getClass().getClassLoader())
      }
    } catch {
      case e: MalformedURLException =>
        throw new JQFException("Could not get project classpath", e)
    }

    val guidance = try {
      val resultsDir = opts.outputDirectory
      val targetName = opts.testClassName + "#" + opts.testMethod
      val seedsDirOpt = opts.inputDirectory
      val guidance = (opts.engine, seedsDirOpt) match {
        case (Zest, Some(seedsDir)) =>
          new ZestGuidance(targetName, duration, resultsDir, seedsDir)
        case (Zest, None) =>
          new ZestGuidance(targetName, duration, resultsDir)
        case (Zeal, Some(seedsDir)) =>
          new ExecutionIndexingGuidance(targetName, duration, resultsDir, seedsDir)
        case (Zeal, None) =>
          throw new JQFException("--inputDirectory required when using zeal engine")
      }
      guidance.setBlind(opts.blind)
      guidance
    } catch {
      case e: FileNotFoundException =>
        throw new JQFException("File not found", e)
      case e: IOException =>
        throw new JQFException("I/O error", e)
    }

    val result = try {
      GuidedFuzzing.run(opts.testClassName, opts.testMethod, loader, guidance, System.out)
    } catch {
      case e: ClassNotFoundException =>
        throw new JQFException("could not load test class", e)
      case e: IllegalArgumentException =>
        throw new JQFException("Bad request", e)
      case e: RuntimeException =>
        throw new JQFException("Internal error", e)
    }

    if (!result.wasSuccessful()) {
      throw new JQFException(
        "Fuzzing revealed errors. Use mvn jqf:repro to reproduce failing test case.")
    }
  }
}
