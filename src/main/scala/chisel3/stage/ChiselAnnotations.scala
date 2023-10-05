// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{
  BufferedCustomFileEmission,
  CustomFileEmission,
  HasShellOptions,
  OptionsException,
  ShellOption,
  StageOptions,
  Unserializable
}
import firrtl.options.internal.WriteableCircuitAnnotation
import firrtl.options.Viewer.view
import chisel3.{deprecatedMFCMessage, ChiselException, Module}
import chisel3.RawModule
import chisel3.internal.{Builder, WarningFilter}
import chisel3.internal.firrtl.{Circuit, Converter}
import firrtl.AnnotationSeq
import firrtl.ir.{CircuitWithAnnos, Serializer}
import scala.util.control.NonFatal
import java.io.{BufferedWriter, File, FileWriter}
import java.lang.reflect.InvocationTargetException

/** Mixin that indicates that this is an [[firrtl.annotations.Annotation]] used to generate a [[ChiselOptions]] view.
  */
sealed trait ChiselOption { this: Annotation => }

/** On an exception, this will cause the full stack trace to be printed as opposed to a pruned stack trace.
  */
case object PrintFullStackTraceAnnotation
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "full-stacktrace",
      toAnnotationSeq = _ => Seq(PrintFullStackTraceAnnotation),
      helpText = "Show full stack trace when an exception is thrown"
    )
  )

}

/** On recoverable errors, this will cause Chisel to throw an exception instead of continuing.
  */
case object ThrowOnFirstErrorAnnotation
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "throw-on-first-error",
      toAnnotationSeq = _ => Seq(ThrowOnFirstErrorAnnotation),
      helpText = "Throw an exception on the first error instead of continuing"
    )
  )

}

/** When enabled, warnings will be treated as errors.
  */
case object WarningsAsErrorsAnnotation
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "warnings-as-errors",
      toAnnotationSeq = _ => Seq(WarningsAsErrorsAnnotation),
      helpText = "Treat warnings as errors"
    )
  )

  private[chisel3] def asFilter: WarningFilter = WarningFilter(None, None, WarningFilter.Error)
}

// TODO shoud this be Unserializable or should it be propagated to MFC? Perhaps in a different form?
case class WarningConfigurationAnnotation(value: String)
    extends NoTargetAnnotation
    with Unserializable
    with ChiselOption {

  // This is eager so that the validity of the value String can be checked right away
  private[chisel3] val filters: Seq[WarningFilter] = {
    import chisel3.internal.ListSyntax
    val filters = value.split(",")
    filters.toList
      // Add accumulating index to each filter for error reporting
      .mapAccumulate(0) { case (idx, s) => (idx + 1 + s.length, (idx, s)) } // + 1 for removed ','
      ._2 // Discard accumulator
      .map {
        case (idx, s) =>
          WarningFilter.parse(s) match {
            case Right(wf) => wf
            case Left((jdx, msg)) =>
              val carat = (" " * (idx + jdx)) + "^"
              // Note tab before value and carat
              throw new Exception(s"Failed to parse configuration: $msg\n  $value\n  $carat")
          }
      }
  }
}

object WarningConfigurationAnnotation extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "warn-conf",
      toAnnotationSeq = { value =>
        try {
          Seq(WarningConfigurationAnnotation(value))
        } catch {
          case NonFatal(e) => throw new OptionsException(e.getMessage)
        }
      },
      helpText = "Warning configuration",
      helpValueName = Some("<value>")
    )
  )
}

// TODO shoud this be Unserializable or should it be propagated to MFC? Perhaps in a different form?
case class WarningConfigurationFileAnnotation(value: File)
    extends NoTargetAnnotation
    with Unserializable
    with ChiselOption {

  /** Removes line comments (starting with '#') and trims leading and trailing whitespace
    *
    * Returns the trimmed String and the number of whitespace characters trimmed from the beginning
    */
  private def trimAndRemoveComments(s: String): (String, Int) = {
    val commentStart = s.indexOf('#')
    val noComment = if (commentStart == -1) s else s.splitAt(commentStart)._1 // Only take part before line comment
    val trimmed = noComment.trim()
    // We still need to calculate how much whitespace was removed for use in error messages
    val amountTrimmedFromStart =
      trimmed.headOption.map(c => s.indexOf(c)).filter(_ > 0).getOrElse(0)
    (trimmed, amountTrimmedFromStart)
  }

  // This is eager so that the validity of the value String can be checked right away
  private[chisel3] val filters: Seq[WarningFilter] = {
    require(value.exists, s"Warning configuration file '$value' must exist!")
    require(value.isFile && value.canRead, s"Warning configuration file '$value' must be a readable file!")
    val lines = scala.io.Source.fromFile(value).getLines()
    lines.zipWithIndex.flatMap {
      case (contents, lineNo) =>
        val (str, jdx) = trimAndRemoveComments(contents)
        Option.when(str.nonEmpty) {
          WarningFilter.parse(str) match {
            case Right(wf) => wf
            case Left((idx, msg)) =>
              val carat = (" " * (idx + jdx)) + "^"
              val info = s"$value:${lineNo + 1}:$idx" // +1 to lineNo because we start at 0 but files start with 1
              // Note tab before value and carat
              throw new Exception(
                s"Failed to parse configuration at $info: $msg\n  $contents\n  $carat"
              )
          }
        }
    }.toVector
  }
}

object WarningConfigurationFileAnnotation extends HasShellOptions {
  val options = Seq(
    new ShellOption[File](
      longOption = "warn-conf-file",
      toAnnotationSeq = { value =>
        try {
          Seq(WarningConfigurationFileAnnotation(value))
        } catch {
          case NonFatal(e) => throw new OptionsException(e.getMessage)
        }
      },
      helpText = "Warning configuration",
      helpValueName = Some("<value>")
    )
  )
}

/** A root directory for source files, used for enhanced error reporting
  *
  * More than one may be provided. If a source file is found in more than one source root,
  * the first match will be used in error reporting.
  */
case class SourceRootAnnotation(directory: File) extends NoTargetAnnotation with Unserializable with ChiselOption

object SourceRootAnnotation extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "source-root",
      toAnnotationSeq = { dir =>
        val f = new File(dir)
        if (!f.isDirectory()) {
          throw new OptionsException(s"Must be directory that exists!")
        }
        Seq(SourceRootAnnotation(f))
      },
      helpText = "Root directory for source files, used for enhanced error reporting",
      helpValueName = Some("<file>")
    )
  )
}

/** An [[firrtl.annotations.Annotation]] storing a function that returns a Chisel module
  * @param gen a generator function
  */
case class ChiselGeneratorAnnotation(gen: () => RawModule) extends NoTargetAnnotation with Unserializable {

  /** Run elaboration on the Chisel module generator function stored by this [[firrtl.annotations.Annotation]]
    */
  def elaborate: AnnotationSeq = (new chisel3.stage.phases.Elaborate).transform(Seq(this))
}

object ChiselGeneratorAnnotation extends HasShellOptions {

  /** Construct a [[ChiselGeneratorAnnotation]] with a generator function that will try to construct a Chisel Module
    * from using that Module's name. The Module must both exist in the class path and not take parameters.
    * @param name a module name
    * @throws firrtl.options.OptionsException if the module name is not found or if no parameterless constructor for
    * that Module is found
    */
  def apply(name: String): ChiselGeneratorAnnotation = {
    val gen = () =>
      try {
        Class.forName(name).asInstanceOf[Class[_ <: RawModule]].getDeclaredConstructor().newInstance()
      } catch {
        // The reflective instantiation will box any exceptions thrown, unbox them here.
        // Note that this does *not* need to chain with the catches below which are triggered by an
        // invalid name or a constructor that takes arguments rather than by the code being run
        // itself.
        case e: InvocationTargetException =>
          throw e.getCause
        case e: ClassNotFoundException =>
          throw new OptionsException(s"Unable to locate module '$name'! (Did you misspell it?)", e)
        case e: NoSuchMethodException =>
          throw new OptionsException(
            s"Unable to create instance of module '$name'! (Does this class take parameters?)",
            e
          )
      }
    ChiselGeneratorAnnotation(gen)
  }

  val options = Seq(
    new ShellOption[String](
      longOption = "module",
      toAnnotationSeq = (a: String) => Seq(ChiselGeneratorAnnotation(a)),
      helpText = "The name of a Chisel module to elaborate (module must be in the classpath)",
      helpValueName = Some("<package>.<module>")
    )
  )

}

/** Stores a Chisel Circuit
  * @param circuit a Chisel Circuit
  */
case class ChiselCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with ChiselOption with Unserializable {
  /* Caching the hashCode for a large circuit is necessary due to repeated queries.
   * Not caching the hashCode will cause severe performance degredations for large [[Circuit]]s.
   */
  override lazy val hashCode: Int = circuit.hashCode
}

object CircuitSerializationAnnotation {
  sealed trait Format {
    def extension: String
  }
  case object FirrtlFileFormat extends Format {
    def extension = ".fir"
  }
}

import CircuitSerializationAnnotation._

/** Wraps a `Circuit` for serialization via `CustomFileEmission`
  * @param circuit a Chisel Circuit
  * @param filename name of destination file (excludes file extension)
  * @param format serialization file format (sets file extension)
  */
case class CircuitSerializationAnnotation(circuit: Circuit, filename: String, format: Format)
    extends NoTargetAnnotation
    with BufferedCustomFileEmission
    with WriteableCircuitAnnotation {
  /* Caching the hashCode for a large circuit is necessary due to repeated queries.
   * Not caching the hashCode will cause severe performance degredations for large [[Circuit]]s.
   */
  override lazy val hashCode: Int = circuit.hashCode

  protected def baseFileName(annotations: AnnotationSeq): String = filename

  protected def suffix: Option[String] = Some(format.extension)

  /** Write the circuit and annotations to the .fir file
    */
  override protected def writeToFileImpl(file: File, annos: Seq[Annotation]): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))

    val it = emitLazily(annos)
    it.foreach(writer.write(_))
    writer.close()
  }

  // Make this API visible in package chisel3 as well
  private[chisel3] def doWriteToFile(file: File, annos: Seq[Annotation]): Unit = writeToFileImpl(file, annos)

  /** Emit the circuit including annotations
    *
    * @note This API is lazy to improve performance and enable emitting circuits larger than 2 GiB
    */
  def emitLazily(annos: Seq[Annotation]): Iterable[String] = {
    // First emit all circuit logic without modules
    val prelude = {
      val dummyCircuit = circuit.copy(components = Nil)
      val converted = Converter.convert(dummyCircuit)
      val withAnnos = CircuitWithAnnos(converted, annos)
      Serializer.lazily(withAnnos)
    }
    val typeAliases: Seq[String] = circuit.typeAliases.map(_.name)
    val modules = circuit.components.iterator.map(c => Converter.convert(c, typeAliases))
    val moduleStrings = modules.flatMap { m =>
      Serializer.lazily(m, 1) ++ Seq("\n\n")
    }
    prelude ++ moduleStrings
  }

  override def getBytesBuffered: Iterable[Array[Byte]] = emitLazily(Nil).map(_.getBytes)
}

case class ChiselOutputFileAnnotation(file: String) extends NoTargetAnnotation with ChiselOption with Unserializable

object ChiselOutputFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "chisel-output-file",
      toAnnotationSeq = (a: String) => Seq(ChiselOutputFileAnnotation(a)),
      helpText = "Write Chisel-generated FIRRTL to this file (default: <circuit-main>.fir)",
      helpValueName = Some("<file>")
    )
  )

}

/** Contains the top-level elaborated Chisel design.
  *
  * By default is created during Chisel elaboration and passed to the FIRRTL compiler.
  * @param design top-level Chisel design
  * @tparam DUT Type of the top-level Chisel design
  */
case class DesignAnnotation[DUT <: RawModule](design: DUT) extends NoTargetAnnotation with Unserializable
