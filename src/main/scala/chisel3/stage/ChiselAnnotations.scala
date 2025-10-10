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
import chisel3.{deprecatedMFCMessage, ChiselException, ElaboratedCircuit, Module}
import chisel3.RawModule
import chisel3.layer.Layer
import chisel3.internal.{Builder, WarningFilter}
import chisel3.internal.firrtl.ir.Circuit
import chisel3.internal.firrtl.Converter
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}
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
      .map { case (idx, s) =>
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
    lines.zipWithIndex.flatMap { case (contents, lineNo) =>
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

  /** Try to convert a string to a Scala type. */
  private def stringToAny(str: String): Any = {

    /* Something that looks like object creation, e.g., "Foo(42)" */
    val classPattern = "([a-zA-Z0-9_$.]+)\\((.*)\\)".r

    str match {
      case boolean if boolean.toBooleanOption.isDefined => boolean.toBoolean
      case integer if integer.toIntOption.isDefined     => integer.toInt
      case float if float.toDoubleOption.isDefined      => float.toDouble
      case classPattern(a, b) =>
        val constructor = Class.forName(a).getConstructors()(0)
        if (b.isEmpty) {
          constructor.newInstance()
        } else {
          val arguments = b.split(',').map(stringToAny).toSeq
          constructor.newInstance(arguments: _*)
        }
      case string => str
    }
  }

  /** Construct a [[ChiselGeneratorAnnotation]] with a generator function that will try to construct a Chisel Module
    * from using that Module's name. The Module must both exist in the class path and not take parameters.
    * @param name a module name
    * @throws firrtl.options.OptionsException if the module name is not found or if no parameterless constructor for
    * that Module is found
    */
  def apply(name: String): ChiselGeneratorAnnotation = {
    val gen = () =>
      try {
        stringToAny(name).asInstanceOf[RawModule]
      } catch {
        // The reflective instantiation will box any exceptions thrown, unbox them here.
        // Note that this does *not* need to chain with the catches below which are triggered by an
        // invalid name or a constructor that takes arguments rather than by the code being run
        // itself.
        case e: InvocationTargetException =>
          throw e.getCause
        case e: ClassNotFoundException =>
          throw new OptionsException(
            s"Unable to run module generator '$name' because it or one of its arguments could not be found. (Did you misspell it or them?)",
            e
          )
        case e: IllegalArgumentException =>
          throw new OptionsException(
            s"Unable to run module generator '$name' because the arguments are invalid. (Did you pass the correct number and type of arguments?)",
            e
          )
        case e: ClassCastException =>
          throw new OptionsException(
            s"Unable to run module generator '$name' because this is not a 'RawModule'. (Did you try to construct something that is not a 'RawModule' or did you forget to append '()' to indicate that this is not a string?)",
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
class ChiselCircuitAnnotation private (val elaboratedCircuit: ElaboratedCircuit)
    extends NoTargetAnnotation
    with ChiselOption
    with Unserializable
    with Product
    with Serializable {
  def canEqual(that: Any): Boolean = that.isInstanceOf[ChiselCircuitAnnotation]

  override def equals(obj: Any): Boolean = obj match {
    case that: ChiselCircuitAnnotation => this.elaboratedCircuit == that.elaboratedCircuit
    case _ => false
  }

  def productArity: Int = 1

  def productElement(n: Int): Any = n match {
    case 0 => elaboratedCircuit
    case _ => throw new IndexOutOfBoundsException(s"Invalid index $n")
  }

  /* Caching the hashCode for a large circuit is necessary due to repeated queries.
   * Not caching the hashCode will cause severe performance degredations for large [[Circuit]]s.
   */
  override lazy val hashCode: Int = elaboratedCircuit.hashCode
}

object ChiselCircuitAnnotation extends scala.runtime.AbstractFunction1[ElaboratedCircuit, ChiselCircuitAnnotation] {

  def apply(elaboratedCircuit: ElaboratedCircuit): ChiselCircuitAnnotation = new ChiselCircuitAnnotation(
    elaboratedCircuit
  )
}

object CircuitSerializationAnnotation {
  sealed trait Format {
    def extension: String
  }
  case object FirrtlFileFormat extends Format {
    def extension = ".fir"
  }

  @deprecated("Construct with ElaboratedCircuit instead.", "Chisel 6.7.0")
  def apply(circuit: Circuit, filename: String, format: Format): CircuitSerializationAnnotation =
    new CircuitSerializationAnnotation(Left(circuit), filename, format)

  def apply(elaboratedCircuit: ElaboratedCircuit, filename: String): CircuitSerializationAnnotation =
    new CircuitSerializationAnnotation(Right(elaboratedCircuit), filename, FirrtlFileFormat)

  @deprecated("Unapply is deprecated, access fields directly.", "Chisel 6.7.0")
  def unapply(c: CircuitSerializationAnnotation): Option[(Circuit, String, Format)] = Some(
    (c.circuit, c.filename, FirrtlFileFormat)
  )
}

import CircuitSerializationAnnotation.{FirrtlFileFormat, Format}

/** Wraps a `Circuit` for serialization via `CustomFileEmission`
  * @param circuit a Chisel Circuit
  * @param filename name of destination file (excludes file extension)
  * @param format serialization file format (sets file extension)
  */
class CircuitSerializationAnnotation private (
  private val _circuit: Either[Circuit, ElaboratedCircuit],
  val filename:         String,
  val format:           Format
) extends NoTargetAnnotation
    with BufferedCustomFileEmission
    with WriteableCircuitAnnotation
    with Product
    with Serializable {

  @deprecated("Construct with ElaboratedCircuit instead", "Chisel 6.7.0")
  def this(circuit: Circuit, filename: String, format: Format) = this(Left(circuit), filename, format)

  @deprecated("Use elaboratedCircuit instead", "Chisel 6.7.0")
  def circuit: Circuit = _circuit.map(_._circuit).merge

  def elaboratedCircuit: ElaboratedCircuit = _circuit.toOption.getOrElse(
    throw new Exception("This object was built with the deprecated Circuit constructor, cannot get elaboratedCircuit.")
  )

  /* Caching the hashCode for a large circuit is necessary due to repeated queries.
   * Not caching the hashCode will cause severe performance degredations for large [[Circuit]]s.
   */
  override lazy val hashCode: Int = _circuit.hashCode

  override def canEqual(that: Any): Boolean = that.isInstanceOf[CircuitSerializationAnnotation]

  override def equals(obj: Any): Boolean = obj match {
    case that: CircuitSerializationAnnotation =>
      this._circuit == that._circuit && this.filename == that.filename && this.format == that.format
    case _ => false
  }

  override def productArity: Int = 3

  def productElement(n: Int): Any = n match {
    case 0 => _circuit
    case 1 => filename
    case 2 => format
    case _ => throw new IndexOutOfBoundsException(s"Invalid index $n")
  }

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
  def emitLazily(annos: Seq[Annotation]): Iterable[String] = _circuit match {
    case Left(c)   => ElaboratedCircuit(c, Nil).lazilySerialize(annos)
    case Right(ec) => ec.lazilySerialize(annos)
  }

  override def getBytesBuffered: Iterable[Array[Byte]] = emitLazily(Nil).map(_.getBytes)

  @deprecated("Don't copy, just create a new one.", "Chisel 6.7.0")
  def copy(circuit: Circuit = this.circuit, filename: String = this.filename, format: Format = this.format) =
    new CircuitSerializationAnnotation(Left(circuit), filename, format)
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
case class DesignAnnotation[DUT <: RawModule](design: DUT, layers: Seq[chisel3.layer.Layer] = Seq.empty)
    extends NoTargetAnnotation
    with Unserializable

/** Use legacy Chisel width behavior.
  *
  * '''This should only be used for checking for unexpected semantic changes when bumping to Chisel 7.0.0.'''
  *
  * Use as CLI option `--use-legacy-width`.
  *
  * There are two width bugs fixed in Chisel 7.0 that could affect the semantics of user code.
  * Enabling this option will restore the old, buggy behavior, described below:
  *
  * 1. The width of shift-right when shift amount is >= the width of the argument
  *
  * When this option is enabled, the behavior is as follows:
  *   - Calling `.getWidth` on the resulting value will report the width as 0.
  *   - The width of the resulting value will be treated as 1-bit for generating Verilog.
  *
  * 2. The width of `ChiselEnum` values
  *
  * When this option is enabled, the behavior is as follows:
  *   - Calling `.getWidth` on a specific ChiselEnum value will give the width needed to encode the enum.
  *     This is the minimum width needed to encode the maximum value encoded by the enum.
  *   - The resulting FIRRTL will have the minimum width needed to encode the literal value for just that specific
  *   enum value.
  */
case object UseLegacyWidthBehavior
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "use-legacy-width",
      toAnnotationSeq = _ => Seq(UseLegacyWidthBehavior),
      helpText = "Use legacy (buggy) width behavior (pre-Chisel 7.0.0)"
    )
  )
}

/** This records a mapping from an old [[chisel3.layer.Layer]] to a new [[chisel3.layer.Layer]].
  *
  * This is intended to be used by a downstream Chisel project that is using an
  * upstream Chisel project which has different layers and the user would like
  * to align the upstream project with the downstream.
  * @param oldLayer the old layer that should be remapped
  * @param newLayer the new layer that the old layer should be replaced with
  */
case class RemapLayer(oldLayer: Layer, newLayer: Layer) extends NoTargetAnnotation with ChiselOption with Unserializable

object RemapLayer extends HasShellOptions {

  private def getLayer(name: String): Layer = try {
    Class.forName(name).getField("MODULE$").get(null).asInstanceOf[Layer]
  } catch {
    case e: NoSuchFieldException =>
      throw new OptionsException(
        s"Layer '$name' exists, but is not a singleton object. (Is this wrapped in an outer class?)",
        e
      )
    case e: ClassNotFoundException =>
      throw new OptionsException(
        s"Unable to reflectively find layer '$name'. (Did you misspell it?)",
        e
      )
    case e: ClassCastException =>
      throw new OptionsException(
        s"Object '$name' must be a `Layer`, but could not be cast as one.",
        e
      )
  }

  def apply(oldLayerName: String, newLayerName: String): RemapLayer = {
    RemapLayer(getLayer(oldLayerName), getLayer(newLayerName))
  }

  // Match things like `foo.bar.LayerA$,baz.LayerB$`
  private val layerMapRegex = "([\\w\\$.]+),([\\w\\$.]+)".r

  override val options = Seq(
    new ShellOption[String](
      longOption = "remap-layer",
      toAnnotationSeq = (raw: String) =>
        raw match {
          case layerMapRegex(oldLayerName, newLayerName) => Seq(RemapLayer(oldLayerName, newLayerName))
          case _ => throw new OptionsException(s"Invalid layer remap format: '$raw'")
        },
      helpText = "Globally remap a layer to another layer",
      helpValueName = Some("<oldLayer>,<newLayer>")
    )
  )

}

/** Include metadata for chisel utils.
  *
  * Some built-in Chisel utilities (like [[chisel3.util.SRAM]]) can optionally be built with metadata.
  * Adding this option will include the metadata when building relevant blocks.
  *
  * Use as CLI option `--include-util-metadata`.
  */
case object IncludeUtilMetadata extends NoTargetAnnotation with ChiselOption with HasShellOptions with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "include-util-metadata",
      toAnnotationSeq = _ => Seq(IncludeUtilMetadata),
      helpText = "Include metadata for chisel utils"
    )
  )
}

/** Use Blackbox implementation for SRAM
  *
  * Use as CLI option `--use-sram-blackbox`.
  */
case object UseSRAMBlackbox extends NoTargetAnnotation with ChiselOption with HasShellOptions with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "use-sram-blackbox",
      toAnnotationSeq = _ => Seq(UseSRAMBlackbox),
      helpText = "Use Blackbox implementation for SRAM"
    )
  )
}

case class IncludeInlineTestsForModuleAnnotation(glob: String)
    extends NoTargetAnnotation
    with Unserializable
    with ChiselOption

case object IncludeInlineTestsForModule extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "include-tests-module",
      toAnnotationSeq = glob => Seq(IncludeInlineTestsForModuleAnnotation(glob)),
      helpText = "Elaborate inline tests when the module-under-test name matches this glob"
    )
  )
}

case class IncludeInlineTestsWithNameAnnotation(glob: String)
    extends NoTargetAnnotation
    with Unserializable
    with ChiselOption

case object IncludeInlineTestsWithName extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "include-tests-name",
      toAnnotationSeq = glob => Seq(IncludeInlineTestsWithNameAnnotation(glob)),
      helpText = "Elaborate inline tests whose name matches this glob"
    )
  )
}

/** Suppress emission of source locators in FIRRTL output.
  *
  * Use as CLI option `--no-source-locators`.
  *
  * When this option is enabled, source locators (e.g., @[MyFile.scala 42:10]) will not be
  * emitted in the generated FIRRTL output. This can be useful for reducing output size or
  * for generating more stable output.
  */
case object SuppressSourceLocatorsAnnotation
    extends NoTargetAnnotation
    with ChiselOption
    with HasShellOptions
    with Unserializable {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-source-locators",
      toAnnotationSeq = _ => Seq(SuppressSourceLocatorsAnnotation),
      helpText = "Suppress emission of source locators in FIRRTL output"
    )
  )
}
