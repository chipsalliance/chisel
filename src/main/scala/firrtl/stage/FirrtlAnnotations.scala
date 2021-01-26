// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl._
import firrtl.ir.Circuit
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{HasShellOptions, OptionsException, ShellOption, Unserializable}
import java.io.FileNotFoundException
import java.nio.file.NoSuchFileException

import firrtl.stage.TransformManager.TransformDependency

/** Indicates that this is an [[firrtl.annotations.Annotation Annotation]] directly used in the construction of a
  * [[FirrtlOptions]] view.
  */
sealed trait FirrtlOption extends Unserializable { this: Annotation => }

/** Indicates that this [[firrtl.annotations.Annotation Annotation]] contains information that is directly convertable
  * to a FIRRTL [[firrtl.ir.Circuit Circuit]].
  */
sealed trait CircuitOption extends Unserializable { this: Annotation =>

  /** Convert this [[firrtl.annotations.Annotation Annotation]] to a [[FirrtlCircuitAnnotation]]
    */
  def toCircuit(info: Parser.InfoMode): FirrtlCircuitAnnotation

}

/** An explicit input FIRRTL file to read
  *  - set with `-i/--input-file`
  *  - If unset, an [[FirrtlFileAnnotation]] with the default input file __will not be generated__
  * @param file input filename
  */
case class FirrtlFileAnnotation(file: String) extends NoTargetAnnotation with CircuitOption {

  def toCircuit(info: Parser.InfoMode): FirrtlCircuitAnnotation = {
    val circuit =
      try {
        FirrtlStageUtils.getFileExtension(file) match {
          case ProtoBufFile => proto.FromProto.fromFile(file)
          case FirrtlFile   => Parser.parseFile(file, info)
        }
      } catch {
        case a @ (_: FileNotFoundException | _: NoSuchFileException) =>
          throw new OptionsException(s"Input file '$file' not found! (Did you misspell it?)", a)
      }
    FirrtlCircuitAnnotation(circuit)
  }

}

object FirrtlFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "input-file",
      toAnnotationSeq = a => Seq(FirrtlFileAnnotation(a)),
      helpText = "An input FIRRTL file",
      shortOption = Some("i"),
      helpValueName = Some("<file>")
    )
  )

}

/** An explicit output file the emitter will write to
  *   - set with `-o/--output-file`
  *  @param file output filename
  */
case class OutputFileAnnotation(file: String) extends NoTargetAnnotation with FirrtlOption

object OutputFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "output-file",
      toAnnotationSeq = a => Seq(OutputFileAnnotation(a)),
      helpText = "The output FIRRTL file",
      shortOption = Some("o"),
      helpValueName = Some("<file>")
    )
  )

}

/** Sets the info mode style
  *  - set with `--info-mode`
  * @param mode info mode name
  * @note This cannote be directly converted to [[Parser.InfoMode]] as that depends on an optional [[FirrtlFileAnnotation]]
  */
case class InfoModeAnnotation(modeName: String = "use") extends NoTargetAnnotation with FirrtlOption {
  require(
    modeName match { case "use" | "ignore" | "gen" | "append" => true; case _ => false },
    s"Unknown info mode '$modeName'! (Did you misspell it?)"
  )

  /** Return the [[Parser.InfoMode]] equivalent for this [[firrtl.annotations.Annotation Annotation]]
    * @param infoSource the name of a file to use for "gen" or "append" info modes
    */
  def toInfoMode(infoSource: Option[String] = None): Parser.InfoMode = modeName match {
    case "use"    => Parser.UseInfo
    case "ignore" => Parser.IgnoreInfo
    case _ =>
      val a = infoSource.getOrElse("unknown source")
      modeName match {
        case "gen"    => Parser.GenInfo(a)
        case "append" => Parser.AppendInfo(a)
      }
  }
}

object InfoModeAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "info-mode",
      toAnnotationSeq = a => Seq(InfoModeAnnotation(a)),
      helpText = s"Source file info handling mode (default: ${apply().modeName})",
      helpValueName = Some("<ignore|use|gen|append>")
    )
  )

}

/** Holds a [[scala.Predef.String String]] containing FIRRTL source to read as input
  *  - set with `--firrtl-source`
  * @param value FIRRTL source as a [[scala.Predef.String String]]
  */
case class FirrtlSourceAnnotation(source: String) extends NoTargetAnnotation with CircuitOption {

  def toCircuit(info: Parser.InfoMode): FirrtlCircuitAnnotation =
    FirrtlCircuitAnnotation(Parser.parseString(source, info))

}

object FirrtlSourceAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "firrtl-source",
      toAnnotationSeq = a => Seq(FirrtlSourceAnnotation(a)),
      helpText = "An input FIRRTL circuit string",
      helpValueName = Some("<string>")
    )
  )

}

/** helpValueName a [[Compiler]] that should be run
  *  - set stringly with `-X/--compiler`
  *  - If unset, a [[CompilerAnnotation]] with the default [[VerilogCompiler]]
  * @param compiler compiler name
  */
@deprecated("Use a RunFirrtlTransformAnnotation targeting a specific Emitter.", "FIRRTL 1.4.0")
case class CompilerAnnotation(compiler: Compiler = new VerilogCompiler()) extends NoTargetAnnotation with FirrtlOption

@deprecated("Use a RunFirrtlTransformAnnotation targeting a specific Emitter.", "FIRRTL 1.4.0")
object CompilerAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "compiler",
      toAnnotationSeq = a => Seq(RunFirrtlTransformAnnotation.stringToEmitter(a)),
      helpText = "The FIRRTL compiler to use (default: verilog)",
      shortOption = Some("X"),
      helpValueName = Some("<none|high|middle|low|verilog|mverilog|sverilog>")
    )
  )

}

/** Holds the unambiguous class name of a [[Transform]] to run
  *  - will be append to [[FirrtlExecutionOptions.customTransforms]]
  *  - set with `-fct/--custom-transforms`
  * @param transform the full class name of the transform
  */
case class RunFirrtlTransformAnnotation(transform: Transform) extends NoTargetAnnotation

object RunFirrtlTransformAnnotation extends HasShellOptions {

  def apply(transform: TransformDependency): RunFirrtlTransformAnnotation =
    RunFirrtlTransformAnnotation(transform.getObject())

  private[firrtl] def stringToEmitter(a: String): RunFirrtlTransformAnnotation = {
    val emitter = a match {
      case "none"     => new ChirrtlEmitter
      case "high"     => new HighFirrtlEmitter
      case "low"      => new LowFirrtlEmitter
      case "middle"   => new MiddleFirrtlEmitter
      case "verilog"  => new VerilogEmitter
      case "mverilog" => new MinimumVerilogEmitter
      case "sverilog" => new SystemVerilogEmitter
      case _          => throw new OptionsException(s"Unknown compiler name '$a'! (Did you misspell it?)")
    }
    RunFirrtlTransformAnnotation(emitter)
  }

  val options = Seq(
    new ShellOption[Seq[String]](
      longOption = "custom-transforms",
      toAnnotationSeq = _.map(txName =>
        try {
          val tx = Class.forName(txName).asInstanceOf[Class[_ <: Transform]].newInstance()
          RunFirrtlTransformAnnotation(tx)
        } catch {
          case e: ClassNotFoundException =>
            throw new OptionsException(s"Unable to locate custom transform $txName (did you misspell it?)", e)
          case e: InstantiationException =>
            throw new OptionsException(
              s"Unable to create instance of Transform $txName (is this an anonymous class?)",
              e
            )
          case e: Throwable => throw new OptionsException(s"Unknown error when instantiating class $txName", e)
        }
      ),
      helpText = "Run these transforms during compilation",
      shortOption = Some("fct"),
      helpValueName = Some("<package>.<class>")
    ),
    new ShellOption[String](
      longOption = "change-name-case",
      toAnnotationSeq = _ match {
        case "lower" => Seq(RunFirrtlTransformAnnotation(new firrtl.features.LowerCaseNames))
        case "upper" => Seq(RunFirrtlTransformAnnotation(new firrtl.features.UpperCaseNames))
        case a       => throw new OptionsException(s"Unknown case '$a'. Did you misspell it?")
      },
      helpText = "Convert all FIRRTL names to a specific case",
      helpValueName = Some("<lower|upper>")
    ),
    new ShellOption[String](
      longOption = "compiler",
      toAnnotationSeq = a => Seq(stringToEmitter(a)),
      helpText = "The FIRRTL compiler to use (default: verilog)",
      shortOption = Some("X"),
      helpValueName = Some("<none|high|middle|low|verilog|mverilog|sverilog>")
    )
  )

}

/** Holds a FIRRTL [[firrtl.ir.Circuit Circuit]]
  * @param circuit a circuit
  */
case class FirrtlCircuitAnnotation(circuit: Circuit) extends NoTargetAnnotation with FirrtlOption {
  /* Caching the hashCode for a large circuit is necessary due to repeated queries, e.g., in
   * [[Compiler.propagateAnnotations]]. Not caching the hashCode will cause severe performance degredations for large
   * [[Annotations]].
   */
  override lazy val hashCode: Int = circuit.hashCode

}

/** Suppresses warning about Scala 2.11 deprecation
  *
  *  - set with `--warn:no-scala-version-deprecation`
  */
case object WarnNoScalaVersionDeprecation extends NoTargetAnnotation with FirrtlOption with HasShellOptions {
  def longOption: String = "warn:no-scala-version-deprecation"
  val options = Seq(
    new ShellOption[Unit](
      longOption = longOption,
      toAnnotationSeq = { _ => Seq(this) },
      helpText = "Suppress Scala 2.11 deprecation warning (ignored in Scala 2.12+)"
    )
  )
}

/** Turn off all expression inlining
  *
  * @note this primarily applies to emitted Verilog
  */
case object PrettyNoExprInlining extends NoTargetAnnotation with FirrtlOption with HasShellOptions {
  def longOption: String = "pretty:no-expr-inlining"
  val options = Seq(
    new ShellOption[Unit](
      longOption = longOption,
      toAnnotationSeq = { _ => Seq(this) },
      helpText = "Disable expression inlining"
    )
  )
}

/** Turn off folding a specific primitive operand
  * @param op the op that should never be folded
  */
case class DisableFold(op: ir.PrimOp) extends NoTargetAnnotation with FirrtlOption

object DisableFold extends HasShellOptions {

  private val mapping: Map[String, ir.PrimOp] = PrimOps.builtinPrimOps.map { case op => op.toString -> op }.toMap

  override val options = Seq(
    new ShellOption[String](
      longOption = "dont-fold",
      toAnnotationSeq = a => {
        mapping
          .get(a)
          .orElse(throw new OptionsException(s"Unknown primop '$a'. (Did you misspell it?)"))
          .map(DisableFold(_))
          .toSeq
      },
      helpText = "Disable folding of specific primitive operations",
      helpValueName = Some("<primop>")
    )
  )

}
