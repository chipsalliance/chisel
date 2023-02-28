// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl._
import firrtl.ir.Circuit
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.{Dependency, HasShellOptions, OptionsException, ShellOption, Unserializable}
import java.io.{File, FileNotFoundException}
import java.nio.file.{NoSuchFileException, NotDirectoryException}

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

/** Read a directory of ProtoBufs
  *  - set with `-I/--input-directory`
  *
  * TODO: Does not currently support FIRRTL files.
  * @param dir input directory name
  */
case class FirrtlDirectoryAnnotation(dir: String) extends NoTargetAnnotation with CircuitOption {

  def toCircuit(info: Parser.InfoMode): FirrtlCircuitAnnotation = {
    val circuit =
      try {
        proto.FromProto.fromDirectory(dir)
      } catch {
        case a @ (_: FileNotFoundException | _: NoSuchFileException) =>
          throw new OptionsException(s"Directory '$dir' not found! (Did you misspell it?)", a)
        case _: NotDirectoryException =>
          throw new OptionsException(s"Directory '$dir' is not a directory")
      }
    FirrtlCircuitAnnotation(circuit)
  }

}

object FirrtlDirectoryAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "input-directory",
      toAnnotationSeq = a => Seq(FirrtlDirectoryAnnotation(a)),
      helpText = "A directory of FIRRTL files",
      shortOption = Some("I"),
      helpValueName = Some("<directory>")
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
      helpValueName = Some("<none|mhigh|high|middle|low|verilog|mverilog|sverilog>")
    )
  )

}

/** Holds the unambiguous class name of a [[Transform]] to run
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
      case "mhigh"    => new MinimumHighFirrtlEmitter
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
   * @note Uses the hashCode of the name of the circuit. Creating a HashMap with different Circuits
   *       that nevertheless have the same name is extremely uncommon so collisions are not a concern.
   *       Include productPrefix so that this doesn't collide with other types that use a similar
   *       strategy and hash the same String.
   */
  override lazy val hashCode: Int = (this.productPrefix + circuit.main).hashCode

}

/** Suppresses warning about Scala 2.11 deprecation
  *
  *  - set with `--warn:no-scala-version-deprecation`
  */
@deprecated("Support for Scala 2.11 has been dropped, this object no longer does anything", "FIRRTL 1.5")
case object WarnNoScalaVersionDeprecation extends NoTargetAnnotation with FirrtlOption with HasShellOptions {
  def longOption: String = "warn:no-scala-version-deprecation"
  val options = Seq(
    new ShellOption[Unit](
      longOption = longOption,
      toAnnotationSeq = { _ =>
        val msg = s"'$longOption' no longer does anything and will be removed in FIRRTL 1.6"
        firrtl.options.StageUtils.dramaticWarning(msg)
        Seq(this)
      },
      helpText = "(deprecated, this option does nothing)"
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

case object AllowUnrecognizedAnnotations extends NoTargetAnnotation with FirrtlOption with HasShellOptions {
  val options = Seq(
    new ShellOption[Unit](
      longOption = "allow-unrecognized-annotations",
      toAnnotationSeq = _ => Seq(this),
      helpText = "Allow annotation files to contain unrecognized annotations"
    )
  )
}

/** Turn off folding a specific primitive operand
  * @param op the op that should never be folded
  */
case class DisableFold(op: ir.PrimOp) extends NoTargetAnnotation with FirrtlOption

@deprecated("will be removed and merged into ConstantPropagation in 1.5", "1.4")
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

/** Indicate to the FIRRTL compiler that specific transforms have already been run.
  *
  * The intended use of this is for advanced users who want to skip specific transforms in the FIRRTL compiler.  It is
  * far safer for users to use the command line options to the FIRRTL compiler via `--start-from = <form>`.
  * @param currentState a sequence of transforms that have already been run on the circuit
  */
case class CurrentFirrtlStateAnnotation(currentState: Seq[TransformDependency])
    extends NoTargetAnnotation
    with FirrtlOption

private[stage] object CurrentFirrtlStateAnnotation extends HasShellOptions {

  /** This is just the transforms necessary for resolving types and checking that everything is okay. */
  private val dontSkip: Set[TransformDependency] = Set(
    Dependency[firrtl.stage.transforms.CheckScalaVersion],
    Dependency(passes.ResolveKinds),
    Dependency(passes.InferTypes),
    Dependency(passes.ResolveFlows)
  ) ++ Forms.Checks

  override val options = Seq(
    new ShellOption[String](
      longOption = "start-from",
      toAnnotationSeq = a =>
        (a match {
          case "chirrtl" => Seq.empty
          case "mhigh"   => Forms.MinimalHighForm
          case "high"    => Forms.HighForm
          case "middle"  => Forms.MidForm
          case "low"     => Forms.LowForm
          case "low-opt" => Forms.LowFormOptimized
          case _         => throw new OptionsException(s"Unknown start-from argument '$a'! (Did you misspell it?)")
        }).filterNot(dontSkip) match {
          case b if a.isEmpty => Seq.empty
          case b              => Seq(CurrentFirrtlStateAnnotation(b))
        },
      helpText = "",
      helpValueName = Some("<chirrtl|mhigh|high|middle|low|low-opt>")
    )
  )

}
