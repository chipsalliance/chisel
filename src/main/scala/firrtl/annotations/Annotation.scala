// See LICENSE for license details.

package firrtl
package annotations

import net.jcazevedo.moultingyaml._
import firrtl.annotations.AnnotationYamlProtocol._
import firrtl.Utils.throwInternalError

import scala.collection.mutable

case class AnnotationException(message: String) extends Exception(message)

/** Base type of auxiliary information */
trait Annotation extends Product {

  /** Update the target based on how signals are renamed */
  def update(renames: RenameMap): Seq[Annotation]

  /** Pretty Print
    *
    * @note In [[logger.LogLevel.Debug]] this is called on every Annotation after every Transform
    */
  def serialize: String = this.toString

  /** Recurses through ls to find all [[Target]] instances
    * @param ls
    * @return
    */
  private def extractComponents(ls: scala.collection.Traversable[_]): Seq[Target] = {
    ls.collect {
      case c: Target => Seq(c)
      case ls: scala.collection.Traversable[_] => extractComponents(ls)
    }.foldRight(Seq.empty[Target])((seq, c) => c ++ seq)
  }

  /** Returns all [[Target]] members in this annotation
    * @return
    */
  def getTargets: Seq[Target] = extractComponents(productIterator.toSeq)
}

/** If an Annotation does not target any [[Named]] thing in the circuit, then all updates just
  * return the Annotation itself
  */
trait NoTargetAnnotation extends Annotation {
  def update(renames: RenameMap): Seq[NoTargetAnnotation] = Seq(this)
}

/** An Annotation that targets a single [[Named]] thing */
trait SingleTargetAnnotation[T <: Named] extends Annotation {
  val target: T

  /** Create another instance of this Annotation */
  def duplicate(n: T): Annotation

  // This mess of @unchecked and try-catch is working around the fact that T is unknown due to type
  // erasure. We cannot that newTarget is of type T, but a CastClassException will be thrown upon
  // invoking duplicate if newTarget cannot be cast to T (only possible in the concrete subclass)
  def update(renames: RenameMap): Seq[Annotation] = {
    target match {
      case c: Target =>
        val x = renames.get(c)
        x.map(newTargets => newTargets.map(t => duplicate(t.asInstanceOf[T]))).getOrElse(List(this))
      case _: Named =>
        val ret = renames.get(Target.convertNamed2Target(target))
        ret.map(_.map(newT => Target.convertTarget2Named(newT: @unchecked) match {
          case newTarget: T @unchecked =>
            try {
              duplicate(newTarget)
            }
            catch {
              case _: java.lang.ClassCastException =>
                val msg = s"${this.getClass.getName} target ${target.getClass.getName} " +
                  s"cannot be renamed to ${newTarget.getClass}"
                throw AnnotationException(msg)
            }
        })).getOrElse(List(this))
    }
  }
}

@deprecated("Just extend NoTargetAnnotation", "1.1")
trait SingleStringAnnotation extends NoTargetAnnotation {
  def value: String
}

object Annotation {
  @deprecated("This returns a LegacyAnnotation, use an explicit Annotation type", "1.1")
  def apply(target: Named, transform: Class[_ <: Transform], value: String): LegacyAnnotation =
    new LegacyAnnotation(target, transform, value)
  @deprecated("This uses LegacyAnnotation, use an explicit Annotation type", "1.1")
  def unapply(a: LegacyAnnotation): Option[(Named, Class[_ <: Transform], String)] =
    Some((a.target, a.transform, a.value))
}

// Constructor is private so that we can still construct these internally without deprecation
// warnings
final case class LegacyAnnotation private[firrtl] (
    target: Named,
    transform: Class[_ <: Transform],
    value: String) extends SingleTargetAnnotation[Named] {
  val targetString: String = target.serialize
  val transformClass: String = transform.getName

  def targets(named: Named): Boolean = named == target
  def targets(transform: Class[_ <: Transform]): Boolean = transform == this.transform

  /**
    * This serialize is basically a pretty printer, actual serialization is handled by
    * AnnotationYamlProtocol
    * @return a nicer string than the raw case class default
    */
  override def serialize: String = {
    s"Annotation(${target.serialize},${transform.getCanonicalName},$value)"
  }

  def update(tos: Seq[Named]): Seq[Annotation] = {
    check(target, tos, this)
    propagate(target, tos, duplicate)
  }
  def propagate(from: Named, tos: Seq[Named], dup: Named=>Annotation): Seq[Annotation] = tos.map(dup(_))
  def check(from: Named, tos: Seq[Named], which: Annotation): Unit = {}
  def duplicate(n: Named): LegacyAnnotation = new LegacyAnnotation(n, transform, value)
}

// Private so that LegacyAnnotation can only be constructed via deprecated Annotation.apply
private[firrtl] object LegacyAnnotation {
  // ***** Everything below here is to help people migrate off of old annotations *****
  def errorIllegalAnno(name: String): Annotation =
    throw new Exception(s"Old-style annotations that look like $name are no longer supported")

  private val OldDeletedRegex = """(?s)DELETED by ([^\n]*)\n(.*)""".r
  private val PinsRegex = "pins:(.*)".r
  private val SourceRegex = "source (.+)".r
  private val SinkRegex = "sink (.+)".r

  import firrtl.transforms._
  import firrtl.passes._
  import firrtl.passes.memlib._
  import firrtl.passes.wiring._
  import firrtl.passes.clocklist._

  // Attempt to convert common Annotations and error on the rest of old-style build-in annotations
  // scalastyle:off
  def convertLegacyAnno(anno: LegacyAnnotation): Annotation = anno match {
    // All old-style Emitter annotations are illegal
    case LegacyAnnotation(_,_,"emitCircuit") => errorIllegalAnno("EmitCircuitAnnotation")
    case LegacyAnnotation(_,_,"emitAllModules") => errorIllegalAnno("EmitAllModulesAnnotation")
    case LegacyAnnotation(_,_,value) if value.startsWith("emittedFirrtlCircuit") =>
      errorIllegalAnno("EmittedFirrtlCircuitAnnotation")
    case LegacyAnnotation(_,_,value) if value.startsWith("emittedFirrtlModule") =>
      errorIllegalAnno("EmittedFirrtlModuleAnnotation")
    case LegacyAnnotation(_,_,value) if value.startsWith("emittedVerilogCircuit") =>
      errorIllegalAnno("EmittedVerilogCircuitAnnotation")
    case LegacyAnnotation(_,_,value) if value.startsWith("emittedVerilogModule") =>
      errorIllegalAnno("EmittedVerilogModuleAnnotation")
    // People shouldn't be trying to pass deleted annotations to Firrtl
    case LegacyAnnotation(_,_,OldDeletedRegex(_,_)) => errorIllegalAnno("DeletedAnnotation")
    // Some annotations we'll try to support
    case LegacyAnnotation(named, t, _) if t == classOf[InlineInstances] => InlineAnnotation(named)
    case LegacyAnnotation(n: ModuleName, t, outputConfig) if t == classOf[ClockListTransform] =>
      ClockListAnnotation(n, outputConfig)
    case LegacyAnnotation(CircuitName(_), transform, "") if transform == classOf[InferReadWrite] =>
      InferReadWriteAnnotation
    case LegacyAnnotation(_,_,PinsRegex(pins)) => PinAnnotation(pins.split(" "))
    case LegacyAnnotation(_, t, value) if t == classOf[ReplSeqMem] =>
      val args = value.split(" ")
      require(args.size == 2, "Something went wrong, stop using legacy ReplSeqMemAnnotation")
      ReplSeqMemAnnotation(args(0), args(1))
    case LegacyAnnotation(c: ComponentName, transform, "nodedupmem!")
      if transform == classOf[ResolveMemoryReference] => NoDedupMemAnnotation(c)
    case LegacyAnnotation(m: ModuleName, transform, "nodedup!")
      if transform == classOf[DedupModules] => NoDedupAnnotation(m)
    case LegacyAnnotation(c: ComponentName, _, SourceRegex(pin)) => SourceAnnotation(c, pin)
    case LegacyAnnotation(n, _, SinkRegex(pin)) => SinkAnnotation(n, pin)
    case LegacyAnnotation(m: ModuleName, t, text) if t == classOf[BlackBoxSourceHelper] =>
      val nArgs = 3
      text.split("\n", nArgs).toList match {
        case "resource" :: id ::  _ => BlackBoxResourceAnno(m, id)
        case "inline" :: name :: text :: _ => BlackBoxInlineAnno(m, name, text)
        case "targetDir" :: targetDir :: _ => BlackBoxTargetDirAnno(targetDir)
        case _ => errorIllegalAnno("BlackBoxSourceAnnotation")
      }
    case LegacyAnnotation(_, transform, "noDCE!") if transform == classOf[DeadCodeElimination] =>
      NoDCEAnnotation
    case LegacyAnnotation(c: ComponentName, _, "DONTtouch!") => DontTouchAnnotation(c.toTarget)
    case LegacyAnnotation(c: ModuleName, _, "optimizableExtModule!") =>
      OptimizableExtModuleAnnotation(c)
    case other => other
  }
  // scalastyle:on
  def convertLegacyAnnos(annos: Seq[Annotation]): Seq[Annotation] = {
    var warned: Boolean = false
    annos.map {
      case legacy: LegacyAnnotation =>
        val annox = convertLegacyAnno(legacy)
        if (!warned && (annox ne legacy)) {
          val msg = s"A LegacyAnnotation was automatically converted.\n" + (" "*9) +
            "This functionality will soon be removed. Please migrate to new annotations."
          Driver.dramaticWarning(msg)
          warned = true
        }
        annox
      case other => other
    }
  }
}

case class DeletedAnnotation(xFormName: String, anno: Annotation) extends NoTargetAnnotation {
  override def serialize: String = s"""DELETED by $xFormName\n${anno.serialize}"""
}

