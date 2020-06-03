// See LICENSE for license details.

package firrtl
package annotations

import firrtl.options.StageUtils


case class AnnotationException(message: String) extends Exception(message)

/** Base type of auxiliary information */
trait Annotation extends Product {

  /** Update the target based on how signals are renamed */
  def update(renames: RenameMap): Seq[Annotation]

  /** Optional pretty print
    *
    * @note rarely used
    */
  def serialize: String = this.toString

  /** Recurses through ls to find all [[Target]] instances
    * @param ls
    * @return
    */
  private def extractComponents(ls: scala.collection.Traversable[_]): Seq[Target] = {
    ls.collect {
      case c: Target => Seq(c)
      case o: Product => extractComponents(o.productIterator.toIterable)
      case x: scala.collection.Traversable[_] => extractComponents(x)
    }.foldRight(Seq.empty[Target])((seq, c) => c ++ seq)
  }

  /** Returns all [[firrtl.annotations.Target Target]] members in this annotation
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
      case from: Named =>
        val ret = renames.get(Target.convertNamed2Target(target))
        ret.map(_.map { newT =>
          val result = newT match {
            case c: InstanceTarget => ModuleName(c.ofModule, CircuitName(c.circuit))
            case c: IsMember =>
              val local = Target.referringModule(c)
              c.setPathTarget(local)
            case c: CircuitTarget => c.toNamed
            case other => throw Target.NamedException(s"Cannot convert $other to [[Named]]")
          }
          Target.convertTarget2Named(result) match {
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
          }
        }).getOrElse(List(this))
    }
  }
}

/** [[MultiTargetAnnotation]] keeps the renamed targets grouped within a single annotation. */
trait MultiTargetAnnotation extends Annotation {
  /** Contains a sequence of [[firrtl.annotations.Target Target]].
    * When created, [[targets]] should be assigned by `Seq(Seq(TargetA), Seq(TargetB), Seq(TargetC))`
    */
  val targets: Seq[Seq[Target]]

  /** Create another instance of this Annotation*/
  def duplicate(n: Seq[Seq[Target]]): Annotation

  /** Assume [[RenameMap]] is `Map(TargetA -> Seq(TargetA1, TargetA2, TargetA3), TargetB -> Seq(TargetB1, TargetB2))`
    * in the update, this Annotation is still one annotation, but the contents are renamed in the below form
    * Seq(Seq(TargetA1, TargetA2, TargetA3), Seq(TargetB1, TargetB2), Seq(TargetC))
    **/
  def update(renames: RenameMap): Seq[Annotation] = Seq(duplicate(targets.map(ts => ts.flatMap(renames(_)))))

  private def crossJoin[T](list: Seq[Seq[T]]): Seq[Seq[T]] =
    list match {
      case Nil => Nil
      case x :: Nil => x map (Seq(_))
      case x :: xs =>
        val xsJoin = crossJoin(xs)
        for {
          i <- x
          j <- xsJoin
        } yield {
          Seq(i) ++ j
        }
    }

  /** Assume [[RenameMap]] is `Map(TargetA -> Seq(TargetA1, TargetA2, TargetA3), TargetB -> Seq(TargetB1, TargetB2))`
    * After flat, this Annotation will be flat to the [[AnnotationSeq]] in the below form
    * Seq(Seq(TargetA1), Seq(TargetB1), Seq(TargetC)); Seq(Seq(TargetA1), Seq(TargetB2), Seq(TargetC))
    * Seq(Seq(TargetA2), Seq(TargetB1), Seq(TargetC)); Seq(Seq(TargetA2), Seq(TargetB2), Seq(TargetC))
    * Seq(Seq(TargetA3), Seq(TargetB1), Seq(TargetC)); Seq(Seq(TargetA3), Seq(TargetB2), Seq(TargetC))
    * */
  def flat(): AnnotationSeq = crossJoin(targets).map(r => duplicate(r.map(Seq(_))))
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
  def convertLegacyAnnos(annos: AnnotationSeq): AnnotationSeq = {
    var warned: Boolean = false
    annos.map {
      case legacy: LegacyAnnotation =>
        val annox = convertLegacyAnno(legacy)
        if (!warned && (annox ne legacy)) {
          val msg = s"A LegacyAnnotation was automatically converted.\n" + (" "*9) +
            "This functionality will soon be removed. Please migrate to new annotations."
          StageUtils.dramaticWarning(msg)
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
