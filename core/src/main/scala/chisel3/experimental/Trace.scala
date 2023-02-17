package chisel3.experimental

import chisel3.internal.HasId
import chisel3.{Aggregate, Data, Element, RawModule}
import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, CompleteTarget, SingleTargetAnnotation}

/** The util that records the reference map from original [[Data]]/[[Module]] annotated in Chisel and final FIRRTL.
  * @example
  * {{{
  *   class Dut extends Module {
  *     val a = WireDefault(Bool())
  *     Trace.traceName(a)
  *   }
  *   val annos = (new ChiselStage).execute(Seq(ChiselGeneratorAnnotation(() => new Dut)))
  *   val dut = annos.collectFirst { case DesignAnnotation(dut) => dut }.get.asInstanceOf[CollideModule]
  *   // get final reference of `a` Seq(ReferenceTarget("Dut", "Dut", Seq.empty, "a", Seq.empty))
  *   val firrtlReferenceOfDutA = finalTarget(annos)(dut.a)
  * }}}
  */
object Trace {

  /** Trace a Instance name. */
  def traceName(x: RawModule): Unit = {
    annotate(new ChiselAnnotation {
      def toFirrtl: Annotation = TraceAnnotation(x.toAbsoluteTarget, x.toAbsoluteTarget)
    })
  }

  /** Trace a Data name. This does NOT add "don't touch" semantics to the traced data. If you want this behavior, use an explicit [[chisel3.dontTouch]]. */
  def traceName(x: Data): Unit = {
    x match {
      case aggregate: Aggregate =>
        annotate(new ChiselAnnotation {
          def toFirrtl: Annotation = TraceAnnotation(aggregate.toAbsoluteTarget, aggregate.toAbsoluteTarget)
        })
        aggregate.elementsIterator.foreach(traceName)
      case element: Element =>
        annotate(new ChiselAnnotation {
          def toFirrtl: Annotation = TraceAnnotation(element.toAbsoluteTarget, element.toAbsoluteTarget)
        })
    }
  }

  /** An Annotation that records the original target annotate from Chisel.
    *
    * @param target target that should be renamed by [[firrtl.RenameMap]] in the firrtl transforms.
    * @param chiselTarget original annotated target in Chisel, which should not be changed or renamed in FIRRTL.
    */
  private case class TraceAnnotation[T <: CompleteTarget](target: T, chiselTarget: T)
      extends SingleTargetAnnotation[T] {
    def duplicate(n: T): Annotation = this.copy(target = n)
  }

  /** Get `CompleteTarget` of the target `x` for `annos`.
    * This API can be used to find the final reference to a signal or module which is marked by `traceName`
    */
  def finalTarget(annos: AnnotationSeq)(x: HasId): Seq[CompleteTarget] = finalTargetMap(annos)
    .getOrElse(x.toAbsoluteTarget, Seq.empty)

  /** Get all traced signal/module for `annos`
    * This API can be used to gather all final reference to the signal or module which is marked by `traceName`
    */
  def finalTargetMap(annos: AnnotationSeq): Map[CompleteTarget, Seq[CompleteTarget]] = annos.collect {
    case TraceAnnotation(t, chiselTarget) => chiselTarget -> t
  }.groupBy(_._1).map { case (k, v) => k -> v.map(_._2) }
}
