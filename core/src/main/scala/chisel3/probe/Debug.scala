package chisel3.probe

import chisel3._
import chisel3.reflect.DataMirror
import chisel3.connectable.{AlignedWithRoot, Alignment, FlippedWithRoot}
import chisel3.experimental.requireIsChiselType
import chisel3.experimental.OpaqueType
import scala.collection.immutable.SeqMap
import chisel3.SpecifiedDirection.Flip
import chisel3.SpecifiedDirection.Unspecified
import chisel3.Data.ProbeInfo
import scala.collection.IndexedSeqView
import chisel3.InternalErrorException
import chisel3.probe.forceInitial
import chisel3.internal.sourceinfo.SourceInfo

sealed trait DebugKind
case object ProducerKind extends DebugKind
case object ConsumerKind extends DebugKind
case object ReadOnlyKind extends DebugKind
case object TakeOverKind extends DebugKind

class Debug[T <: Data] private (original: T, kind: DebugKind) extends Record with OpaqueType {

  override def cloneType: this.type = {
    new Debug(original, kind).asInstanceOf[this.type]
  }

  /// Do not allow asUInt.
  protected override def errorOnAsUInt: Boolean = true

  // TODO: Look at what Property.scala implements for possible details to consider.

  // Walk members, recursing if callback returns true.
  private def walkMembers[D: DataMirror.HasMatchingZipOfChildren, L, R](
    left:    Option[D],
    right:   Option[D]
  )(process: ((Option[D], Option[D])) => Boolean
  ): Unit = {
    if (!process(left, right)) return
    val matcher = implicitly[DataMirror.HasMatchingZipOfChildren[D]]
    val childItems = matcher.matchingZipOfChildren(left, right).foreach {
      case (l, r) => walkMembers(l, r)(process)
    }
  }
  private def isFlipped(x: Alignment): Boolean = x match {
    case _: AlignedWithRoot => false
    case _: FlippedWithRoot => true
    case _ => throw new InternalErrorException(s"Match Error: $x")
  }

  private def shouldBeWritable(flip: Boolean): Boolean = kind match {
      case ProducerKind => flip
      case ConsumerKind => !flip
      case ReadOnlyKind => false
      case TakeOverKind => true
    }

  private def debugify(original: T, kind: DebugKind): Data = {
    // Pre-condition: original has no probeinfo, recursively.

    val isConsumer = true; // TODO: Revisit
    val copy = original.cloneTypeFull

    walkMembers(Some(Alignment(copy, isConsumer)), None) {
      case (Some(x: Alignment), _) =>
        (isFlipped(x), x.member) match {
          // Elements: Set probeinfo according to flip.
          case (flip: Boolean, x: Element) => {
            setProbeModifier(x, Some(ProbeInfo(shouldBeWritable(flip))))
            false
          }
          // Vector: special handling (sample_element).
          case (flip: Boolean, x: Vec[_]) => {
            x.sample_element.probeInfo = Some(ProbeInfo(shouldBeWritable(flip)))
            false
          }
          // Otherwise, recurse until hit one of the above.
          case _ => true
        }
      case _ => false
    }
    copy.specifiedDirection_=(SpecifiedDirection.Output)
    copy
    // TODO: Investigate why "Output(copy)" loses the probe info
    // PR3654 fixes this for "top-level" modifiers, but nothing recursively copies them presently?
    //Output(copy)
  }
  private val underlying = debugify(original, kind)
  def elements = SeqMap("" -> underlying)

  def define(value: T)(implicit sourceInfo: SourceInfo): Unit = {
    // Pre-condition: No probe info on 'value'.

    // TODO: Ensure value is a static reference?
    // For now, just let firtool tell us.
    walkMembers(Some(Alignment(value, true)), Some(Alignment(underlying, true))) {
      case (Some(a: Alignment), Some(b: Alignment)) =>
        (a.member, b.member) match {
          case (am: Element, bm: Element) => {
            probe.`package`.define(bm, if (shouldBeWritable(isFlipped(a))) RWProbeValue(am) else ProbeValue(am))
            false
          }
          case _ => true
        }
      case _ => throw new InternalErrorException("Debug initialization mismatch")
    }
    // "only use once"
  }
  def materialize(implicit sourceInfo: SourceInfo): T = {
    val w = Wire(original.cloneTypeFull)
    walkMembers(Some(Alignment(w, true)), Some(Alignment(underlying, true))) {
      case (Some(a: Alignment), Some(b: Alignment)) =>
        (a.member, b.member) match {
          case (am: Element, bm: Element) => {
            // TODO: Ensure not under a 'when'?
            if (isFlipped(a))
              forceInitial(bm, am)
            else
              am := read(bm)
            false
          }
          case _ => true
        }
      case _ => throw new InternalErrorException("Debug materialization mismatch")
    }
    w
  }

  // TODO:
  // def asProducer: Debug[T] = ??? // illegal on ReadOnlyKind
  // def asConsumer: Debug[T] = ??? // illegal on ReadOnlyKind
  // def asReadOnly: Debug[T] = ??? // illegal on ReadOnlyKind
}

object Debug {
  def producer[T <: Data](d:   T): Debug[T] = new Debug(d, ProducerKind)
  def consumer[T <: Data](d:   T): Debug[T] = new Debug(d, ConsumerKind)
  def takeOver[T <: Data](d:   T): Debug[T] = new Debug(d, TakeOverKind)
  def readAlways[T <: Data](d: T): Debug[T] = new Debug(d, ReadOnlyKind)
}
