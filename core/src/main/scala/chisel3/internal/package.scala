// SPDX-License-Identifier: Apache-2.0

package chisel3

import firrtl.annotations.{IsModule, ModuleTarget}
import chisel3.experimental.{BaseModule, SourceInfo, UnlocatableSourceInfo}
import chisel3.reflect.DataMirror.hasProbeTypeModifier
import chisel3.internal.binding._
import chisel3.internal.firrtl.ir.{Component, DefModule}
import chisel3.internal.Builder.Prefix

import scala.util.Try
import scala.annotation.implicitNotFound
import scala.collection.mutable
import chisel3.ChiselException

package object internal {

  @implicitNotFound("You are trying to access a macro-only API. Please use the @public annotation instead.")
  trait MacroGenerated

  /** Marker trait for modules that are not true modules */
  private[chisel3] trait PseudoModule extends BaseModule

  /* Check if a String name is a temporary name */
  def isTemp(name: String): Boolean = name.nonEmpty && name.head == '_'

  /** Creates a name String from a prefix and a seed
    * @param prefix The prefix associated with the seed (must be in correct order, *not* reversed)
    * @param seed The seed for computing the name (if available)
    */
  def buildName(seed: String, prefix: Prefix): String = {
    // Don't bother copying the String if there's no prefix
    if (prefix.isEmpty) {
      seed
    } else {
      // Using Java's String builder to micro-optimize appending a String excluding 1st character
      // for temporaries
      val builder = new java.lang.StringBuilder()
      // Starting with _ is the indicator of a temporary
      val temp = isTemp(seed)
      // Make sure the final result is also a temporary if this is a temporary
      if (temp) {
        builder.append('_')
      }
      prefix.foreach { p =>
        // Don't append _ if the prefix is empty
        if (p.nonEmpty) {
          builder.append(p)
          builder.append('_')
        }
      }
      if (temp) {
        // We've moved the leading _ to the front, drop it here
        builder.append(seed, 1, seed.length)
      } else {
        builder.append(seed)
      }
      builder.toString
    }
  }

  // Sanitizes a name, e.g. from a `HasId`, by stripping all non ANSI-C characters
  private[chisel3] def sanitize(s: String, leadingDigitOk: Boolean = false): String = {
    // TODO what character set does FIRRTL truly support? using ANSI C for now
    def legalStart(c: Char) = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'
    def legal(c:      Char) = legalStart(c) || (c >= '0' && c <= '9')
    val res = if (s.forall(legal)) s else s.filter(legal)
    val headOk = (!res.isEmpty) && (leadingDigitOk || legalStart(res.head))
    if (headOk) res else s"_$res"
  }

  // Workaround for https://github.com/chipsalliance/chisel/issues/4162
  // We can't use the .asTypeOf workaround because this is used to implement .asTypeOf
  private[chisel3] def _padHandleBool[A <: Bits](
    x:      A,
    width:  Int,
    isUInt: Boolean
  )(
    implicit sourceInfo: SourceInfo
  ): A = x match {
    case b: Bool if !b.isLit && width > 1 && isUInt =>
      val _pad = Wire(UInt(width.W))
      _pad := b
      _pad.asInstanceOf[A] // This cast is safe because we know A is UInt on this path
    case u => u.pad(width).asInstanceOf[A]
  }

  // Resize that to this width (if known)
  private[chisel3] def _resizeToWidth[A <: Bits](
    that:           A,
    targetWidthOpt: Option[Int],
    isUInt:         Boolean
  )(fromUInt: UInt => A)(
    implicit sourceInfo: SourceInfo
  ): A = {
    (targetWidthOpt, that.widthOption) match {
      case (Some(targetWidth), Some(thatWidth)) =>
        if (targetWidth == thatWidth) that
        else if (targetWidth > thatWidth) _padHandleBool(that, targetWidth, isUInt)
        else fromUInt(that.take(targetWidth))
      case (Some(targetWidth), None) => fromUInt(_padHandleBool(that, targetWidth, isUInt).take(targetWidth))
      case (None, Some(thatWidth))   => that
      case (None, None)              => that
    }
  }

  /** Internal API for [[ViewParent]] */
  sealed private[chisel3] class ViewParentAPI extends RawModule() with PseudoModule {
    // We must provide `absoluteTarget` but not `toTarget` because otherwise they would be exactly
    // the same and we'd have no way to distinguish the kind of target when renaming view targets in
    // the Converter
    // Note that this is not overriding .toAbsoluteTarget, that is a final def in BaseModule that delegates
    // to this method
    private[chisel3] val absoluteTarget: IsModule = ModuleTarget("_$$AbsoluteView$$_")

    // This module is not instantiable
    override private[chisel3] def generateComponent():  Option[Component] = None
    override private[chisel3] def initializeInParent(): Unit = ()
    // This module is not really part of the circuit
    _parent = None

    // Sigil to mark views, starts with '_' to make it a legal FIRRTL target
    override def desiredName = "_$$View$$_"

    private[chisel3] val fakeComponent: Component = DefModule(this, desiredName, false, Nil, Nil, null)
  }

  /** Special internal object representing the parent of all views
    *
    * @note this is a val instead of an object because of the need to wrap in Module(...)
    * @note this is a lazy val so that calling functions in this package object doesn't create it
    */
  private[chisel3] lazy val ViewParent =
    Module._applyImpl(new ViewParentAPI)(UnlocatableSourceInfo)

  private[chisel3] def requireHasProbeTypeModifier(
    probe:        Data,
    errorMessage: String = ""
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val msg = if (errorMessage.isEmpty) s"Expected a probe." else errorMessage
    if (!hasProbeTypeModifier(probe)) Builder.error(msg)
  }

  private[chisel3] def requireNoProbeTypeModifier(
    probe:        Data,
    errorMessage: String = ""
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val msg = if (errorMessage.isEmpty) s"Did not expect a probe." else errorMessage
    if (hasProbeTypeModifier(probe)) Builder.error(msg)
  }

  private[chisel3] def requireHasWritableProbeTypeModifier(
    probe:        Data,
    errorMessage: String = ""
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val msg = if (errorMessage.isEmpty) s"Expected a writable probe." else errorMessage
    requireHasProbeTypeModifier(probe, msg)
    if (!probe.probeInfo.get.writable) Builder.error(msg)
  }

  private[chisel3] def containsProbe(data: Data): Boolean = data match {
    case a: Aggregate => a.probeInfo.nonEmpty || a.elementsContainProbe
    case leaf => leaf.probeInfo.nonEmpty
  }

  private[chisel3] def requireCompatibleDestinationProbeColor(
    dest:         Data,
    errorMessage: => String = ""
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val destLayer = dest.probeInfo match {
      case Some(Data.ProbeInfo(_, Some(color))) =>
        color
      case _ => return
    }
    if (Builder.layerStack.headOption.forall(_.canWriteTo(destLayer))) {
      return
    }
    Builder.error(errorMessage)
  }

  private[chisel3] def requireNotChildOfProbe(
    probe:        Data,
    errorMessage: => String = ""
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    probe.binding match {
      case Some(ChildBinding(parent)) =>
        if (parent.probeInfo.nonEmpty) {
          val providedMsg = errorMessage // only evaluate by-name argument once
          val msg = if (providedMsg.isEmpty) "Expected a root of a probe." else providedMsg
          Builder.error(msg)
        }
      case _ => ()
    }
  }

  // TODO this exists in cats.Traverse, should we just use that?
  private[chisel3] implicit class IterableSyntax[A, CC[X] <: Iterable[X]](xs: CC[A]) {
    def mapAccumulate[B, C](
      z: B
    )(f: (B, A) => (B, C))(implicit bf: scala.collection.BuildFrom[CC[A], C, CC[C]]): (B, CC[C]) = {
      val builder = bf.newBuilder(xs)
      var acc = z
      xs.foreach { a =>
        val (newAcc, c) = f(acc, a)
        acc = newAcc
        builder += c
      }
      (acc, builder.result())
    }
  }

  /** This is effectively a "LazyVal" box type, we can create the object but delay executing the argument
    *
    * @note This is similar to cats.Eval.later but we don't depend on Cats
    */
  private[chisel3] class Delayed[A](a: => A) {
    lazy val value: A = a
  }
  private[chisel3] object Delayed {
    def apply[A](a: => A): Delayed[A] = new Delayed(a)
  }

  /** The list of banned type alias words which will cause generation of bad FIRRTL. These are usually
    * keyword tokens that would be automatically lexed by firtool, and so cause parsing errors.
    */
  private[chisel3] val illegalTypeAliases = Seq(
    "FIRRTL",
    "Clock",
    "UInt",
    "Reset",
    "AsyncReset",
    "Analog",
    "Probe",
    "RWProbe",
    "version",
    "type",
    "circuit",
    "parameter",
    "input",
    "output",
    "extmodule",
    "module",
    "intmodule",
    "intrinsic",
    "defname",
    "const",
    "flip",
    "reg",
    "smem",
    "cmem",
    "mport",
    "define",
    "attach",
    "inst",
    "of",
    "reset",
    "printf",
    "skip",
    "node"
  )

  /** Similar to Seq.groupBy except that it preserves ordering of elements within each group */
  private[chisel3] def groupByIntoSeq[A, K](xs: Iterable[A])(f: A => K): Seq[(K, Seq[A])] = {
    val map = mutable.LinkedHashMap.empty[K, mutable.ListBuffer[A]]
    for (x <- xs) {
      val key = f(x)
      val l = map.getOrElseUpdate(key, mutable.ListBuffer.empty[A])
      l += x
    }
    map.view.map({ case (k, vs) => k -> vs.toList }).toList
  }
}
