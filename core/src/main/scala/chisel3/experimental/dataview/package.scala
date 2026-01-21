// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal._
import chisel3.internal.binding._
import chisel3.properties.Property

import scala.annotation.{implicitNotFound, tailrec}
import scala.collection.mutable
import chisel3.reflect.DataMirror
import chisel3.experimental.ClonePorts

package object dataview {
  case class InvalidViewException(message: String) extends chisel3.ChiselException(message)

  /** Provides `viewAs` for types that have an implementation of [[DataProduct]]
    *
    * Calling `viewAs` also requires an implementation of [[DataView]] for the target type
    */
  implicit class DataViewable[T](target: T) {
    private def _viewAsImpl[V <: Data](
      writability: ViewWriteability
    )(
      implicit dataproduct: DataProduct[T],
      dataView:             DataView[T, V],
      sourceInfo:           SourceInfo
    ): V = {
      // TODO put a try catch here for ExpectedHardwareException and perhaps others
      // It's likely users will accidentally use chiselTypeOf or something that may error,
      // The right thing to use is DataMirror...chiselTypeClone because of composition with DataView.andThen
      // Another option is that .andThen could give a fake binding making chiselTypeOfs in the user code safe
      val result: V = dataView.mkView(target)
      requireIsChiselType(result, "viewAs")

      doBind(target, result, dataView, writability)

      // Setting the parent marks these Data as Views
      result.setAllParents(Some(ViewParent))
      // The names of views do not matter except for when a view is annotated. For Views that correspond
      // To a single Data, we just forward the name of the Target. For Views that correspond to more
      // than one Data, we return this assigned name but rename it in the Convert stage
      result.forceName("view", Builder.viewNamespace)
      result
    }

    def viewAs[V <: Data](
      implicit dataproduct: DataProduct[T],
      dataView:             DataView[T, V],
      sourceInfo:           SourceInfo
    ): V = _viewAsImpl(ViewWriteability.Default)

    private[chisel3] def viewAsReadOnlyDeprecated[V <: Data](
      getWarning: SourceInfo => Warning
    )(
      implicit dataproduct: DataProduct[T],
      dataView:             DataView[T, V],
      sourceInfo:           SourceInfo
    ): V = _viewAsImpl(ViewWriteability.ReadOnlyDeprecated(getWarning))

    private[chisel3] def viewAsReadOnly[V <: Data](
      getError: SourceInfo => String
    )(
      implicit dataproduct: DataProduct[T],
      dataView:             DataView[T, V],
      sourceInfo:           SourceInfo
    ): V = _viewAsImpl(ViewWriteability.ReadOnly(getError))
  }

  /** Provides `viewAsSupertype` for subclasses of [[Record]] */
  implicit class RecordUpcastable[T <: Record](target: T) {

    /** View a [[Bundle]] or [[Record]] as a parent type (upcast) */
    def viewAsSupertype[V <: Record](proto: V)(implicit ev: ChiselSubtypeOf[T, V], sourceInfo: SourceInfo): V = {
      implicit val dataView: DataView[T, V] = PartialDataView.supertype[T, V](_ => proto)
      target.viewAs[V]
    }
  }

  private def nonTotalViewException(
    dataView:     DataView[_, _],
    target:       Any,
    view:         Data,
    targetFields: Seq[String],
    viewFields:   Seq[String]
  ) = {
    def missingMsg(name: String, fields: Seq[String]): Option[String] = {
      val str = fields.mkString(", ")
      fields.size match {
        case 0 => None
        case 1 => Some(s"$name field '$str' is missing")
        case _ => Some(s"$name fields '$str' are missing")
      }
    }
    val vs = missingMsg("view", viewFields)
    val ts = missingMsg("target", targetFields)
    val reasons = (ts ++ vs).mkString(" and ").capitalize
    val suggestion = if (ts.nonEmpty) "\n  If the view *should* be non-total, try a 'PartialDataView'." else ""
    val msg = s"Viewing $target as $view is non-Total!\n  $reasons.\n  DataView used is $dataView.$suggestion"
    throw InvalidViewException(msg)
  }

  // TODO should this be moved to class Aggregate / can it be unified with Aggregate.bind?
  private def doBind[T: DataProduct, V <: Data](
    target:      T,
    view:        V,
    dataView:    DataView[T, V],
    writability: ViewWriteability
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val mapping = dataView.mapping(target, view)
    val total = dataView.total
    // Lookups to check the mapping results
    val viewFieldLookup: Map[Data, String] = getRecursiveFields(view, "_").toMap
    val targetContains:  Data => Boolean = implicitly[DataProduct[T]].dataSet(target)

    // Resulting bindings for each Element of the View
    // Kept separate from Aggregates for totality checking
    val elementBindings =
      new mutable.LinkedHashMap[Data, mutable.ListBuffer[Element]] ++
        getRecursiveFields
          .lazilyNoPath(view)
          .collect { case (elt: Element) => elt }
          .map(_ -> new mutable.ListBuffer[Element])

    // Record any Aggregates that correspond 1:1 for reification
    // Using Data instead of Aggregate to avoid unnecessary checks
    val aggregateMappings = mutable.ArrayBuffer.empty[(Data, Data)]

    def viewFieldName(d: Data): String =
      viewFieldLookup.get(d).map(_ + " ").getOrElse("") + d.toString

    // Helper for recording the binding of each
    def onElt(te: Element, ve: Element): Unit = {
      // TODO can/should we aggregate these errors?
      def err(name: String, arg: Data) =
        throw InvalidViewException(s"View mapping must only contain Elements within the $name, got $arg")

      // The elements may themselves be views, look through the potential chain of views for the Elements
      // that are actually members of the target or view
      val tex = unfoldView(te).find(x => targetContains(x) || x.isLit || x == DontCare).getOrElse(err("Target", te))
      val vex = unfoldView(ve).find(viewFieldLookup.contains).getOrElse(err("View", ve))
      if (!tex.isSynthesizable) {
        Builder.exception(s".viewAs should only be called on hardware")
      }

      (tex, vex) match {
        /* Allow views where the types are equal. */
        case (a, b) if a.getClass == b.getClass =>
          // View width must be unknown or match target width
          if (vex.widthKnown && vex.width != tex.width) {
            def widthAsString(x: Element) = x.widthOption.map("<" + _ + ">").getOrElse("<unknown>")
            val fieldName = viewFieldName(vex)
            val vwidth = widthAsString(vex)
            val twidth = widthAsString(tex)
            throw InvalidViewException(
              s"View field $fieldName has width ${vwidth} that is incompatible with target value $tex's width ${twidth}"
            )
          }
        /* allow bool <=> reset views. */
        case (a: Bool, _: Reset) =>
        case (_: Reset, a: Bool) =>
        /* Allow AsyncReset <=> Reset views. */
        case (a: AsyncReset, _: Reset) =>
        case (_: Reset, a: AsyncReset) =>
        /* Allow DontCare in the target only */
        case (DontCare, _) =>
        /* Allow Property[_] <=> Property[_] views when the underlying type is the same */
        case (a: Property[_], b: Property[_]) if a.getPropertyType == b.getPropertyType =>
        /* All other views produce a runtime error. */
        case _ =>
          val fieldName = viewFieldName(vex)
          throw InvalidViewException(s"Field $fieldName specified as view of non-type-equivalent value $tex")
      }

      elementBindings(vex) += tex
    }

    mapping.foreach {
      // Special cased because getMatchedFields checks typeEquivalence on Elements (and is used in Aggregate path)
      // Also saves object allocations on common case of Elements
      case (ae: Element, be: Element) => onElt(ae, be)

      case (aa: Aggregate, ba: Aggregate) =>
        if (!ba.typeEquivalent(aa)) {
          val fieldName = viewFieldLookup(ba)
          val reason = ba
            .findFirstTypeMismatch(aa, strictTypes = true, strictWidths = true, strictProbeInfo = true)
            .map(s => s"\nbecause $s")
            .getOrElse("")
          throw InvalidViewException(
            s"Field $fieldName specified as view of non-type-equivalent value $aa$reason"
          )
        }
        getMatchedFields(aa, ba).foreach {
          case (aelt: Element, belt: Element) => onElt(aelt, belt)
          case (t, v)                         => aggregateMappings += (v -> t)
        }
    }

    // Errors in totality of the View, use var List to keep fast path cheap (no allocation)
    var viewNonTotalErrors:   List[Data] = Nil
    var targetNonTotalErrors: List[String] = Nil

    val targetSeen: Option[mutable.Set[Data]] = if (total) Some(mutable.Set.empty[Data]) else None

    val elementResult = elementBindings.map { case (data, targets) =>
      val targetsx = targets match {
        case collection.Seq(target: Element) => target
        case collection.Seq() =>
          viewNonTotalErrors = data :: viewNonTotalErrors
          data.asInstanceOf[Element] // Return the Data itself, will error after this map, cast is safe
        case x =>
          throw InvalidViewException(s"Got $x, expected Seq(_: Direct)")
      }
      // TODO record and report aliasing errors
      targetSeen.foreach(_ += targetsx)
      data -> targetsx
    }.toMap

    // Check for totality of Target
    targetSeen.foreach { seen =>
      val lookup = implicitly[DataProduct[T]].dataIterator(target, "_")
      for (missed <- lookup.collect { case (d: Element, name) if !seen(d) => name }) {
        targetNonTotalErrors = missed :: targetNonTotalErrors
      }
    }
    if (viewNonTotalErrors != Nil || targetNonTotalErrors != Nil) {
      val viewErrors = viewNonTotalErrors.map(f => viewFieldLookup.getOrElse(f, f.toString))
      nonTotalViewException(dataView, target, view, targetNonTotalErrors, viewErrors)
    }

    view match {
      case elt: Element => view.bind(ViewBinding(elementResult(elt), writability))
      case agg: Aggregate =>
        val fullResult = elementResult ++ aggregateMappings
        val aggWritability = Option.when(writability.isReadOnly)(
          Map((agg: Data) -> writability)
        )
        agg.bind(AggregateViewBinding(fullResult, aggWritability))
    }
  }

  // When annotating views that are not identity mappings, we need to record them for renaming
  // Technically, this adds any Aggregate that is not an identity mapping,
  // but we don't have a cheap way to check for single-target.
  private[chisel3] def recordViewForRenaming(view: Data): Unit = {
    view.topBinding match {
      case _: ViewBinding => () // No need for renaming
      case AggregateViewBinding(mapping, _) =>
        getRecursiveFields.lazily(view, "_").foreach {
          case (unnamed: Aggregate, _) if !mapping.contains(unnamed) =>
            Builder.unnamedViews += unnamed
          case _ => () // Do nothing
        }
      case _ =>
        Builder.exception(s"Internal Error! recordViewForRenaming($view)")(UnlocatableSourceInfo)
    }
  }

  // Traces an Element that may (or may not) be a view until it no longer maps
  // Inclusive of the argument
  // Note that this does *not* include writability so do not use this in place of reify.
  private def unfoldView(elt: Element): LazyList[Element] = {
    def rec(e: Element): LazyList[Element] = e.topBindingOpt match {
      case Some(ViewBinding(target, _)) => target #:: rec(target)
      case Some(avb: AggregateViewBinding) =>
        val target = avb.lookup(e).get
        target #:: rec(target)
      case Some(_) | None => LazyList.empty
    }
    elt #:: rec(elt)
  }

  // Safe for all Data
  private[chisel3] def isView(d: Data): Boolean = d._parent.exists(_ == ViewParent)

  /** Turn any [[Element]] that could be a View into a concrete Element
    *
    * This is the fundamental "unwrapping" or "tracing" primitive operation for handling Views within
    * Chisel.
    */
  private[chisel3] def reify(elt: Element): (Element, ViewWriteability) =
    reify(elt, elt.topBinding, ViewWriteability.Default)

  /** Turn any [[Element]] that could be a View into a concrete Element
    *
    * This is the fundamental "unwrapping" or "tracing" primitive operation for handling Views within
    * Chisel.
    */
  @tailrec private[chisel3] def reify(
    elt:        Element,
    topBinding: TopBinding,
    wrAcc:      ViewWriteability
  ): (Element, ViewWriteability) = {
    topBinding match {
      case ViewBinding(target, writeability) =>
        reify(target, target.topBinding, wrAcc.combine(writeability))
      case _ => (elt, wrAcc)
    }
  }

  /** Determine the target of a View if the view is an identity mapping.
    *
    * This is only true if the target of the view is of the same type and fields correspond 1:1.
    * For example, it would *not* be an identity view to view a Vec as a Vec with its elements in reverse order.
    *
    * @return The identity target of this view or None if not an identity view.
    * @note Returns Some(_) of the argument if it is not a view.
    */
  private[chisel3] def reifyIdentityView[T <: Data](
    data:  T,
    wrAcc: ViewWriteability = ViewWriteability.Default
  ): Option[(T, ViewWriteability)] = {
    val candidate: Option[(Data, ViewWriteability)] =
      data.topBindingOpt match {
        case None                                       => None
        case Some(ViewBinding(target, wr))              => Some(target -> wr)
        case Some(vb @ AggregateViewBinding(lookup, _)) => lookup.get(data).map(_ -> vb.lookupWritability(data))
        case Some(_)                                    => Some(data -> ViewWriteability.Default)
      }
    candidate.flatMap { case (d, wr) =>
      val wrx = wrAcc.combine(wr)
      // This cast is safe by construction, we only put Data in the view mapping if it is an identity mapping.
      val cast = d.asInstanceOf[T]
      // Candidate may itself be a view, keep tracing in those cases.
      if (isView(d)) reifyIdentityView(cast, wrx) else Some(cast -> wrx)
    }
  }

  // Return all parents of a Data, including itself
  private def allParents(d: Data): List[Data] = d.binding match {
    case Some(ChildBinding(parent)) => d :: allParents(parent)
    case _                          => List(d)
  }

  /** Determine the target of a View if the view maps to a single `Data`.
    *
    * An Aggregate may be a view of `non-Data` (like a `Seq` or tuple) and thus
    * there is no single Data representing the Target and this function will return None.
    *
    * @return The single Data target of this view or None if a single Data doesn't exist.
    * @note Returns Some(_) of the argument if it is not a view.
    * @note You should never attempt to write the result of this function.
    */
  private[chisel3] def reifySingleTarget(data: Data): Option[Data] = {
    def err(msg: String) = throwException(s"Internal Error! $msg reifySingleTarget($data)")
    // Identity views are obviously single targets.
    // We ignore writability because the return of this function should never be written.
    reifyIdentityView(data).map(_._1).orElse {
      // Otherwise, all children of data need to map to all of the children of another Data.
      // This is really expensive, is there a better way?
      data.topBindingOpt.flatMap {
        case AggregateViewBinding(mapping, _) =>
          // Take every single leaf and map to its target
          val leaves = DataMirror.collectLeafMembers(data)
          val targets = leaves.map { l =>
            // All leaves are stored in the mapping.
            val oneLevel = mapping(l)
            // This .get is safe because collectLeafMembers returns leaves.
            // It is of type Data (not Element) because of Probes, but Probes are atomic.
            reifySingleTarget(oneLevel).get
          }
          // Now, if there are any targets, check if all of the targets share a common parent,
          // and if that parent is exclusively composed of these targets.
          val tset = targets.toSet
          targets.headOption.flatMap { head =>
            allParents(head).find {
              // This is kind of a hack but ClonePorts is itself a hack.
              // We must ignore ClonePorts because it isn't a real Data, so it cannot be the "Single Target" to which we map.
              case _: ClonePorts => false
              case p =>
                val pset = DataMirror.collectLeafMembers(p).toSet
                pset == tset
            }
          }
        // Anything else should've been handled by reifyIdentityView
        case bad => err(s"This should not be reachable. Got binding = $bad in")

      }
    }
  }
}
