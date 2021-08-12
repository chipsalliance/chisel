// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal._
import chisel3.internal.sourceinfo.SourceInfo

import scala.annotation.{implicitNotFound, tailrec}
import scala.collection.mutable
import scala.collection.immutable.LazyList // Needed for 2.12 alias

package object dataview {
  case class InvalidViewException(message: String) extends ChiselException(message)

  /** Provides `viewAs` for types that have an implementation of [[DataProduct]]
    *
    * Calling `viewAs` also requires an implementation of [[DataView]] for the target type
    */
  implicit class DataViewable[T](target: T) {
    def viewAs[V <: Data](implicit dataproduct: DataProduct[T], dataView: DataView[T, V]): V = {
      // TODO put a try catch here for ExpectedHardwareException and perhaps others
      // It's likely users will accidentally use chiselTypeOf or something that may error,
      // The right thing to use is DataMirror...chiselTypeClone because of composition with DataView.andThen
      // Another option is that .andThen could give a fake binding making chiselTypeOfs in the user code safe
      val result: V = dataView.mkView(target)
      requireIsChiselType(result, "viewAs")

      doBind(target, result, dataView)

      // Setting the parent marks these Data as Views
      result.setAllParents(Some(ViewParent))
      // The names of views do not matter except for when a view is annotated. For Views that correspond
      // To a single Data, we just forward the name of the Target. For Views that correspond to more
      // than one Data, we return this assigned name but rename it in the Convert stage
      result.forceName(None, "view", Builder.viewNamespace)
      result
    }
  }

  // This private type alias lets us provide a custom error message for misuing the .viewAs for upcasting Bundles
  @implicitNotFound("${A} is not a subtype of ${B}! Did you mean .viewAs[${B}]? " +
    "Please see https://www.chisel-lang.org/chisel3/docs/cookbooks/dataview")
  private type SubTypeOf[A, B] = A <:< B

  /** Provides `viewAsSupertype` for subclasses of [[Bundle]] */
  implicit class BundleUpcastable[T <: Bundle](target: T) {
    /** View a [[Bundle]] or [[Record]] as a parent type (upcast) */
    def viewAsSupertype[V <: Bundle](proto: V)(implicit ev: SubTypeOf[T, V], sourceInfo: SourceInfo): V = {
      implicit val dataView = PartialDataView.mapping[T, V](_ => proto, {
        case (a, b) =>
          val aElts = a.elements
          val bElts = b.elements
          val bKeys = bElts.keySet
          val keys = aElts.keysIterator.filter(bKeys.contains)
          keys.map(k => aElts(k) -> bElts(k)).toSeq
      })
      target.viewAs[V]
    }
  }

  private def nonTotalViewException(dataView: DataView[_, _], target: Any, view: Data, targetFields: Seq[String], viewFields: Seq[String]) = {
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
  private def doBind[T : DataProduct, V <: Data](target: T, view: V, dataView: DataView[T, V]): Unit = {
    val mapping = dataView.mapping(target, view)
    val total = dataView.total
    // Lookups to check the mapping results
    val viewFieldLookup: Map[Data, String] = getRecursiveFields(view, "_").toMap
    val targetContains: Data => Boolean = implicitly[DataProduct[T]].dataSet(target)

    // Resulting bindings for each Element of the View
    val childBindings =
      new mutable.HashMap[Data, mutable.ListBuffer[Element]] ++
        viewFieldLookup.view
          .collect { case (elt: Element, _) => elt }
          .map(_ -> new mutable.ListBuffer[Element])

    def viewFieldName(d: Data): String =
      viewFieldLookup.get(d).map(_ + " ").getOrElse("") + d.toString

    // Helper for recording the binding of each
    def onElt(te: Element, ve: Element): Unit = {
      // TODO can/should we aggregate these errors?
      def err(name: String, arg: Data) =
        throw InvalidViewException(s"View mapping must only contain Elements within the $name, got $arg")

      // The elements may themselves be views, look through the potential chain of views for the Elements
      // that are actually members of the target or view
      val tex = unfoldView(te).find(targetContains).getOrElse(err("Target", te))
      val vex = unfoldView(ve).find(viewFieldLookup.contains).getOrElse(err("View", ve))

      if (tex.getClass != vex.getClass) {
        val fieldName = viewFieldName(vex)
        throw InvalidViewException(s"Field $fieldName specified as view of non-type-equivalent value $tex")
      }
      // View width must be unknown or match target width
      if (vex.widthKnown && vex.width != tex.width) {
        def widthAsString(x: Element) = x.widthOption.map("<" + _ + ">").getOrElse("<unknown>")
        val fieldName = viewFieldName(vex)
        val vwidth = widthAsString(vex)
        val twidth = widthAsString(tex)
        throw InvalidViewException(s"View field $fieldName has width ${vwidth} that is incompatible with target value $tex's width ${twidth}")
      }
      childBindings(vex) += tex
    }

    mapping.foreach {
      // Special cased because getMatchedFields checks typeEquivalence on Elements (and is used in Aggregate path)
      // Also saves object allocations on common case of Elements
      case (ae: Element, be: Element) => onElt(ae, be)

      case (aa: Aggregate, ba: Aggregate) =>
        if (!ba.typeEquivalent(aa)) {
          val fieldName = viewFieldLookup(ba)
          throw InvalidViewException(s"field $fieldName specified as view of non-type-equivalent value $aa")
        }
        getMatchedFields(aa, ba).foreach {
          case (aelt: Element, belt: Element) => onElt(aelt, belt)
          case _ => // Ignore matching of Aggregates
        }
    }

    // Errors in totality of the View, use var List to keep fast path cheap (no allocation)
    var viewNonTotalErrors: List[Data] = Nil
    var targetNonTotalErrors: List[String] = Nil

    val targetSeen: Option[mutable.Set[Data]] = if (total) Some(mutable.Set.empty[Data]) else None

    val resultBindings = childBindings.map { case (data, targets) =>
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
      case elt: Element => view.bind(ViewBinding(resultBindings(elt)))
      case agg: Aggregate =>
        // We record total Data mappings to provide a better .toTarget
        val topt = target match {
          case d: Data if total => Some(d)
          case _ =>
            // Record views that don't have the simpler .toTarget for later renaming
            Builder.unnamedViews += view
            None
        }
        // TODO We must also record children as unnamed, some could be namable but this requires changes to the Binding
        getRecursiveFields.lazily(view, "_").foreach {
          case (agg: Aggregate, _) if agg != view =>
            Builder.unnamedViews += agg
          case _ => // Do nothing
        }
        agg.bind(AggregateViewBinding(resultBindings, topt))
    }
  }

  // Traces an Element that may (or may not) be a view until it no longer maps
  // Inclusive of the argument
  private def unfoldView(elt: Element): LazyList[Element] = {
    def rec(e: Element): LazyList[Element] = e.topBindingOpt match {
      case Some(ViewBinding(target)) => target #:: rec(target)
      case Some(AggregateViewBinding(mapping, _)) =>
        val target = mapping(e)
        target #:: rec(target)
      case Some(_) | None => LazyList.empty
    }
    elt #:: rec(elt)
  }

  // Safe for all Data
  private[chisel3] def isView(d: Data): Boolean = d._parent.contains(ViewParent)

  /** Turn any [[Element]] that could be a View into a concrete Element
    *
    * This is the fundamental "unwrapping" or "tracing" primitive operation for handling Views within
    * Chisel.
    */
  private[chisel3] def reify(elt: Element): Element =
    reify(elt, elt.topBinding)

  /** Turn any [[Element]] that could be a View into a concrete Element
    *
    * This is the fundamental "unwrapping" or "tracing" primitive operation for handling Views within
    * Chisel.
    */
  @tailrec private[chisel3] def reify(elt: Element, topBinding: TopBinding): Element =
    topBinding match {
      case ViewBinding(target) => reify(target, elt.topBinding)
      case _ => elt
    }

    /** Determine the target of a View if it is a single Target
      *
      * @note An Aggregate may be a view of unrelated [[Data]] (eg. like a Seq or tuple) and thus this
      *       there is no single Data representing the Target and this function will return None
      * @return The single Data target of this view or None if a single Data doesn't exist
      */
    private[chisel3] def reifySingleData(data: Data): Option[Data] = {
      val candidate: Option[Data] =
        data.binding.collect { // First check if this is a total mapping of an Aggregate
          case AggregateViewBinding(_, Some(t)) => t
        }.orElse { // Otherwise look via top binding
          data.topBindingOpt match {
            case None => None
            case Some(ViewBinding(target)) => Some(target)
            case Some(AggregateViewBinding(lookup, _)) => lookup.get(data)
            case Some(_) => None
          }
        }
      candidate.flatMap { d =>
        // Candidate may itself be a view, keep tracing in those cases
        if (isView(d)) reifySingleData(d) else Some(d)
      }
    }

}
