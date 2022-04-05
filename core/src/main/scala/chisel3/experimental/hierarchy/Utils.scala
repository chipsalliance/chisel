package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.SourceInfo

import scala.annotation.implicitNotFound
import scala.collection.mutable.HashMap
import chisel3._
import chisel3.experimental.dataview.{isView, reify, reifySingleData}
import chisel3.internal.firrtl.{Arg, ILit, Index, Slot, ULit}
import chisel3.internal.{throwException, AggregateViewBinding, Builder, ChildBinding, ViewBinding, ViewParent}
import chisel3.experimental.hierarchy.core._

private[chisel3] object Utils {

  /** Given a Data, find the root of its binding, apply a function to the root to get a "new root",
    * and find the equivalent child Data in the "new root"
    *
    * @example {{{
    * Given `arg = a.b[2].c` and some `f`:
    * 1. a = root(arg) = root(a.b[2].c)
    * 2. newRoot = f(root(arg)) = f(a)
    * 3. return newRoot.b[2].c
    * }}}
    *
    * Invariants that elt is a Child of something of the type of data is dynamically checked as we traverse
    */
  def mapRootAndExtractSubField[A <: Data](arg: A, f: Data => Data): A = {
    def err(msg:               String) = throwException(s"Internal Error! $msg")
    def unrollCoordinates(res: List[Arg], d: Data): (List[Arg], Data) = d.binding.get match {
      case ChildBinding(parent) =>
        d.getRef match {
          case arg @ (_: Slot | _: Index) => unrollCoordinates(arg :: res, parent)
          case other => err(s"Unroll coordinates failed for '$arg'! Unexpected arg '$other'")
        }
      case _ => (res, d)
    }
    def applyCoordinates(fullCoor: List[Arg], start: Data): Data = {
      def rec(coor: List[Arg], d: Data): Data = {
        if (coor.isEmpty) d
        else {
          val next = (coor.head, d) match {
            case (Slot(_, name), rec: Record) => rec.elements(name)
            case (Index(_, ILit(n)), vec: Vec[_]) => vec.apply(n.toInt)
            case (arg, _) => err(s"Unexpected Arg '$arg' applied to '$d'! Root was '$start'.")
          }
          applyCoordinates(coor.tail, next)
        }
      }
      rec(fullCoor, start)
    }
    val (coor, root) = unrollCoordinates(Nil, arg)
    val newRoot = f(root)
    val result = applyCoordinates(coor, newRoot)
    try {
      result.asInstanceOf[A]
    } catch {
      case _: ClassCastException => err(s"Applying '$coor' to '$newRoot' somehow resulted in '$result'")
    }
  }
  // Helper for co-iterating on Elements of aggregates, they must be the same type but that is unchecked
  def coiterate(a: Data, b: Data): Iterable[(Element, Element)] = {
    val as = getRecursiveFields.lazily(a, "_")
    val bs = getRecursiveFields.lazily(b, "_")
    as.zip(bs).collect { case ((ae: Element, _), (be: Element, _)) => (ae, be) }
  }

  def doLookupData[A, B <: Data](
    data:  B,
    ioMap: Option[Map[Data, Data]],
    self:  BaseModule
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): B = {
    def impl[C <: Data](d: C): C = d match {
      case x: Data if ioMap.nonEmpty && ioMap.get.contains(x) => ioMap.get(x).asInstanceOf[C]
      case _ => ??? // cloneDataToContext(d, self)
    }
    data.binding match {
      case Some(_: ChildBinding) => mapRootAndExtractSubField(data, impl)
      case _ => impl(data)
    }
  }

  // TODO this logic is complicated, can any of it be unified with viewAs?
  // If `.viewAs` would capture its arguments, we could potentially use it
  // TODO Describe what this is doing at a high level
  def cloneViewToContext[A, B <: Data](
    data:    B,
    ioMap:   Option[Map[Data, Data]],
    context: Option[BaseModule]
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): B = {
    // alias to shorten lookups
    def lookupData[C <: Data](d: C) = doLookupData(d, ioMap, context.get)

    val result = data.cloneTypeFull

    // We have to lookup the target(s) of the view since they may need to be underlying into the current context
    val newBinding = data.topBinding match {
      case ViewBinding(target) => ViewBinding(lookupData(reify(target)))
      case avb @ AggregateViewBinding(map, targetOpt) =>
        data match {
          case _: Element   => ViewBinding(lookupData(reify(map(data))))
          case _: Aggregate =>
            // Provide a 1:1 mapping if possible
            val singleTargetOpt = targetOpt.filter(_ => avb == data.binding.get).flatMap(reifySingleData)
            singleTargetOpt match {
              case Some(singleTarget) => // It is 1:1!
                // This is a little tricky because the values in newMap need to point to Elements of newTarget
                val newTarget = lookupData(singleTarget)
                val newMap = coiterate(result, data).map {
                  case (res, from) =>
                    (res: Data) -> mapRootAndExtractSubField(map(from), _ => newTarget)
                }.toMap
                AggregateViewBinding(newMap, Some(newTarget))

              case None => // No 1:1 mapping so we have to do a flat binding
                // Just remap each Element of this aggregate
                val newMap = coiterate(result, data).map {
                  // Upcast res to Data since Maps are invariant in the Key type parameter
                  case (res, from) => (res: Data) -> lookupData(reify(map(from)))
                }.toMap
                AggregateViewBinding(newMap, None)
            }
        }
    }

    // TODO Unify the following with `.viewAs`
    // We must also mark non-1:1 and child Aggregates in the view for renaming
    newBinding match {
      case _: ViewBinding => // Do nothing
      case AggregateViewBinding(_, target) =>
        if (target.isEmpty) {
          Builder.unnamedViews += result
        }
        // Binding does not capture 1:1 for child aggregates views
        getRecursiveFields.lazily(result, "_").foreach {
          case (agg: Aggregate, _) if agg != result =>
            Builder.unnamedViews += agg
          case _ => // Do nothing
        }
    }

    result.bind(newBinding)
    result.setAllParents(Some(ViewParent))
    result.forceName(None, "view", Builder.viewNamespace)
    result
  }

}
