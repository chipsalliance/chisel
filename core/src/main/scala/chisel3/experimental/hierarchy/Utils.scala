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
  // TODO This is wrong, the standIn is T, but I'm calling the proto T...
  def toUnderlyingAsInstance[T <: BaseModule](module: T): Proxy[BaseModule] = module match {
    case i: IsStandIn[BaseModule] => StandIn(i)
    case other: T => Proto(other, other._parent.map{ case p: BaseModule with IsContext => toUnderlyingAsInstance(p).asInstanceOf[Proxy[IsContext]]})
  }
  def toUnderlyingAsDefinition[T <: BaseModule](module: T): Proxy[BaseModule] = {
    module match {
      case i: IsStandIn[BaseModule] =>
        StandIn(StandInDefinition(i.proto, module.getCircuit))
      case other: T =>
        StandIn(StandInDefinition(other, module.getCircuit))
    }
  }
  //def getInnerDataContext[T](h: Hierarchy[T])(implicit tc: Contexter[T]): Option[BaseModule] = tc.lookupContext(h) match {
  //  case Some(StandIn(standin: BaseModule)) => Some(standin)
  //  case Some(Proto(proto: BaseModule, _)) => Some(proto)
  //  case Some(StandIn(StandInIsInstantiable(p, parent: Option[BaseModule]))) => parent
  //  case other => None
  //} 
  //h match {
  //  case d: Definition[_] => 
  //  case i: Instance[_] =>
  //  case value: BaseModule =>
  //    val newChild = Module.do_pseudo_apply(new StandInDefinition(value))(
  //      chisel3.internal.sourceinfo.UnlocatableSourceInfo,
  //      chisel3.ExplicitCompileOptions.Strict
  //    )
  //    newChild._circuit = value._circuit.orElse(Some(value))
  //    newChild._parent = None
  //    Some(newChild)
  //  case value: IsContext => None
  //}
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
  /** Clones a data and sets its internal references to its parent module to be in a new context.
    *
    * @param data data to be cloned
    * @param context new context
    * @return
    */
  def cloneDataToContext[T <: Data](
    data:    T,
    context: BaseModule
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    //println(s"HERE!!!! $data")
    internal.requireIsHardware(data, "cross module reference type")
    val ret = data._parent match {
      case None => data
      case Some(parent) =>
        //println(s"OLD PARENT: $parent")
        //println(s"CONTEXT: $context")
        val newParent = cloneModuleToContext(Proto(parent, parent._parent.map(toUnderlyingAsInstance)), context)
        //println(s"NEW PARENT: $newParent")
        newParent match {
          case Proto(p, _) if p == parent => data
          case StandIn(m: BaseModule) =>
            val newChild = data.cloneTypeFull
            newChild.setRef(data.getRef, true)
            newChild.bind(internal.CrossModuleBinding)
            newChild.setAllParents(Some(m))
            newChild
        }
    }
    //println(s"THERE!!!! $ret, ${ret._parent}")
    ret
  }
  // Helper for co-iterating on Elements of aggregates, they must be the same type but that is unchecked
  def coiterate(a: Data, b: Data): Iterable[(Element, Element)] = {
    val as = getRecursiveFields.lazily(a, "_")
    val bs = getRecursiveFields.lazily(b, "_")
    as.zip(bs).collect { case ((ae: Element, _), (be: Element, _)) => (ae, be) }
  }

  def doLookupData[A, B <: Data](
    data:    B,
    ioMap:   Option[Map[Data, Data]],
    self: BaseModule
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): B = {
    def impl[C <: Data](d: C): C = d match {
      case x: Data if ioMap.nonEmpty && ioMap.get.contains(x) => ioMap.get(x).asInstanceOf[C]
      case _ => cloneDataToContext(d, self)
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

  /** Given a module (either original or a clone), clone it to a new context
    *
    * This function effectively recurses up the parents of module to find whether:
    *   (1) A parent is already in the context; then we do nothing and return module
    *   (2) A parent is in a different clone of the context; then we clone all the parents up
    *         to that parent and set their parents to be in this underlying context
    *   (3) A parent has no root; in that case, we do nothing and return the module.
    *
    * @param module original or clone to be underlying into a new context
    * @param context new context
    * @return original or clone in the new context
    */
  private[chisel3] def cloneModuleToContext[T <: BaseModule](
    module:  Proxy[T],
    context: BaseModule // TODO! This needs to be Hierarchy, so we can get the contexts and pass to StandInInstance, so they are accessible in the Select operators.
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Proxy[T] = {
    // Recursive call
    def rec[A <: BaseModule](m: A): Proxy[A] = {
      def clone(x: A, p: Option[BaseModule], name: () => String): Proxy[A] = {
        val newChild = x match {
          case ic: IsStandIn[ _] => Module.do_pseudo_apply(new StandInInstance(x, name, p))
          case other => Module.do_pseudo_apply(new StandInInstance(other, name, p))
        }
        StandIn(newChild)
      }
      (m, context) match {
        case (c, ctx) if ctx == c => Proto(c, c._parent.map(toUnderlyingAsInstance))
        //case (c, ctx: IsStandIn[_]) if ctx.hasSameProto(c) => Clone(ctx.asInstanceOf[IsStandIn[A]])
        case (c, ctx: IsStandIn[ _]) if ctx.hasSameProto(c) && (c._parent.isEmpty || ctx._parent.isEmpty) =>
          //println(s"Matched ctx with empty parent: $c, $ctx")
          StandIn(ctx.asInstanceOf[IsStandIn[A]])
        case (c, ctx: IsStandIn[ _]) if ctx.hasSameProto(c) =>
          //println(s"Matched ctx")
          cloneModuleToContext(toUnderlyingAsInstance(m._parent.get), ctx._parent.get) match {
            case Proto(p, _) => Proto(m, m._parent.map(toUnderlyingAsInstance))
            case StandIn(p: BaseModule) =>
              //println(s"Cloning2: $m, $p")
              clone(m, Some(p), () => m.instanceName)
          }
        case (c, ctx) if c._parent.isEmpty => Proto(c, c._parent.map(toUnderlyingAsInstance))
        case (_, _) =>
          //println(s"Rec: ${m._parent.get.toTarget}, ${context}")
          cloneModuleToContext(toUnderlyingAsInstance(m._parent.get), context) match {
            case Proto(p, _) => Proto(m, m._parent.map(toUnderlyingAsInstance))
            case StandIn(p: BaseModule) =>
              //println(s"Cloning: $m, $p")
              clone(m, Some(p), () => m.instanceName)
          }
      }
    }
    module match {
      case Proto(m, parent) => rec(m)
      case StandIn(m: experimental.hierarchy.StandInModule[T]) =>
        //println("experimental.hierarchy.StandInModule")
        rec(m) match {
          case Proto(mx, _) => StandIn(mx)
          case StandIn(i: StandInInstance[T]) =>
            val newChild = Module.do_pseudo_apply(new StandInInstance(m.proto, () => m.instanceName, i._parent))
            StandIn(newChild)
        }
      case StandIn(m: StandInInstance[_]) =>
        //println("StandInInstance")
        rec(m) match {
          case Proto(mx, _) => StandIn(mx)
          case StandIn(i: StandInInstance[_]) =>
            val newChild = Module.do_pseudo_apply(new StandInInstance(m.proto, () => m.instanceName, i.parent))
            StandIn(newChild)
        }
    }
  }



  def cloneMemToContext[T <: MemBase[_]](
    mem:     T,
    context: BaseModule
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    mem._parent match {
      case None => mem
      case Some(parent) =>
        val newParent = cloneModuleToContext(toUnderlyingAsInstance(parent), context)
        newParent match {
          case Proto(p, _) if p == parent => mem
          case StandIn(mod: BaseModule) =>
            val existingMod = Builder.currentModule
            Builder.currentModule = Some(mod)
            val newChild: T = mem match {
              case m: Mem[_] => new Mem(m.t.asInstanceOf[Data].cloneTypeFull, m.length).asInstanceOf[T]
              case m: SyncReadMem[_] =>
                new SyncReadMem(m.t.asInstanceOf[Data].cloneTypeFull, m.length, m.readUnderWrite).asInstanceOf[T]
            }
            Builder.currentModule = existingMod
            newChild.setRef(mem.getRef, true)
            newChild
        }
    }
  }
}