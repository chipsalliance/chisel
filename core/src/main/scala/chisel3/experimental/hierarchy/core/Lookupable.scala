// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.experimental.hierarchy.{InstanceClone, InstantiableClone, ModuleClone}

import scala.annotation.implicitNotFound
import scala.collection.mutable.HashMap
import chisel3._
import chisel3.experimental.dataview.{isView, reify, reifyIdentityView}
import chisel3.internal.firrtl.ir.{Arg, ILit, Index, LitIndex, ModuleIO, Slot, ULit}
import chisel3.internal.{throwException, Builder, ViewParent}
import chisel3.internal.binding.{AggregateViewBinding, ChildBinding, CrossModuleBinding, ViewBinding, ViewWriteability}

/** Typeclass used to recontextualize values from an original Definition to an Instance
  *
  * Implementations for Chisel types and other common Scala types are provided in the companion object.
  * Users can define Lookupable for their own types.
  *
  * @example {{{
  * case class Foo[T <: Data](name: String, data: T)
  * object Foo {
  *   implicit def lookupable[T <: Data]: Lookupable.Simple[Foo[T]] =
  *     Lookupable.product2[Foo[T], String, T](
  *       x => (x.name, x.data),
  *       Foo.apply
  *     )
  * }
  * }}}
  */
@implicitNotFound(
  "@public is only legal within a class or trait marked @instantiable, and only on vals of types" +
    " that have a Lookupable implementation. Chisel types like Data, BaseModule, and MemBase are supported," +
    " as are common Scala types like String, Int, Boolean, Iterable, Option, Either, and Tuples." +
    " Please implement Lookupable for ${B}."
)
@deprecatedInheritance(
  "Users should use Lookupable factory methods instead (e.g. Lookupable.isLookupable or Lookupable.product<n>)",
  "Chisel 7.0.0"
)
trait Lookupable[-B] {
  type C // Return type of the lookup
  /** Function called to modify the returned value of type B from A, into C
    *
    * @param that function that selects B from A
    * @param instance Instance of A, used to determine C's context
    * @return
    */
  def instanceLookup[A](that: A => B, instance: Instance[A]): C

  /** Function called to modify the returned value of type B from A, into C
    *
    * @param that function that selects B from A
    * @param definition Definition of A, used to determine C's context
    * @return
    */
  def definitionLookup[A](that:     A => B, definition: Definition[A]): C
  protected def getProto[A](h:      Hierarchy[A]):                      A = h.proto
  protected def getUnderlying[A](h: Hierarchy[A]):                      Underlying[A] = h.underlying

  // Single method that may eventually replace instanceLookup and definitionLookup.
  private[chisel3] def hierarchyLookup[A](that: A => B, hierarchy: Hierarchy[A]): C = {
    hierarchy match {
      case h: Instance[A]   => instanceLookup(that, h)
      case h: Definition[A] => definitionLookup(that, h)
      case h => throw new InternalErrorException(s"Match error: hierarchy=$hierarchy")
    }
  }
}
// Simplify the implementation of Lookupable by only having 1 virtual method.
// Only the Chisel core types need separate implementations for Instance and Definition.
private trait LookupableImpl[-B] extends Lookupable[B] {

  protected def impl[A](that: A => B, hierarchy: Hierarchy[A]): C

  override def instanceLookup[A](that:   A => B, instance:   Instance[A]):   C = impl(that, instance)
  override def definitionLookup[A](that: A => B, definition: Definition[A]): C = impl(that, definition)
}

object Lookupable {

  /** Type alias for simplifying explicit Lookupable type ascriptions */
  type Aux[B, C0] = Lookupable[B] { type C = C0 }

  /** Type alias for simple Lookupable types */
  type Simple[B] = Aux[B, B]

  /** Provides a lookupable for a type X that does not need special handling
    *
    * This can only be used by types that do not contain a [[Data]], [[BaseModule]], or any [[IsInstantiable]].
    */
  def isLookupable[X]: Simple[X] = new LookupableImpl[X] {
    type C = X
    type B = X
    override protected def impl[A](that: A => B, hierarchy: Hierarchy[A]): C = that(hierarchy.proto)
  }

  /** Factory method for creating Lookupable for user-defined types
    */
  def product1[X, T1: Lookupable](
    in:  X => T1,
    out: T1 => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = new LookupableImpl[X] {
    type C = X
    override protected def impl[A](that: A => X, hierarchy: Hierarchy[A]): C = {
      val t1res = implicitly[Lookupable[T1]].hierarchyLookup[A](a => in(that(a)), hierarchy).asInstanceOf[T1]
      out(t1res)
    }
  }

  /** Factory method for creating Lookupable for user-defined types
    */
  def product2[X, T1: Lookupable, T2: Lookupable](
    in:  X => (T1, T2),
    out: (T1, T2) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = new LookupableImpl[X] {
    type C = X
    override protected def impl[A](that: A => X, hierarchy: Hierarchy[A]): C = {
      val t1res = implicitly[Lookupable[T1]].hierarchyLookup[A](a => in(that(a))._1, hierarchy).asInstanceOf[T1]
      val t2res = implicitly[Lookupable[T2]].hierarchyLookup[A](a => in(that(a))._2, hierarchy).asInstanceOf[T2]
      out(t1res, t2res)
    }
  }

  /** Factory method for creating Lookupable for user-defined types
    */
  def product3[X, T1: Lookupable, T2: Lookupable, T3: Lookupable](
    in:  X => (T1, T2, T3),
    out: (T1, T2, T3) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = new LookupableImpl[X] {
    type C = X
    override protected def impl[A](that: A => X, hierarchy: Hierarchy[A]): C = {
      val t1res = implicitly[Lookupable[T1]].hierarchyLookup[A](a => in(that(a))._1, hierarchy).asInstanceOf[T1]
      val t2res = implicitly[Lookupable[T2]].hierarchyLookup[A](a => in(that(a))._2, hierarchy).asInstanceOf[T2]
      val t3res = implicitly[Lookupable[T3]].hierarchyLookup[A](a => in(that(a))._3, hierarchy).asInstanceOf[T3]
      out(t1res, t2res, t3res)
    }
  }

  /** Factory method for creating Lookupable for user-defined types
    */
  def product4[X, T1: Lookupable, T2: Lookupable, T3: Lookupable, T4: Lookupable](
    in:  X => (T1, T2, T3, T4),
    out: (T1, T2, T3, T4) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = new LookupableImpl[X] {
    type C = X
    override protected def impl[A](that: A => X, hierarchy: Hierarchy[A]): C = {
      val t1res = implicitly[Lookupable[T1]].hierarchyLookup[A](a => in(that(a))._1, hierarchy).asInstanceOf[T1]
      val t2res = implicitly[Lookupable[T2]].hierarchyLookup[A](a => in(that(a))._2, hierarchy).asInstanceOf[T2]
      val t3res = implicitly[Lookupable[T3]].hierarchyLookup[A](a => in(that(a))._3, hierarchy).asInstanceOf[T3]
      val t4res = implicitly[Lookupable[T4]].hierarchyLookup[A](a => in(that(a))._4, hierarchy).asInstanceOf[T4]
      out(t1res, t2res, t3res, t4res)
    }
  }

  /** Factory method for creating Lookupable for user-defined types
    */
  def product5[X, T1: Lookupable, T2: Lookupable, T3: Lookupable, T4: Lookupable, T5: Lookupable](
    in:  X => (T1, T2, T3, T4, T5),
    out: (T1, T2, T3, T4, T5) => X
  )(
    implicit sourceInfo: SourceInfo
  ): Simple[X] = new LookupableImpl[X] {
    type C = X
    override protected def impl[A](that: A => X, hierarchy: Hierarchy[A]): C = {
      val t1res = implicitly[Lookupable[T1]].hierarchyLookup[A](a => in(that(a))._1, hierarchy).asInstanceOf[T1]
      val t2res = implicitly[Lookupable[T2]].hierarchyLookup[A](a => in(that(a))._2, hierarchy).asInstanceOf[T2]
      val t3res = implicitly[Lookupable[T3]].hierarchyLookup[A](a => in(that(a))._3, hierarchy).asInstanceOf[T3]
      val t4res = implicitly[Lookupable[T4]].hierarchyLookup[A](a => in(that(a))._4, hierarchy).asInstanceOf[T4]
      val t5res = implicitly[Lookupable[T5]].hierarchyLookup[A](a => in(that(a))._5, hierarchy).asInstanceOf[T5]
      out(t1res, t2res, t3res, t4res, t5res)
    }
  }

  /** Clones a data and sets its internal references to its parent module to be in a new context.
    *
    * @param data data to be cloned
    * @param context new context
    * @return
    */
  private[chisel3] def cloneDataToContext[T <: Data](
    data:    T,
    context: BaseModule
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    experimental.requireIsHardware(data, "cross module reference type")
    data._parent match {
      case None => data
      case Some(parent) =>
        val newParent = cloneModuleToContext(Proto(parent), context)
        newParent match {
          case Proto(p) if p == parent => data
          case Clone(m: BaseModule) =>
            val newChild = data.cloneTypeFull
            newChild.setRef(data.getRef, true)
            newChild.bind(CrossModuleBinding)
            newChild.setAllParents(Some(m))
            newChild
          case _ => throw new InternalErrorException(s"Match error: newParent=$newParent")
        }
    }
  }
  // The business logic of lookupData
  // Also called by cloneViewToContext which potentially needs to lookup stuff from ioMap or the cache
  private[chisel3] def doLookupData[A, B <: Data](
    data:    B,
    cache:   HashMap[Data, Data],
    ioMap:   Option[Map[Data, Data]],
    context: Option[BaseModule]
  )(
    implicit sourceInfo: SourceInfo
  ): B = {
    def impl[C <: Data](d: C): C = d match {
      case x: Data if !x.isSynthesizable                      => x.asInstanceOf[C]
      case x: Data if ioMap.nonEmpty && ioMap.get.contains(x) => ioMap.get(x).asInstanceOf[C]
      case x: Data if cache.contains(x)                       => cache(x).asInstanceOf[C]
      case _ =>
        assert(context.nonEmpty) // TODO is this even possible? Better error message here
        val ret = cloneDataToContext(d, context.get)
        cache(d) = ret
        ret
    }
    data.binding match {
      case Some(_: ChildBinding) => mapRootAndExtractSubField(data, impl)
      case _                     => impl(data)
    }
  }

  // Helper for co-iterating on all Elements of aggregates, they must be the same type but that is unchecked
  private def coiterate(a: Data, b: Data): Iterable[(Data, Data)] = {
    val as = getRecursiveFields.lazily(a, "_")
    val bs = getRecursiveFields.lazily(b, "_")
    as.zip(bs).collect { case ((ae: Data, _), (be: Data, _)) => (ae, be) }
  }

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
  private def mapRootAndExtractSubField[A <: Data](arg: A, f: Data => Data): A = {
    def err(msg: String) = throwException(s"Internal Error! $msg")
    def unrollCoordinates(res: List[Arg], d: Data): (List[Arg], Data) = d.binding.get match {
      case ChildBinding(parent) =>
        d.getRef match {
          case arg @ (_: Slot | _: Index | _: LitIndex | _: ModuleIO) => unrollCoordinates(arg :: res, parent)
          case other => err(s"unrollCoordinates failed for '$arg'! Unexpected arg '$other'")
        }
      case _ => (res, d)
    }
    def applyCoordinates(fullCoor: List[Arg], start: Data): Data = {
      def rec(coor: List[Arg], d: Data): Data = {
        if (coor.isEmpty) d
        else {
          val next = (coor.head, d) match {
            case (Slot(_, name), rec: Record)     => rec._elements(name)
            case (LitIndex(_, n), vec: Vec[_])    => vec.apply(n)
            case (Index(_, ILit(n)), vec: Vec[_]) => vec.apply(n.toInt)
            case (ModuleIO(_, name), rec: Record) => rec._elements(name)
            case (arg, _)                         => err(s"Unexpected Arg '$arg' applied to '$d'! Root was '$start'.")
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

  // TODO this logic is complicated, can any of it be unified with viewAs?
  // If `.viewAs` would capture its arguments, we could potentially use it
  // TODO Describe what this is doing at a high level
  private[chisel3] def cloneViewToContext[A, B <: Data](
    data:    B,
    cache:   HashMap[Data, Data],
    ioMap:   Option[Map[Data, Data]],
    context: Option[BaseModule]
  )(
    implicit sourceInfo: SourceInfo
  ): B = {
    // alias to shorten lookups
    def lookupData[C <: Data](d: C) = doLookupData(d, cache, ioMap, context)

    val result = data.cloneTypeFull

    // We have to lookup the target(s) of the view since they may need to be underlying into the current context
    val newBinding = data.topBinding match {
      case ViewBinding(target, writable1) =>
        val (reified, writable2) = reify(target)
        ViewBinding(lookupData(reified), writable1.combine(writable2))
      case avb @ AggregateViewBinding(map, _) =>
        data match {
          case e: Element =>
            val (reified, writable) = reify(avb.lookup(e).get)
            ViewBinding(lookupData(reified), avb.lookupWritability(e).combine(writable))
          case _: Aggregate =>
            // We could just call reifyIdentityView, but since we already have avb, it is
            // faster to just use it but then call reifyIdentityView in case the target is itself a view
            def reifyOpt(data: Data): Option[(Data, ViewWriteability)] = map.get(data).flatMap(reifyIdentityView(_))
            // Just remap each Data present in the map
            val mapping = coiterate(result, data).flatMap { case (res, from) =>
              reifyOpt(from).map { case (t, w) => (res, lookupData(t), w) }
            }
            val newMap = mapping.map { case (from, to, _) => from -> to }.toMap
            val wrMap = mapping.flatMap { case (from, _, wr) => Option.when(wr.isReadOnly)(from -> wr) }.toMap
            AggregateViewBinding(newMap, Option.when(wrMap.nonEmpty)(wrMap))
        }
      case _ => throw new InternalErrorException(s"Match error: data.topBinding=${data.topBinding}")
    }

    result.bind(newBinding)
    result.setAllParents(Some(ViewParent))
    result.forceName("view", Builder.viewNamespace)
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
    module:  Underlying[T],
    context: BaseModule
  )(
    implicit sourceInfo: SourceInfo
  ): Underlying[T] = {
    // Recursive call
    def rec[A <: BaseModule](m: A): Underlying[A] = {
      def clone(x: A, p: Option[BaseModule], name: () => String): Underlying[A] = {
        val newChild = Module.do_pseudo_apply(new experimental.hierarchy.InstanceClone(x, name), p)
        Clone(newChild)
      }
      (m, context) match {
        case (c, ctx) if ctx == c                        => Proto(c)
        case (c, ctx: IsClone[_]) if ctx.hasSameProto(c) => Clone(ctx.asInstanceOf[IsClone[A]])
        case (c, ctx) if c._parent.isEmpty               => Proto(c)
        case (_, _) =>
          cloneModuleToContext(Proto(m._parent.get), context) match {
            case Proto(p) => Proto(m)
            case Clone(p: BaseModule) =>
              clone(m, Some(p), () => m.instanceName)
            case _ =>
              throw new Exception(
                s"Match Error: cloneModuleToContext(Proto(m._parent.get), context)=" +
                  s"${cloneModuleToContext(Proto(m._parent.get), context)}"
              )
          }
      }
    }
    module match {
      case Proto(m) => rec(m)
      case Clone(m: ModuleClone[_]) =>
        rec(m) match {
          case Proto(mx) => Clone(mx)
          case Clone(i: InstanceClone[_]) =>
            val newChild = Module.do_pseudo_apply(new InstanceClone(m.getProto, () => m.instanceName), i._parent)
            Clone(newChild)
          case _ => throw new InternalErrorException(s"Match error: rec(m)=${rec(m)}")
        }
      case Clone(m: InstanceClone[_]) =>
        rec(m) match {
          case Proto(mx) => Clone(mx)
          case Clone(i: InstanceClone[_]) =>
            val newChild = Module.do_pseudo_apply(new InstanceClone(m.getProto, () => m.instanceName), i._parent)
            Clone(newChild)
          case _ => throw new InternalErrorException(s"Match error: rec(m)=${rec(m)}")
        }
      case _ => throw new InternalErrorException(s"Match error: module=$module")
    }
  }

  @deprecated("Use Lookupable.isLookupable instead of extending this.", "Chisel 7.0.0")
  class SimpleLookupable[X] extends Lookupable[X] {
    type B = X
    type C = X
    def definitionLookup[A](that: A => B, definition: Definition[A]): C = that(definition.proto)
    def instanceLookup[A](that:   A => B, instance:   Instance[A]):   C = that(instance.proto)
  }

  implicit def lookupInstance[B <: BaseModule](implicit sourceInfo: SourceInfo): Simple[Instance[B]] =
    new Lookupable[Instance[B]] {
      type C = Instance[B]
      def definitionLookup[A](that: A => Instance[B], definition: Definition[A]): C = {
        val ret = that(definition.proto)
        new Instance(cloneModuleToContext(ret.underlying, definition.getInnerDataContext.get))
      }
      def instanceLookup[A](that: A => Instance[B], instance: Instance[A]): C = {
        val ret = that(instance.proto)
        instance.underlying match {
          // If instance is just a normal module, no changing of context is necessary
          case Proto(_) => new Instance(ret.underlying)
          case Clone(_) => new Instance(cloneModuleToContext(ret.underlying, instance.getInnerDataContext.get))
        }
      }
    }

  @deprecated("Looking up Modules is deprecated, please cast to Instance instead (.toInstance).", "Chisel 7.0.0")
  implicit def lookupModule[B <: BaseModule](implicit sourceInfo: SourceInfo): Aux[B, Instance[B]] =
    new Lookupable[B] {
      type C = Instance[B]
      def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
        val ret = that(definition.proto)
        new Instance(cloneModuleToContext(Proto(ret), definition.getInnerDataContext.get))
      }
      def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
        val ret = that(instance.proto)
        instance.underlying match {
          // If instance is just a normal module, no changing of context is necessary
          case Proto(_) => new Instance(Proto(ret))
          case Clone(_) => new Instance(cloneModuleToContext(Proto(ret), instance.getInnerDataContext.get))
        }
      }
    }

  implicit def lookupData[B <: Data](implicit sourceInfo: SourceInfo): Simple[B] =
    new Lookupable[B] {
      type C = B
      def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
        val ret = that(definition.proto)
        if (isView(ret)) {
          ??? // TODO!!!!!!  cloneViewToContext(ret, instance, ioMap, instance.getInnerDataContext)
        } else {
          doLookupData(ret, definition.cache, None, definition.getInnerDataContext)
        }
      }
      def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
        val ret = that(instance.proto)

        // As Property ports are not yet Lookupable, they are skipped here.
        def getIoMap(hierarchy: Hierarchy[_]): Option[Map[Data, Data]] = {
          hierarchy.underlying match {
            case Clone(x: ModuleClone[_])       => Some(x.ioMap)
            case Proto(x: BaseModule)           => Some(x.getIOs.map { data => data -> data }.toMap)
            case Clone(x: InstantiableClone[_]) => getIoMap(x._innerContext)
            case Clone(x: InstanceClone[_])     => None
            case other => {
              Builder.exception(s"Internal Error! Unexpected case where we can't get IO Map: $other")
            }
          }
        }
        val ioMap = getIoMap(instance)

        if (isView(ret)) {
          cloneViewToContext(ret, instance.cache, ioMap, instance.getInnerDataContext)
        } else {
          doLookupData(ret, instance.cache, ioMap, instance.getInnerDataContext)
        }

      }
    }

  private[chisel3] def cloneMemToContext[T <: MemBase[_]](
    mem:     T,
    context: BaseModule
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    mem._parent match {
      case None => mem
      case Some(parent) =>
        val newParent = cloneModuleToContext(Proto(parent), context)
        newParent match {
          case Proto(p) if p == parent => mem
          case Clone(mod: BaseModule) =>
            val existingMod = Builder.currentModule
            Builder.currentModule = Some(mod)
            val newChild: T = mem match {
              case m: Mem[_] => new Mem(m.t.asInstanceOf[Data].cloneTypeFull, m.length, sourceInfo).asInstanceOf[T]
              case m: SyncReadMem[_] =>
                new SyncReadMem(m.t.asInstanceOf[Data].cloneTypeFull, m.length, m.readUnderWrite, sourceInfo)
                  .asInstanceOf[T]
            }
            Builder.currentModule = existingMod
            newChild.setRef(mem.getRef, true)
            newChild
          case _ =>
            throw new InternalErrorException(s"Match error: newParent=$newParent")
        }
    }
  }

  implicit def lookupMem[B <: MemBase[_]](implicit sourceInfo: SourceInfo): Simple[B] =
    new LookupableImpl[B] {
      type C = B
      override protected def impl[A](that: A => B, hierarchy: Hierarchy[A]): C = {
        cloneMemToContext(that(hierarchy.proto), hierarchy.getInnerDataContext.get)
      }
    }

  // TODO, this, cloneMemToContext, and cloneDataToContext should be unified
  private def cloneHasTargetToContext(
    hasTarget: HasTarget,
    context:   BaseModule
  )(
    implicit sourceInfo: SourceInfo
  ): HasTarget = {
    hasTarget match {
      case HasTarget.Impl(st: SramTarget) =>
        st._parent match {
          case None => hasTarget
          case Some(parent) =>
            val newParent = cloneModuleToContext(Proto(parent), context)
            newParent match {
              case Proto(p) if p == parent => hasTarget
              case Clone(mod: BaseModule) =>
                val existingMod = Builder.currentModule
                Builder.currentModule = Some(mod)
                val newChild = new SramTarget
                Builder.currentModule = existingMod
                newChild.setRef(st.getRef, true)
                HasTarget(newChild)
              case _ =>
                throw new InternalErrorException(s"Match error: newParent=$newParent")
            }
        }
    }
  }

  implicit def lookupHasTarget(implicit sourceInfo: SourceInfo): Simple[HasTarget] =
    new LookupableImpl[HasTarget] {
      type C = HasTarget
      override protected def impl[A](that: A => HasTarget, hierarchy: Hierarchy[A]): C = {
        cloneHasTargetToContext(that(hierarchy.proto), hierarchy.getInnerDataContext.get)
      }
    }

  import scala.language.higherKinds // Required to avoid warning for lookupIterable type parameter
  implicit def lookupIterable[B, F[_] <: Iterable[_]](
    implicit sourceInfo: SourceInfo,
    lookupable:          Lookupable[B]
  ): Aux[F[B], F[lookupable.C]] = new LookupableImpl[F[B]] {
    type C = F[lookupable.C]
    override protected def impl[A](that: A => F[B], hierarchy: Hierarchy[A]): C = {
      val ret = that(hierarchy.proto).asInstanceOf[Iterable[B]]
      ret.map { (x: B) => lookupable.hierarchyLookup[A](_ => x, hierarchy) }.asInstanceOf[C]
    }
  }
  implicit def lookupOption[B](
    implicit sourceInfo: SourceInfo,
    lookupable:          Lookupable[B]
  ): Aux[Option[B], Option[lookupable.C]] = new LookupableImpl[Option[B]] {
    type C = Option[lookupable.C]
    override protected def impl[A](that: A => Option[B], hierarchy: Hierarchy[A]): C = {
      val ret = that(hierarchy.proto)
      ret.map { (x: B) => lookupable.hierarchyLookup[A](_ => x, hierarchy) }
    }
  }
  implicit def lookupEither[L, R](
    implicit sourceInfo: SourceInfo,
    lookupableL:         Lookupable[L],
    lookupableR:         Lookupable[R]
  ): Aux[Either[L, R], Either[lookupableL.C, lookupableR.C]] = new LookupableImpl[Either[L, R]] {
    type C = Either[lookupableL.C, lookupableR.C]
    override protected def impl[A](that: A => Either[L, R], hierarchy: Hierarchy[A]): C = {
      val ret = that(hierarchy.proto)
      ret.map { (x: R) => lookupableR.hierarchyLookup[A](_ => x, hierarchy) }.left.map { (x: L) =>
        lookupableL.hierarchyLookup[A](_ => x, hierarchy)
      }
    }
  }

  // TODO Once Lookupable return type change is removed, we can just call product factory above.
  implicit def lookupTuple2[T1, T2](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2]
  ): Aux[(T1, T2), (lookupableT1.C, lookupableT2.C)] = new LookupableImpl[(T1, T2)] {
    type C = (lookupableT1.C, lookupableT2.C)
    override protected def impl[A](that: A => (T1, T2), hierarchy: Hierarchy[A]): C = {
      val t1res = lookupableT1.hierarchyLookup[A](that(_)._1, hierarchy)
      val t2res = lookupableT2.hierarchyLookup[A](that(_)._2, hierarchy)
      (t1res, t2res)
    }
  }

  // TODO Once Lookupable return type change is removed, we can just call product factory above.
  implicit def lookupTuple3[T1, T2, T3](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2],
    lookupableT3:        Lookupable[T3]
  ): Aux[(T1, T2, T3), (lookupableT1.C, lookupableT2.C, lookupableT3.C)] = new LookupableImpl[(T1, T2, T3)] {
    type C = (lookupableT1.C, lookupableT2.C, lookupableT3.C)
    override protected def impl[A](that: A => (T1, T2, T3), hierarchy: Hierarchy[A]): C = {
      val t1res = lookupableT1.hierarchyLookup[A](that(_)._1, hierarchy)
      val t2res = lookupableT2.hierarchyLookup[A](that(_)._2, hierarchy)
      val t3res = lookupableT3.hierarchyLookup[A](that(_)._3, hierarchy)
      (t1res, t2res, t3res)
    }
  }

  // TODO Once Lookupable return type change is removed, we can just call product factory above.
  implicit def lookupTuple4[T1, T2, T3, T4](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2],
    lookupableT3:        Lookupable[T3],
    lookupableT4:        Lookupable[T4]
  ): Aux[(T1, T2, T3, T4), (lookupableT1.C, lookupableT2.C, lookupableT3.C, lookupableT4.C)] =
    new LookupableImpl[(T1, T2, T3, T4)] {
      type C = (lookupableT1.C, lookupableT2.C, lookupableT3.C, lookupableT4.C)
      override protected def impl[A](that: A => (T1, T2, T3, T4), hierarchy: Hierarchy[A]): C = {
        val t1res = lookupableT1.hierarchyLookup[A](that(_)._1, hierarchy)
        val t2res = lookupableT2.hierarchyLookup[A](that(_)._2, hierarchy)
        val t3res = lookupableT3.hierarchyLookup[A](that(_)._3, hierarchy)
        val t4res = lookupableT4.hierarchyLookup[A](that(_)._4, hierarchy)
        (t1res, t2res, t3res, t4res)
      }
    }

  // TODO Once Lookupable return type change is removed, we can just call product factory above.
  implicit def lookupTuple5[T1, T2, T3, T4, T5](
    implicit sourceInfo: SourceInfo,
    lookupableT1:        Lookupable[T1],
    lookupableT2:        Lookupable[T2],
    lookupableT3:        Lookupable[T3],
    lookupableT4:        Lookupable[T4],
    lookupableT5:        Lookupable[T5]
  ): Aux[(T1, T2, T3, T4, T5), (lookupableT1.C, lookupableT2.C, lookupableT3.C, lookupableT4.C, lookupableT5.C)] =
    new LookupableImpl[(T1, T2, T3, T4, T5)] {
      type C = (lookupableT1.C, lookupableT2.C, lookupableT3.C, lookupableT4.C, lookupableT5.C)
      override protected def impl[A](that: A => (T1, T2, T3, T4, T5), hierarchy: Hierarchy[A]): C = {
        val t1res = lookupableT1.hierarchyLookup[A](that(_)._1, hierarchy)
        val t2res = lookupableT2.hierarchyLookup[A](that(_)._2, hierarchy)
        val t3res = lookupableT3.hierarchyLookup[A](that(_)._3, hierarchy)
        val t4res = lookupableT4.hierarchyLookup[A](that(_)._4, hierarchy)
        val t5res = lookupableT5.hierarchyLookup[A](that(_)._5, hierarchy)
        (t1res, t2res, t3res, t4res, t5res)
      }
    }

  @deprecated(
    "Use of @instantiable on user-defined types is deprecated. Implement Lookupable for your type instead.",
    "Chisel 7.0.0"
  )
  implicit def lookupIsInstantiable[B <: IsInstantiable](
    implicit sourceInfo: SourceInfo
  ): Aux[B, Instance[B]] = new LookupableImpl[B] {
    type C = Instance[B]
    override protected def impl[A](that: A => B, hierarchy: Hierarchy[A]): C = {
      val ret = that(hierarchy.proto)
      val underlying = new InstantiableClone[B] {
        val getProto = ret
        lazy val _innerContext: Hierarchy[_] = hierarchy
      }
      new Instance(Clone(underlying))
    }
  }

  implicit def lookupIsLookupable[B <: IsLookupable](implicit sourceInfo: SourceInfo): Simple[B] = isLookupable[B]

  implicit val lookupInt:             Simple[Int] = isLookupable[Int]
  implicit val lookupByte:            Simple[Byte] = isLookupable[Byte]
  implicit val lookupShort:           Simple[Short] = isLookupable[Short]
  implicit val lookupLong:            Simple[Long] = isLookupable[Long]
  implicit val lookupFloat:           Simple[Float] = isLookupable[Float]
  implicit val lookupDouble:          Simple[Double] = isLookupable[Double]
  implicit val lookupChar:            Simple[Char] = isLookupable[Char]
  implicit val lookupString:          Simple[String] = isLookupable[String]
  implicit val lookupBoolean:         Simple[Boolean] = isLookupable[Boolean]
  implicit val lookupBigInt:          Simple[BigInt] = isLookupable[BigInt]
  implicit val lookupUnit:            Simple[Unit] = isLookupable[Unit]
  implicit val lookupActualDirection: Simple[chisel3.ActualDirection] = isLookupable[chisel3.ActualDirection]
}
