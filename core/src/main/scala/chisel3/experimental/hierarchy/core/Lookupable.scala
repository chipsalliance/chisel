// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3.experimental.{BaseModule, SourceInfo}
import chisel3.experimental.hierarchy.{InstanceClone, InstantiableClone, ModuleClone}

import scala.annotation.implicitNotFound
import scala.collection.mutable.HashMap
import chisel3._
import chisel3.experimental.dataview.{isView, reify, reifyIdentityView}
import chisel3.internal.firrtl.ir.{Arg, ILit, Index, LitIndex, ModuleIO, Slot, ULit}
import chisel3.internal.{
  throwException,
  AggregateViewBinding,
  Builder,
  ChildBinding,
  ViewBinding,
  ViewParent,
  ViewWriteability
}

/** Represents lookup typeclass to determine how a value accessed from an original IsInstantiable
  *   should be tweaked to return the Instance's version
  * Sealed.
  */
@implicitNotFound(
  "@public is only legal within a class or trait marked @instantiable, and only on vals of type" +
    " Data, BaseModule, MemBase, IsInstantiable, IsLookupable, or Instance[_], or in an Iterable, Option, Either, or Tuple2"
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
  protected def getProto[A](h:      Hierarchy[A]): A = h.proto
  protected def getUnderlying[A](h: Hierarchy[A]): Underlying[A] = h.underlying
}

object Lookupable {

  /** Type alias for simplifying explicit Lookupable type ascriptions */
  type Aux[B, C0] = Lookupable[B] { type C = C0 }

  /** Type alias for simple Lookupable types */
  type Simple[B] = Aux[B, B]

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
            newChild.bind(internal.CrossModuleBinding)
            newChild.setAllParents(Some(m))
            newChild
          case _ => throw new InternalErrorException("Match error: newParent=$newParent")
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
      case _ => impl(data)
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
    def err(msg:               String) = throwException(s"Internal Error! $msg")
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
            case (Slot(_, name), rec: Record) => rec._elements(name)
            case (LitIndex(_, n), vec: Vec[_]) => vec.apply(n)
            case (Index(_, ILit(n)), vec: Vec[_]) => vec.apply(n.toInt)
            case (ModuleIO(_, name), rec: Record) => rec._elements(name)
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
            val mapping = coiterate(result, data).flatMap {
              case (res, from) => reifyOpt(from).map { case (t, w) => (res, lookupData(t), w) }
            }
            val newMap = mapping.map { case (from, to, _) => from -> to }.toMap
            val wrMap = mapping.flatMap { case (from, _, wr) => Option.when(wr.isReadOnly)(from -> wr) }.toMap
            AggregateViewBinding(newMap, Option.when(wrMap.nonEmpty)(wrMap))
        }
      case _ => throw new InternalErrorException("Match error: data.topBinding=${data.topBinding}")
    }

    // TODO Unify the following with `.viewAs`
    // We must also mark any non-identity Aggregates as unnammed
    newBinding match {
      case _: ViewBinding => // Do nothing
      case AggregateViewBinding(childMap, _) =>
        // TODO we could do reifySingleTarget instead of just marking non-identity mappings
        getRecursiveFields.lazily(result, "_").foreach {
          case (agg: Aggregate, _) if !childMap.contains(agg) =>
            Builder.unnamedViews += agg
          case _ => ()
        }
      case _ => throw new InternalErrorException("Match error: newBinding=$newBinding")
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
        val newChild = Module.do_pseudo_apply(new experimental.hierarchy.InstanceClone(x, name))
        newChild._parent = p
        Clone(newChild)
      }
      (m, context) match {
        case (c, ctx) if ctx == c => Proto(c)
        case (c, ctx: IsClone[_]) if ctx.hasSameProto(c) => Clone(ctx.asInstanceOf[IsClone[A]])
        case (c, ctx) if c._parent.isEmpty => Proto(c)
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
            val newChild = Module.do_pseudo_apply(new InstanceClone(m.getProto, () => m.instanceName))
            newChild._parent = i._parent
            Clone(newChild)
          case _ => throw new InternalErrorException("Match error: rec(m)=${rec(m)}")
        }
      case Clone(m: InstanceClone[_]) =>
        rec(m) match {
          case Proto(mx) => Clone(mx)
          case Clone(i: InstanceClone[_]) =>
            val newChild = Module.do_pseudo_apply(new InstanceClone(m.getProto, () => m.instanceName))
            newChild._parent = i._parent
            Clone(newChild)
          case _ => throw new InternalErrorException("Match error: rec(m)=${rec(m)}")
        }
      case _ => throw new InternalErrorException("Match error: module=$module")
    }
  }

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
            case Clone(x: ModuleClone[_]) => Some(x.ioMap)
            case Proto(x: BaseModule) => Some(x.getChiselPorts.map { case (_, data: Data) => data -> data }.toMap)
            case Clone(x: InstantiableClone[_]) => getIoMap(x._innerContext)
            case Clone(x: InstanceClone[_]) => None
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
            throw new InternalErrorException("Match error: newParent=$newParent")
        }
    }
  }

  implicit def lookupMem[B <: MemBase[_]](implicit sourceInfo: SourceInfo): Simple[B] =
    new Lookupable[B] {
      type C = B
      def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
        cloneMemToContext(that(definition.proto), definition.getInnerDataContext.get)
      }
      def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
        cloneMemToContext(that(instance.proto), instance.getInnerDataContext.get)
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
    new Lookupable[HasTarget] {
      type C = HasTarget
      def definitionLookup[A](that: A => HasTarget, definition: Definition[A]): C = {
        cloneHasTargetToContext(that(definition.proto), definition.getInnerDataContext.get)
      }
      def instanceLookup[A](that: A => HasTarget, instance: Instance[A]): C = {
        cloneHasTargetToContext(that(instance.proto), instance.getInnerDataContext.get)
      }
    }

  import scala.language.higherKinds // Required to avoid warning for lookupIterable type parameter
  implicit def lookupIterable[B, F[_] <: Iterable[_]](
    implicit sourceInfo: SourceInfo,
    lookupable:          Lookupable[B]
  ): Aux[F[B], F[lookupable.C]] = new Lookupable[F[B]] {
    type C = F[lookupable.C]
    def definitionLookup[A](that: A => F[B], definition: Definition[A]): C = {
      val ret = that(definition.proto).asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.definitionLookup[A](_ => x, definition) }.asInstanceOf[C]
    }
    def instanceLookup[A](that: A => F[B], instance: Instance[A]): C = {
      import instance._
      val ret = that(proto).asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.instanceLookup[A](_ => x, instance) }.asInstanceOf[C]
    }
  }
  implicit def lookupOption[B](
    implicit sourceInfo: SourceInfo,
    lookupable:          Lookupable[B]
  ): Aux[Option[B], Option[lookupable.C]] = new Lookupable[Option[B]] {
    type C = Option[lookupable.C]
    def definitionLookup[A](that: A => Option[B], definition: Definition[A]): C = {
      val ret = that(definition.proto)
      ret.map { x: B => lookupable.definitionLookup[A](_ => x, definition) }
    }
    def instanceLookup[A](that: A => Option[B], instance: Instance[A]): C = {
      import instance._
      val ret = that(proto)
      ret.map { x: B => lookupable.instanceLookup[A](_ => x, instance) }
    }
  }
  implicit def lookupEither[L, R](
    implicit sourceInfo: SourceInfo,
    lookupableL:         Lookupable[L],
    lookupableR:         Lookupable[R]
  ): Aux[Either[L, R], Either[lookupableL.C, lookupableR.C]] = new Lookupable[Either[L, R]] {
    type C = Either[lookupableL.C, lookupableR.C]
    def definitionLookup[A](that: A => Either[L, R], definition: Definition[A]): C = {
      val ret = that(definition.proto)
      ret.map { x: R => lookupableR.definitionLookup[A](_ => x, definition) }.left.map { x: L =>
        lookupableL.definitionLookup[A](_ => x, definition)
      }
    }
    def instanceLookup[A](that: A => Either[L, R], instance: Instance[A]): C = {
      import instance._
      val ret = that(proto)
      ret.map { x: R => lookupableR.instanceLookup[A](_ => x, instance) }.left.map { x: L =>
        lookupableL.instanceLookup[A](_ => x, instance)
      }
    }
  }

  implicit def lookupTuple2[X, Y](
    implicit sourceInfo: SourceInfo,
    lookupableX:         Lookupable[X],
    lookupableY:         Lookupable[Y]
  ): Aux[(X, Y), (lookupableX.C, lookupableY.C)] = new Lookupable[(X, Y)] {
    type C = (lookupableX.C, lookupableY.C)
    def definitionLookup[A](that: A => (X, Y), definition: Definition[A]): C = {
      val ret = that(definition.proto)
      (
        lookupableX.definitionLookup[A](_ => ret._1, definition),
        lookupableY.definitionLookup[A](_ => ret._2, definition)
      )
    }
    def instanceLookup[A](that: A => (X, Y), instance: Instance[A]): C = {
      import instance._
      val ret = that(proto)
      (lookupableX.instanceLookup[A](_ => ret._1, instance), lookupableY.instanceLookup[A](_ => ret._2, instance))
    }
  }

  implicit def lookupIsInstantiable[B <: IsInstantiable](
    implicit sourceInfo: SourceInfo
  ): Aux[B, Instance[B]] = new Lookupable[B] {
    type C = Instance[B]
    def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
      val ret = that(definition.proto)
      val underlying = new InstantiableClone[B] {
        val getProto = ret
        lazy val _innerContext = definition
      }
      new Instance(Clone(underlying))
    }
    def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
      val ret = that(instance.proto)
      val underlying = new InstantiableClone[B] {
        val getProto = ret
        lazy val _innerContext = instance
      }
      new Instance(Clone(underlying))
    }
  }

  implicit def lookupIsLookupable[B <: IsLookupable](implicit sourceInfo: SourceInfo): SimpleLookupable[B] =
    new SimpleLookupable[B]()

  implicit val lookupInt:     SimpleLookupable[Int] = new SimpleLookupable[Int]()
  implicit val lookupByte:    SimpleLookupable[Byte] = new SimpleLookupable[Byte]()
  implicit val lookupShort:   SimpleLookupable[Short] = new SimpleLookupable[Short]()
  implicit val lookupLong:    SimpleLookupable[Long] = new SimpleLookupable[Long]()
  implicit val lookupFloat:   SimpleLookupable[Float] = new SimpleLookupable[Float]()
  implicit val lookupDouble:  SimpleLookupable[Double] = new SimpleLookupable[Double]()
  implicit val lookupChar:    SimpleLookupable[Char] = new SimpleLookupable[Char]()
  implicit val lookupString:  SimpleLookupable[String] = new SimpleLookupable[String]()
  implicit val lookupBoolean: SimpleLookupable[Boolean] = new SimpleLookupable[Boolean]()
  implicit val lookupBigInt:  SimpleLookupable[BigInt] = new SimpleLookupable[BigInt]()
}
