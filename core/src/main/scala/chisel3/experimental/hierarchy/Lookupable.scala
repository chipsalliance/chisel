// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.BaseModule.{InstanceClone, InstantiableClone, IsClone, ModuleClone}

import scala.annotation.implicitNotFound
import scala.collection.mutable.HashMap
import chisel3._
import chisel3.experimental.dataview.{isView, reify, reifySingleData}
import chisel3.internal.firrtl.{Arg, ILit, Index, Slot, ULit}
import chisel3.internal.{AggregateViewBinding, Builder, ChildBinding, ViewBinding, ViewParent, throwException}

/** Represents lookup typeclass to determine how a value accessed from an original IsInstantiable
  *   should be tweaked to return the Instance's version
  * Sealed.
  */
@implicitNotFound("@public is only legal within a class marked @instantiable and only on vals of type" +
  " Data, BaseModule, IsInstantiable, IsLookupable, or Instance[_], or in an Iterable or Option")
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
  def definitionLookup[A](that: A => B, definition: Definition[A]): C
  def getProto[A](definition: Definition[A]): A = definition.proto
  def getProto[A](instance: Instance[A]): A = instance.proto
}
trait CustomLookupable[X] extends Lookupable[X] {
  type B = X
  type C = X
  def definitionLookup[A](that: A => B, definition: Definition[A]): C
  def instanceLookup[A](that: A => B, instance: Instance[A]): C
}


object Lookupable {

  /** Clones a data and sets its internal references to its parent module to be in a new context.
    *
    * @param data data to be cloned
    * @param context new context
    * @return
    */
  private[chisel3] def cloneDataToContext[T <: Data](data: T, context: BaseModule)
                                           (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    internal.requireIsHardware(data, "cross module reference type")
    data._parent match {
      case None => data
      case Some(parent) =>
        val newParent = cloneModuleToContext(Left(parent), context)
        newParent match {
          case Left(p) if p == parent => data
          case Right(m: BaseModule) =>
            val newChild = data.cloneTypeFull
            newChild.setRef(data.getRef, true)
            newChild.bind(internal.CrossModuleBinding)
            newChild.setAllParents(Some(m))
            newChild
        }
    }
  }
  // The business logic of lookupData
  // Also called by cloneViewToContext which potentially needs to lookup stuff from ioMap or the cache
  private[chisel3] def doLookupData[A, B <: Data](data: B, cache: HashMap[Data, Data], ioMap: Option[Map[Data, Data]], context: Option[BaseModule])
                                        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): B = {
    def impl[C <: Data](d: C): C = d match {
      case x: Data if ioMap.nonEmpty && ioMap.get.contains(x) => ioMap.get(x).asInstanceOf[C]
      case x: Data if cache.contains(x) => cache(x).asInstanceOf[C]
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

  // Helper for co-iterating on Elements of aggregates, they must be the same type but that is unchecked
  private def coiterate(a: Data, b: Data): Iterable[(Element, Element)] = {
    val as = getRecursiveFields.lazily(a, "_")
    val bs = getRecursiveFields.lazily(b, "_")
    as.zip(bs).collect { case ((ae: Element, _), (be: Element, _)) => (ae, be) }
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
      case ChildBinding(parent) => d.getRef match {
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

  // TODO this logic is complicated, can any of it be unified with viewAs?
  // If `.viewAs` would capture its arguments, we could potentially use it
  // TODO Describe what this is doing at a high level
  private[chisel3] def cloneViewToContext[A, B <: Data](data: B, cache: HashMap[Data, Data], ioMap: Option[Map[Data, Data]], context: Option[BaseModule])
                                           (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): B = {
    // alias to shorten lookups
    def lookupData[C <: Data](d: C) = doLookupData(d, cache, ioMap, context)

    val result = data.cloneTypeFull

    // We have to lookup the target(s) of the view since they may need to be cloned into the current context
    val newBinding = data.topBinding match {
      case ViewBinding(target) => ViewBinding(lookupData(reify(target)))
      case avb @ AggregateViewBinding(map, targetOpt) => data match {
        case _: Element => ViewBinding(lookupData(reify(map(data))))
        case _: Aggregate =>
          // Provide a 1:1 mapping if possible
          val singleTargetOpt = targetOpt.filter(_ => avb == data.binding.get).flatMap(reifySingleData)
          singleTargetOpt match {
            case Some(singleTarget) => // It is 1:1!
              // This is a little tricky because the values in newMap need to point to Elements of newTarget
              val newTarget = lookupData(singleTarget)
              val newMap = coiterate(result, data).map { case (res, from) =>
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
    *         to that parent and set their parents to be in this cloned context
    *   (3) A parent has no root; in that case, we do nothing and return the module.
    *
    * @param module original or clone to be cloned into a new context
    * @param context new context
    * @return original or clone in the new context
    */
  private[chisel3] def cloneModuleToContext[T <: BaseModule](module: Either[T, IsClone[T]], context: BaseModule)
      (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Either[T, IsClone[T]] = {
    // Recursive call
    def rec[A <: BaseModule](m: A): Either[A, IsClone[A]] = {
      def clone(x: A, p: Option[BaseModule], name: () => String): Either[A, IsClone[A]] = {
        val newChild = Module.do_apply(new internal.BaseModule.InstanceClone(x, name))
        newChild._parent = p
        Right(newChild)
      }
      (m, context) match {
        case (c, ctx) if ctx == c => Left(c)
        case (c, ctx: IsClone[_]) if ctx.isACloneOf(c) => Right(ctx.asInstanceOf[IsClone[A]])
        case (c, ctx) if c._parent.isEmpty => Left(c)
        case (_, _) => 
          cloneModuleToContext(Left(m._parent.get), context) match {
            case Left(p) => Left(m)
            case Right(p: BaseModule) =>
              clone(m, Some(p), () => m.instanceName)
          }
      }
    }
    module match {
      case Left(m) => rec(m)
      case Right(m: ModuleClone[_]) =>
        rec(m) match {
          case Left(mx) => Right(mx)
          case Right(i: InstanceClone[_]) =>
            val newChild = Module.do_apply(new InstanceClone(m._proto, () => m.instanceName))
            newChild._parent = i._parent
            Right(newChild)
        }
      case Right(m: InstanceClone[_]) =>
        rec(m) match {
          case Left(mx) => Right(mx)
          case Right(i: InstanceClone[_]) =>
            val newChild = Module.do_apply(new InstanceClone(m._proto, () => m.instanceName))
            newChild._parent = i._parent
            Right(newChild)
        }
    }
  }

  sealed class SimpleLookupable[X] extends Lookupable[X] {
    type B = X
    type C = X
    def definitionLookup[A](that: A => B, definition: Definition[A]): C = that(definition.proto)
    def instanceLookup[A](that: A => B, instance: Instance[A]): C = that(instance.proto)
  }
  implicit def lookupInstance[B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[Instance[B]] {
    type C = Instance[B]
    def definitionLookup[A](that: A => Instance[B], definition: Definition[A]): C = {
      val ret = that(definition.proto)
      new Instance(cloneModuleToContext(ret.cloned, definition.getInnerDataContext.get))
    }
    def instanceLookup[A](that: A => Instance[B], instance: Instance[A]): C = {
      val ret = that(instance.proto)
      instance.cloned match {
        // If instance is just a normal module, no changing of context is necessary
        case Left(_)  => new Instance(ret.cloned)
        case Right(_) => new Instance(cloneModuleToContext(ret.cloned, instance.getInnerDataContext.get))
      }
    }
  }

  implicit def lookupModule[B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[B] {
    type C = Instance[B]
    def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
      val ret = that(definition.proto)
      new Instance(cloneModuleToContext(Left(ret), definition.getInnerDataContext.get))
    }
    def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
      val ret = that(instance.proto)
      instance.cloned match {
        // If instance is just a normal module, no changing of context is necessary
        case Left(_)  => new Instance(Left(ret))
        case Right(_) => new Instance(cloneModuleToContext(Left(ret), instance.getInnerDataContext.get))
      }
    }
  }

  implicit def lookupData[B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[B] {
    type C = B
    def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
      val ret = that(definition.proto)
      if (isView(ret)) {
        ???  // TODO!!!!!!  cloneViewToContext(ret, instance, ioMap, instance.getInnerDataContext)
      } else {
        doLookupData(ret, definition.cache, None, definition.getInnerDataContext)
      }
    }
    def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
      val ret = that(instance.proto)
      val ioMap: Option[Map[Data, Data]] = instance.cloned match {
        case Right(x: ModuleClone[_]) => Some(x.ioMap)
        case Left(x: BaseModule) => Some(x.getChiselPorts.map { case (_, data) => data -> data }.toMap)
        case _ => None
      }
      if (isView(ret)) {
        cloneViewToContext(ret, instance.cache, ioMap, instance.getInnerDataContext)
      } else {
        doLookupData(ret, instance.cache, ioMap, instance.getInnerDataContext)
      }

    }
  }

  import scala.language.higherKinds // Required to avoid warning for lookupIterable type parameter
  implicit def lookupIterable[B, F[_] <: Iterable[_]](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, lookupable: Lookupable[B]) = new Lookupable[F[B]] {
    type C = F[lookupable.C]
    def definitionLookup[A](that: A => F[B], definition: Definition[A]): C = {
      val ret = that(definition.proto).asInstanceOf[Iterable[B]]
      ret.map{ x: B => lookupable.definitionLookup[A](_ => x, definition) }.asInstanceOf[C]
    }
    def instanceLookup[A](that: A => F[B], instance: Instance[A]): C = {
      import instance._
      val ret = that(proto).asInstanceOf[Iterable[B]]
      ret.map{ x: B => lookupable.instanceLookup[A](_ => x, instance) }.asInstanceOf[C]
    }
  }
  implicit def lookupOption[B](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, lookupable: Lookupable[B]) = new Lookupable[Option[B]] {
    type C = Option[lookupable.C]
    def definitionLookup[A](that: A => Option[B], definition: Definition[A]): C = {
      val ret = that(definition.proto)
      ret.map{ x: B => lookupable.definitionLookup[A](_ => x, definition) }
    }
    def instanceLookup[A](that: A => Option[B], instance: Instance[A]): C = {
      import instance._
      val ret = that(proto)
      ret.map{ x: B => lookupable.instanceLookup[A](_ => x, instance) }
    }
  }
  implicit def lookupIsInstantiable[B <: IsInstantiable](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[B] {
    type C = Instance[B]
    def definitionLookup[A](that: A => B, definition: Definition[A]): C = {
      val ret = that(definition.proto)
      val cloned = new InstantiableClone(ret)
      cloned._parent = definition.getInnerDataContext
      new Instance(Right(cloned))
    }
    def instanceLookup[A](that: A => B, instance: Instance[A]): C = {
      val ret = that(instance.proto)
      val cloned = new InstantiableClone(ret)
      cloned._parent = instance.getInnerDataContext
      new Instance(Right(cloned))
    }
  }

  implicit def lookupIsLookupable[B <: IsLookupable](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new SimpleLookupable[B]()
  
  implicit val lookupInt = new SimpleLookupable[Int]()
  implicit val lookupByte = new SimpleLookupable[Byte]()
  implicit val lookupShort = new SimpleLookupable[Short]()
  implicit val lookupLong = new SimpleLookupable[Long]()
  implicit val lookupFloat = new SimpleLookupable[Float]()
  implicit val lookupChar = new SimpleLookupable[Char]()
  implicit val lookupString = new SimpleLookupable[String]()
  implicit val lookupBoolean = new SimpleLookupable[Boolean]()
  implicit val lookupBigInt = new SimpleLookupable[BigInt]()
}
