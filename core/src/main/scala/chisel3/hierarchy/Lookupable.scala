package chisel3

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.BaseModule.{ModuleClone, InstanceClone, IsClone, InstantiableClone}
import scala.annotation.implicitNotFound
import chisel3._

trait IsLookupable

import scala.language.dynamics

trait IsInstantiable extends Dynamic {
  def selectDynamic(name: String) = throw new Exception(s"Cannot call method $name on $this!")
}
//@implicitNotFound("@instance is only legal when @public is only on subtypes of Data, BaseModule or Instance[_]")
sealed trait Lookupable[A, -B] {
  type C
  def lookup(that: A => B, ih: Instance[A]): C
}

object Lookupable {
  //implicit def lookupModule[A, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
  //  type C = Instance[B]
  //  def lookup(that: A => B, ih: Instance[A]): C = {
  //    val ret = that(ih.definition)
  //    ih.definition match {
  //      case definition: BaseModule =>
  //        val inst = new Instance(() => ret.instanceName, ret, Instance.portMap(ret), ih.context.descend(InstanceContext.getContext(ret)), None)
  //        inst 
  //      case _ =>
  //        new Instance(() => ret.instanceName, ret, Instance.portMap(ret), ih.context.descend(InstanceContext.getContext(ret)), None)
  //    }
  //  }
  //}
  def allParents(x: internal.HasId): Seq[internal.HasId] = {
    Seq(x) ++ x._parent.map(allParents).getOrElse(Nil)
  }
  def cloneDataToContext[T <: Data](child: T, context: BaseModule)
                               (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    internal.requireIsHardware(child, "cross module reference type")
    child._parent match {
      case None => child
      case Some(parent) =>
        val newParent = cloneModuleToContext(Left(parent), context)
        newParent match {
          case Left(p) if p == parent => child
          case Right(m: BaseModule) =>
            val newChild = child.cloneTypeFull
            newChild.setRef(child.getRef, true)
            newChild.bind(internal.XMRBinding)
            internal.BaseModule.setAllParents(newChild, Some(m))
            newChild
        }
    }
  }
  def cloneModuleToContext[T <: BaseModule](child: Either[T, IsClone[T]], context: BaseModule)
                          (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Either[T, IsClone[T]] = {
    def rec[A <: BaseModule](m: A): Either[A, IsClone[A]] = {
      (m._parent, context) match {
        case (None, context) if context == m => Left(m)
        case (None, context: IsClone[_]) if context.isACloneOf(m) => Right(context.asInstanceOf[IsClone[A]])
        case (None, context) => Left(m) // this thing must be from somewhere else, so we don't clone it.
        case (Some(parent), _) =>
          cloneModuleToContext(Left(parent), context) match {
            case Left(p) => Left(m)
            case Right(p: BaseModule) =>
              val newChild = Module.do_apply(new internal.BaseModule.InstanceClone(m, () => m.instanceName))
              newChild._parent = Some(p)
              Right(newChild)
          }
        case other => throw new Exception(s"Other: $other\nChild: $m")
      }
    }
    child match {
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
  implicit def lookupInstance[A, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Instance[B]] {
    type C = Instance[B]
    def lookup(that: A => Instance[B], ih: Instance[A]): C = {
      val ret = that(ih.definition)
      ih.cloned match {
        // If ih is just a normal module, no changing of context is necessary
        case Left(_)  => new Instance(ret.cloned)
        case Right(_) => new Instance(cloneModuleToContext(ret.cloned, ih.getInnerDataContext.get))
      }
    }
  }
  implicit def lookupModule[A, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = Instance[B]
    def lookup(that: A => B, ih: Instance[A]): C = {
      val ret = that(ih.definition)
      ih.cloned match {
        // If ih is just a normal module, no changing of context is necessary
        case Left(_)  => new Instance(Left(ret))
        case Right(_) => new Instance(cloneModuleToContext(Left(ret), ih.getInnerDataContext.get))
      }
    }
  }
  implicit def lookupData[A, B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = B
    def lookup(that: A => B, ih: Instance[A]): C = {
      val ret = that(ih.definition)
      val ioMap = ih.cloned match {
        case Right(x: ModuleClone[_]) => Some(x.ioMap)
        case Left(x: BaseModule) => Some(x.getChiselPorts.map { case (_, data) => data -> data }.toMap)
        case other => None
      }
      ret match {
        case x: Data if ioMap.nonEmpty && ioMap.get.contains(x) => ioMap.get(x).asInstanceOf[B]
        case x: Data if ih.cache.contains(x) => ih.cache(x).asInstanceOf[B]
        case x: Data if ih.getInnerDataContext.nonEmpty => cloneDataToContext(ret, ih.getInnerDataContext.get)
      }
    }
  }
  implicit def lookupList[A, B](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, lookupable: Lookupable[A, B]) = new Lookupable[A, List[B]] {
    type C = List[lookupable.C]
    def lookup(that: A => List[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(definition)
      ret.map{ x: B => lookupable.lookup(_ => x, ih) }
    }
  }
  implicit def lookupOption[A, B](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, lookupable: Lookupable[A, B]) = new Lookupable[A, Option[B]] {
    type C = Option[lookupable.C]
    def lookup(that: A => Option[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(definition)
      ret.map{ x: B => lookupable.lookup(_ => x, ih) }
    }
  }
  implicit def lookupIsLookupable[A, B <: IsLookupable](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = B
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.definition)
  }
  implicit def lookupIsInstantiable[A, B <: IsInstantiable](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = Instance[B]
    def lookup(that: A => B, ih: Instance[A]): C = {
      val ret = that(ih.definition)
      val cloned = new InstantiableClone(ret)
      cloned._parent = ih.getInnerDataContext
      new Instance(Right(cloned))
    }
  }
  implicit def lookupString[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, String] {
    type B = String
    type C = String
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.definition)
  }
  implicit def lookupInt[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Int] {
    type B = Int
    type C = Int
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.definition)
  }
  implicit def lookupBoolean[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Boolean] {
    type B = Boolean
    type C = Boolean
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.definition)
  }
  implicit def lookupBigInt[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, BigInt] {
    type B = BigInt
    type C = BigInt
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.definition)
  }
}
