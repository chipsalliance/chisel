package chisel3

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.SourceInfo
import scala.annotation.implicitNotFound
import chisel3._

trait IsLookupable
trait IsInstantiable
//@implicitNotFound("@instance is only legal when @public is only on subtypes of Data, BaseModule or Instance[_]")
sealed trait Lookupable[A, -B] {
  type C
  def lookup(that: A => B, ih: Instance[A]): C
}

object Lookupable {
  //implicit def lookupModule[A, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
  //  type C = Instance[B]
  //  def lookup(that: A => B, ih: Instance[A]): C = {
  //    val ret = that(ih.template)
  //    ih.template match {
  //      case template: BaseModule =>
  //        val inst = new Instance(() => ret.instanceName, ret, Instance.portMap(ret), ih.context.descend(InstanceContext.getContext(ret)), None)
  //        inst 
  //      case _ =>
  //        new Instance(() => ret.instanceName, ret, Instance.portMap(ret), ih.context.descend(InstanceContext.getContext(ret)), None)
  //    }
  //  }
  //}
  implicit def lookupInstance[A, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Instance[B]] {
    type C = Instance[B]
    def lookup(that: A => Instance[B], ih: Instance[A]): C = {
      val ret = that(ih.template)
      ih.template match {
        case template: BaseModule => ret.copy(context = ih.context.descend(ret.context))
        case _                    => ret.copy(context = ih.context.descend(ret.context))
      }
    }
  }
  implicit def lookupData[A, B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = B
    def lookup(that: A => B, ih: Instance[A]): C = {
      val ret = that(ih.template)
      ret match {
        case x: Data if ih.ioMap.contains(x) => ih.ioMap(x).asInstanceOf[B]
        case x: Data if ih.cache.contains(x)=> ih.cache(x).asInstanceOf[B]
        case x: Data =>
          ih.template match {
            case template: BaseModule =>
              val xContext = InstanceContext.getContext(x._parent.get)
              val xmr = XMR.do_apply(x, ih.context.descend(xContext))
              ih.cache(x) = xmr
              xmr.asInstanceOf[B]
            case _ =>
              val xContext = InstanceContext.getContext(x._parent.get)
              val xmr = XMR.do_apply(x, ih.context.descend(xContext))
              ih.cache(x) = xmr
              xmr.asInstanceOf[B]
          }
      }
    }
  }
  implicit def lookupList[A, B](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, lookupable: Lookupable[A, B]) = new Lookupable[A, List[B]] {
    type C = List[lookupable.C]
    def lookup(that: A => List[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret.map{ x: B => lookupable.lookup(_ => x, ih) }
    }
  }
  implicit def lookupOption[A, B](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions, lookupable: Lookupable[A, B]) = new Lookupable[A, Option[B]] {
    type C = Option[lookupable.C]
    def lookup(that: A => Option[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret.map{ x: B => lookupable.lookup(_ => x, ih) }
    }
  }
  implicit def lookupIsLookupable[A, B <: IsLookupable](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = B
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.template)
  }
  implicit def lookupIsInstantiable[A, B <: IsInstantiable](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = Instance[B]
    def lookup(that: A => B, ih: Instance[A]): C = {
        val ret = that(ih.template)
        ret match {
            case retModule: BaseModule =>
              ih.template match {
                case template: BaseModule =>
                  new Instance(() => retModule.instanceName, ret, Instance.portMap(retModule), ih.context.descend(InstanceContext.getContext(retModule)), None)
                case _ =>
                  new Instance(() => retModule.instanceName, ret, Instance.portMap(retModule), ih.context.descend(InstanceContext.getContext(retModule)), None)
              }
            case other => new Instance(() => "", ret, Map.empty, ih.context, None)
        }
    }
  }
  implicit def lookupString[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, String] {
    type B = String
    type C = String
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.template)
  }
  implicit def lookupInt[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Int] {
    type B = Int
    type C = Int
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.template)
  }
  implicit def lookupBoolean[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Boolean] {
    type B = Boolean
    type C = Boolean
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.template)
  }
  implicit def lookupBigInt[A](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, BigInt] {
    type B = BigInt
    type C = BigInt
    def lookup(that: A => B, ih: Instance[A]): C = that(ih.template)
  }
}
