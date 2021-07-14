package chisel3

import chisel3.experimental.BaseModule
import chisel3.internal.sourceinfo.SourceInfo
import scala.annotation.implicitNotFound
import chisel3._

@implicitNotFound("@instance is only legal when @public is only on subtypes of Data, BaseModule or Instance[_]")
sealed trait Lookupable[A, -B] {
  type C
  def lookup(that: A => B, ih: Instance[A]): C
}

object Lookupable {
  implicit def lookupModule[A <: BaseModule, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = Instance[B]
    def lookup(that: A => B, ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      val inst = new Instance(() => ret.instanceName, ret, Instance.portMap(ret), descendingContext(template).descend(InstanceContext.getContext(ret)), None)
      inst 
    }
  }
  implicit def lookupInstance[A <: BaseModule, B <: BaseModule](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Instance[B]] {
    type C = Instance[B]
    def lookup(that: A => Instance[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret.copy(context = descendingContext(template).descend(ret.context))
    }
  }
  implicit def lookupData[A <: BaseModule, B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, B] {
    type C = B
    def lookup(that: A => B, ih: Instance[A]): C = {
      val ret = that(ih.template)
      ret match {
        case x: Data if ih.ioMap.contains(x) => ih.ioMap(x).asInstanceOf[B]
        case x: Data if ih.cache.contains(x)=> ih.cache(x).asInstanceOf[B]
        case x: Data =>
          val xContext = InstanceContext.getContext(x._parent.get)
          val xmr = XMR.do_apply(x, ih.descendingContext(ih.template).descend(xContext))
          ih.cache(x) = xmr
          xmr.asInstanceOf[B]
      }
    }
  }
  implicit def lookupOption[A <: BaseModule, B <: Data](implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) = new Lookupable[A, Option[B]] {
    type C = Option[B]
    def lookup(that: A => Option[B], ih: Instance[A]): C = {
      import ih._
      val ret = that(template)
      ret.map{ _ match {
          case x: Data if ioMap.contains(x) => ioMap(x).asInstanceOf[B]
          case x: Data if cache.contains(x)=> cache(x).asInstanceOf[B]
          case x: Data =>
            val xContext = InstanceContext.getContext(x._parent.get)
            val xmr = XMR.do_apply(x, descendingContext(template).descend(xContext))
            cache(x) = xmr
            xmr.asInstanceOf[B]
        }
      }
    }
  }
}
