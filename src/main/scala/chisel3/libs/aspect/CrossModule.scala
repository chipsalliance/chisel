package chisel3.libs.aspect

import chisel3._
import chisel3.experimental.{BaseModule, MultiIOModule}
import chisel3.internal.HasId
import firrtl.annotations.{CircuitName, ComponentName, ModuleName}

import scala.util.DynamicVariable

/**/
object CrossModule {
  private[aspect] val dynamicContextVar = new DynamicVariable[Option[MultiIOModule]](None)
  def withRoot[S](m: MultiIOModule)(thunk: => S): S = {
    dynamicContextVar.withValue(Some(m))(thunk)
  }
  def getRoot: MultiIOModule = dynamicContextVar.value.get
  private def getTop(m: HasId): BaseModule = (m, m._parent) match {
    case (b: Aspect, None) => getTop(b.parent)
    case (b: BaseModule, None) => b
    case (_, Some(p)) => getTop(p)
    case other => throw new Exception("Cannot find top!")
  }
  implicit def ref2cmr[T<:HasId](ref: T): CrossModuleReference[T] = new CrossModuleReference(ref)
  class CrossModuleReference[T<:HasId](cmr: T){
    def ref: T = cmr
    def r: T = ref
    def getNamed: ComponentName = {
      val circuitName = CircuitName(ref.circuitName)
      val m = dynamicContextVar.value.getOrElse(getTop(cmr))
      val moduleName = ModuleName(m.name, circuitName)
      def getName(h: HasId): Seq[String] = {
        h match {
          case a: Aspect => getName(a.parent) :+ a.instName
          case root: BaseModule if root.name == m.name => Nil
          case other => other._parent.map(getName).getOrElse(Nil) :+ other.instanceName
        }
      }
      ComponentName(getName(ref).mkString("."), moduleName)
    }
    def path: Seq[String] =
      dynamicContextVar.value.map { m =>
        m.name +: ref.pathTo(m)
      }.getOrElse(ref.circuitName +: ref.pathTo(new MultiIOModule {}))
  }
}

