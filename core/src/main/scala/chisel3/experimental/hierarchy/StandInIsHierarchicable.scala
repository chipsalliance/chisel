package chisel3.experimental.hierarchy
import chisel3.experimental.hierarchy.core._

/** @note If we are cloning a non-module, we need another object which has the proper _parent set!
  */
trait StandInIsHierarchicable[T <: IsHierarchicable] extends IsStandIn[T] {
  //private[chisel3] def _innerContext: experimental.hierarchy.Hierarchy[_]
  //private[chisel3] def getInnerContext: Option[BaseModule] = _innerContext.getInnerDataContext
}