package chisel3.experimental.hierarchy
import chisel3.experimental.hierarchy.core._

/** @note If we are cloning a non-module, we need another object which has the proper _parent set!
  */
//trait StandInIsInstantiable[T <: IsInstantiable] extends IsStandIn[T]