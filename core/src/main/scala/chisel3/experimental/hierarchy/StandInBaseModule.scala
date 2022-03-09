//package chisel3.experimental.hierarchy
//
//import core.{StandIn, IsStandIn}
//import chisel3.experimental.BaseModule
//import chisel3.Data
//import chisel3.internal.PseudoModule
//
//trait LocalBaseModule[T <: BaseModule] extends PseudoModule with LocalStandIn[T] {
//  private[chisel3] def ioMap: Map[Data, Data]
//}
//
//trait LineageBaseModule[T <: BaseModule] extends PseudoModule with LineageStandIn[T] {
//  val lineage: Option[Lineage[BaseModule]]
//  private[chisel3] def instanceName: String = localProxy.standIn.instanceName
//}