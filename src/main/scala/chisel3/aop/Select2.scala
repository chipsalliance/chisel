//// SPDX-License-Identifier: Apache-2.0
//
//package chisel3.aop
//
//import chisel3._
//import chisel3.internal.{HasId}
//import chisel3.experimental.hierarchy.{Definition, Instance, Hierarchy}
//import chisel3.experimental.BaseModule
//import chisel3.experimental.FixedPoint
//import chisel3.internal.firrtl.{Definition => IRDefinition, _}
//import chisel3.internal.PseudoModule
//import chisel3.internal.BaseModule.ModuleClone
//import firrtl.annotations.ReferenceTarget
//
//import scala.collection.mutable
//import scala.reflect.runtime.universe.TypeTag
//import chisel3.internal.naming.chiselName
//
///** Use to select Chisel components in a module, after that module has been constructed
//  * Useful for adding additional Chisel annotations or for use within an [[Aspect]]
//  */
//object Select2 {
//  implicit val mg = new chisel3.internal.MacroGenerated {}
//  // Checks that a module has finished its construction
//  private def check(module: Hierarchy[BaseModule]): Unit = {
//    require(module.getProto.isClosed, "Can't use Selector on modules that have not finished construction!")
//    require(module.getProto._component.isDefined, "Can't use Selector on modules that don't have components!")
//  }
//
//
//  /** Return just leaf components of expanded node
//    *
//    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies are not included
//    * @return
//    */
//  def getLeafs(d: Data): Seq[Data] = d match {
//    case r: Record => r.getElements.flatMap(getLeafs)
//    case v: Vec[_] => v.getElements.flatMap(getLeafs)
//    case other => Seq(other)
//  }
//
//  /** Return all expanded components, including intermediate aggregate nodes
//    *
//    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies ARE included
//    * @return
//    */
//  def getIntermediateAndLeafs(d: Data): Seq[Data] = d match {
//    case r: Record => r +: r.getElements.flatMap(getIntermediateAndLeafs)
//    case v: Vec[_] => v +: v.getElements.flatMap(getIntermediateAndLeafs)
//    case other => Seq(other)
//  }
//
//
//
//
//  /** Selects all ios directly contained within given module
//    * @param module
//    * @return
//    */
//  def ios(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup(p => Select.ios(p))
//
//  /** Selects all SyncReadMems directly contained within given module
//    * @param module
//    * @return
//    */
//  // TODO: Need to add lookupable for SyncReadMem's
//  //def syncReadMems(module: Hierarchy[BaseModule]): Seq[SyncReadMem[_]] = module._lookup(p => Select.syncReadMems(p))
//
//  /** Selects all Mems directly contained within given module
//    * @param module
//    * @return
//    */
//  // TODO: Need to add lookupable for Mem[_]
//  //def mems(module: Hierarchy[BaseModule]): Seq[Mem[_]] = module._lookup(p => Select.mems(p))
//
//  /** Selects all arithmetic or logical operators directly instantiated within given module
//    * @param module
//    * @return
//    */
//  // TODO: Add support for Tuples?
//  //def ops(module: Hierarchy[BaseModule]): Seq[(String, Data)] = module._lookup(p => Select.ops(p))
//
//  /** Selects a kind of arithmetic or logical operator directly instantiated within given module
//    * The kind of operators are contained in [[chisel3.internal.firrtl.PrimOp]]
//    * @param opKind the kind of operator, e.g. "mux", "add", or "bits"
//    * @param module
//    * @return
//    */
//  def ops(opKind: String)(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup(p => Select.ops(opKind)(p))
//
//  /** Selects all wires in a module
//    * @param module
//    * @return
//    */
//  def wires(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup{m => Select.wires(m)}
//
//  /** Selects all memory ports, including their direction and memory
//    * @param module
//    * @return
//    */
//  // TODO: Need to add lookupable for MemPortDirection
//  //def memPorts(module: Hierarchy[BaseModule]): Seq[(Data, MemPortDirection, MemBase[_])] = module._lookup{m => Select.memPorts(m)}
//
//  /** Selects all memory ports of a given direction, including their memory
//    * @param dir The direction of memory ports to select
//    * @param module
//    * @return
//    */
//  // TODO
//  //def memPorts(dir: MemPortDirection)(module: Hierarchy[BaseModule]): Seq[(Data, MemBase[_])] = module._lookup{m => Select.memPorts(dir)(m)}
//
//  /** Selects all components who have been set to be invalid, even if they are later connected to
//    * @param module
//    * @return
//    */
//  def invalids(module: Hierarchy[BaseModule]): Seq[Data] = module._lookup(p => Select.invalids(p))
//
//}
