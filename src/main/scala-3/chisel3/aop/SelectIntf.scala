// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.internal.firrtl.ir.{DefInstance, DefModule}

import scala.reflect.ClassTag
import scala.collection.mutable

private[aop] trait SelectIntf { self: Select.type =>

  /** Selects all Instances of instances/modules directly instantiated within given module, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T], which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type parameters. E.g. instancesOf[List[Int]] would return List[String].
    *
    * @param parent hierarchy which instantiates the returned Definitions
    */
  def instancesOf[T <: BaseModule: ClassTag](parent: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    check(parent)
    implicit val mg: chisel3.internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
    parent.proto._component.get match {
      case d: DefModule =>
        collect(d.block.getCommands()) {
          case d: DefInstance =>
            d.id match {
              case p: IsClone[_] =>
                val i = parent._lookup { x => new Instance(Clone(p)).asInstanceOf[Instance[BaseModule]] }
                if (i.isA[T]) Some(i.asInstanceOf[Instance[T]]) else None
              case other: BaseModule =>
                val i = parent._lookup { x => other }
                if (i.isA[T]) Some(i.asInstanceOf[Instance[T]]) else None
            }
          case other => None
        }.flatten
      case other => Nil
    }
  }

  /** Selects all Instances directly and indirectly instantiated within given root hierarchy, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T], which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type parameters. E.g. allInstancesOf[List[Int]] would return List[String].
    *
    * @param root top of the hierarchy to search for instances/modules of given type
    */
  def allInstancesOf[T <: BaseModule: ClassTag](root: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    val soFar = if (root.isA[T]) Seq(root.toInstance.asInstanceOf[Instance[T]]) else Nil
    val allLocalInstances = instancesIn(root)
    soFar ++ (allLocalInstances.flatMap(allInstancesOf[T]))
  }

  /** Selects all Definitions of instances/modules directly instantiated within given module, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T], which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type parameters. E.g. definitionsOf[List[Int]] would return List[String].
    *
    * @param parent hierarchy which instantiates the returned Definitions
    */
  def definitionsOf[T <: BaseModule: ClassTag](parent: Hierarchy[BaseModule]): Seq[Definition[T]] = {
    check(parent)
    implicit val mg: chisel3.internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
    type DefType = Definition[T]
    val defs = parent.proto._component.get match {
      case d: DefModule =>
        collect(d.block.getCommands()) {
          case d: DefInstance =>
            d.id match {
              case p: IsClone[_] =>
                // Use Proto(p.getProto) for consistent Definition equality (same as definitionsIn)
                val d = parent._lookup { x => new Definition(Proto(p.getProto)).asInstanceOf[Definition[BaseModule]] }
                if (d.isA[T]) Some(d.asInstanceOf[Definition[T]]) else None
              case other: BaseModule =>
                val d = parent._lookup { x => other.toDefinition }
                if (d.isA[T]) Some(d.asInstanceOf[Definition[T]]) else None
            }
          case other => None
        }.flatten
    }
    val (_, defList) = defs.foldLeft((Set.empty[DefType], List.empty[DefType])) {
      case ((set, list), definition: Definition[T]) =>
        if (set.contains(definition)) (set, list) else (set + definition, definition +: list)
    }
    defList.reverse
  }

  /** Selects all Definition's directly and indirectly instantiated within given root hierarchy, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T], which will fail if T is an inner class, i.e.
    *   a class defined within another class.
    * @note IMPORTANT: this function ignores type parameters. E.g. allDefinitionsOf[List[Int]] would return List[String].
    *
    * @param root top of the hierarchy to search for definitions of given type
    */
  def allDefinitionsOf[T <: BaseModule: ClassTag](root: Hierarchy[BaseModule]): Seq[Definition[T]] = {
    type DefType = Definition[T]
    val allDefSet = mutable.HashSet[Definition[BaseModule]]()
    val defSet = mutable.HashSet[DefType]()
    val defList = mutable.ArrayBuffer[DefType]()
    def rec(hier: Definition[BaseModule]): Unit = {
      if (hier.isA[T] && !defSet.contains(hier.asInstanceOf[DefType])) {
        defSet += hier.asInstanceOf[DefType]
        defList += hier.asInstanceOf[DefType]
      }
      allDefSet += hier
      val allDefs = definitionsIn(hier)
      allDefs.collect {
        case d if !allDefSet.contains(d) => rec(d)
      }
    }
    rec(root.toDefinition)
    defList.toList
  }
}
