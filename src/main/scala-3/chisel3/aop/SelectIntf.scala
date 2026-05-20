// SPDX-License-Identifier: Apache-2.0

package chisel3.aop

import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.internal.firrtl.ir.{DefInstance, DefModule}

import scala.reflect.ClassTag
import scala.collection.mutable

private[aop] trait SelectIntf { self: Select.type =>

  /** Selects all Instances of instances/modules directly instantiated
    * within given module, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T],
    * which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type
    * parameters. E.g. instancesOf[List[Int]] would return
    * List[String].
    *
    * @param parent hierarchy which instantiates the returned Definitions
    */
  def instancesOf[T <: BaseModule: ClassTag](parent: Hierarchy[BaseModule]): Seq[Instance[T]] =
    self._instancesOfImpl[T](_.isA[T])(parent)

  /** Selects all Instances directly and indirectly instantiated within
    * given root hierarchy, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T],
    * which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type
    * parameters. E.g. allInstancesOf[List[Int]] would return
    * List[String].
    *
    * @param root top of the hierarchy to search for instances/modules
    * of given type
    */
  def allInstancesOf[T <: BaseModule: ClassTag](root: Hierarchy[BaseModule]): Seq[Instance[T]] = {
    val soFar = if (root.isA[T]) Seq(root.toInstance.asInstanceOf[Instance[T]]) else Nil
    val allLocalInstances = instancesIn(root)
    soFar ++ (allLocalInstances.flatMap(allInstancesOf[T]))
  }

  /** Selects all Definitions of instances/modules directly instantiated
    * within given module, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T],
    * which will fail if T is an inner class.
    * @note IMPORTANT: this function ignores type
    * parameters. E.g. definitionsOf[List[Int]] would return
    * List[String].
    *
    * @param parent hierarchy which instantiates the returned
    * Definitions
    */
  def definitionsOf[T <: BaseModule: ClassTag](parent: Hierarchy[BaseModule]): Seq[Definition[T]] =
    self._definitionsOfImpl[T](_.isA[T])(parent)

  /** Selects all Definition's directly and indirectly instantiated
    * within given root hierarchy, of provided type
    *
    * @note IMPORTANT: this function requires summoning a ClassTag[T],
    * which will fail if T is an inner class, i.e.  a class defined
    * within another class.
    * @note IMPORTANT: this function ignores type
    * parameters. E.g. allDefinitionsOf[List[Int]] would return
    * List[String].
    *
    * @param root top of the hierarchy to search for definitions of
    * given type
    */
  def allDefinitionsOf[T <: BaseModule: ClassTag](root: Hierarchy[BaseModule]): Seq[Definition[T]] =
    self._allDefinitionsOfImpl[T](_.isA[T])(root)
}
