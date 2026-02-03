// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.AffectsChiselPrefix

import scala.collection.mutable.ListBuffer

trait NamedModuleTester extends Module with AffectsChiselPrefix {
  val expectedNameMap = ListBuffer[(InstanceId, String)]()
  val expectedModuleNameMap = ListBuffer[(Module, String)]()

  /** Expects some name for a node that is propagated to FIRRTL.
    * The node is returned allowing this to be called inline.
    */
  def expectName[T <: InstanceId](node: T, fullName: String): T = {
    expectedNameMap += ((node, fullName))
    node
  }

  /** Expects some name for a module declaration that is propagated to FIRRTL.
    * The node is returned allowing this to be called inline.
    */
  def expectModuleName[T <: Module](node: T, fullName: String): T = {
    expectedModuleNameMap += ((node, fullName))
    node
  }

  /** After this module has been elaborated, returns a list of (node, expected name, actual name)
    * that did not match expectations.
    * Returns an empty list if everything was fine.
    */
  def getNameFailures(): List[(InstanceId, String, String)] = {
    val failures = ListBuffer[(InstanceId, String, String)]()
    for ((ref, expectedName) <- expectedNameMap) {
      if (ref.instanceName != expectedName) {
        failures += ((ref, expectedName, ref.instanceName))
      }
    }
    for ((mod, expectedModuleName) <- expectedModuleNameMap) {
      if (mod.name != expectedModuleName) {
        failures += ((mod, expectedModuleName, mod.name))
      }
    }
    failures.toList
  }
}
