// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.hierarchy.core.Underlying
import scala.annotation.implicitNotFound

@implicitNotFound("These functions are only for building hierarchy-compatible Chisel libraries! Users beware!")
// DO NOT extend unless you know what you are doing!!!!!! Not for the casual user!
trait InsideHierarchyLibraryExtension

// Collection of public functions to give non-core-Chisel libraries the ability to build integrations with
// the experimental hierarchy package
object LibraryHooks {

  /** Builds a new instance given a definition and function to create a new instance-specific Underlying, from the
    * definition's Underlying
    * @note Implicitly requires being inside a Hierarchy Library Extension
    */
  def buildInstance[A](
    definition:       Definition[A],
    createUnderlying: Underlying[A] => Underlying[A]
  )(
    implicit inside: InsideHierarchyLibraryExtension
  ): Instance[A] = {
    new Instance(createUnderlying(definition.underlying))
  }

  /** Builds a new definition given an Underlying implementation
    * @note Implicitly requires being inside a Hierarchy Library Extension
    */
  def buildDefinition[A](underlying: Underlying[A])(implicit inside: InsideHierarchyLibraryExtension): Definition[A] = {
    new Definition(underlying)
  }
}
