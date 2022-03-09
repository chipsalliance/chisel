// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

// Wrapper Class
case class Context[+C <: IsContext](context: Option[C]) {
  def get = context.get
}

// Typeclass
//trait Cloner[V]  {
//  type R = Context[C]
//  def apply[P](value: V): R
//}

// Default Typeclass Implementations
//object Contexter {
//  implicit def isContext[V] = new Contexter[V, IsContext] {
//    def apply[P](value: V, hierarchy: Hierarchy[P]): R = {
//      hierarchy.proxy.lookupContext
//    }
//  }
//}
//