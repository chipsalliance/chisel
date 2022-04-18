// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import java.util.IdentityHashMap

trait Root[+P] extends IsLookupable with Hierarchy[P] {

  private[chisel3] def proxy: RootProxy[P]

  def toDefinition: Definition[P]
  def toRoot: Root[P] = this

  override def proxyAs[T]: DefinitionProxy[P] with T = proxy.asInstanceOf[DefinitionProxy[P] with T]
} 