// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3.RawModule
import chisel3.experimental.{BaseIntrinsicModule, BaseModule}
import chisel3.experimental.hierarchy.core.{Definition, Instance}
import chisel3.properties.Class

import scala.collection.mutable

/** Shared by all Definition elaboration contexts within one circuit. */
private[chisel3] class DefinitionRegistry {
  private val modules = mutable.HashSet.empty[BaseModule]
  private val importedDefinitions = mutable.HashSet.empty[Definition[_ <: BaseModule]]
  private val externalModuleNames = mutable.HashSet.empty[String]
  // Stable IDs avoid invoking user-defined hashCode on unfinished modules.
  private val pendingExternalModules = mutable.HashSet.empty[Long]

  def add(definition: Definition[_ <: BaseModule]): Unit = definition.proto match {
    case c: Class                                => modules += c
    case c: RawModule                            => modules += c
    case c: Instance.ImportedDefinitionExtModule => importedDefinitions += c.importedDefinition
    case _: BaseBlackBox | _: BaseIntrinsicModule =>
      val module = definition.proto
      if (module.isClosed) {
        externalModuleNames += module.name
      } else {
        pendingExternalModules += module._id
      }
    case _ =>
  }

  /** Complete registrations made before the module finished construction. */
  def moduleClosed(module: BaseModule): Unit = {
    if (pendingExternalModules.remove(module._id)) {
      externalModuleNames += module.name
    }
  }

  def contains(definition: Definition[_ <: BaseModule]): Boolean =
    modules.contains(definition.proto) || importedDefinitions.contains(definition) ||
      externalModuleNames.contains(definition.proto.name)
}
