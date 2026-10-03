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
  private val pendingExternalModules = mutable.ArrayBuffer.empty[BaseModule]

  def add(definition: Definition[_ <: BaseModule]): Unit = definition.proto match {
    case c: Class                                => modules += c
    case c: RawModule                            => modules += c
    case c: Instance.ImportedDefinitionExtModule => importedDefinitions += c.importedDefinition
    case c: BaseBlackBox                         => pendingExternalModules += c
    case c: BaseIntrinsicModule                  => pendingExternalModules += c
    case _ =>
  }

  def contains(definition: Definition[_ <: BaseModule]): Boolean =
    modules.contains(definition.proto) || importedDefinitions.contains(definition) || {
      // toDefinition can register a module before its desiredName has been initialized.
      // Index names at lookup time, processing each registration only once.
      pendingExternalModules.foreach(c => externalModuleNames += c.name)
      pendingExternalModules.clear()
      externalModuleNames.contains(definition.proto.name)
    }
}
