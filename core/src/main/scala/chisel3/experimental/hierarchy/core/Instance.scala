// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{InstanceTransform}
import java.util.IdentityHashMap

import scala.language.experimental.macros
import chisel3._
import chisel3.experimental.hierarchy.{InstantiableClone, ModuleClone}
import chisel3.internal.Builder
import chisel3.internal.sourceinfo.{InstanceTransform, SourceInfo}
import chisel3.experimental.{BaseModule, ExtModule}
import chisel3.internal.firrtl.{Component, DefBlackBox, DefModule, Port}
import firrtl.annotations.IsModule
import chisel3.internal.throwException

/** Represents an Instance of a proto, from a specific hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class Instance[+P] private[chisel3] (private[chisel3] proxy: InstanceProxy[P]) extends Hierarchy[P] {

  override def toRoot = proxy.toRoot

  override def proxyAs[T]: InstanceProxy[P] with T = proxy.asInstanceOf[InstanceProxy[P] with T]
}

object Instance {
  def apply[P](root: Root[P]): Instance[P] =
    macro InstanceTransform.apply[P]
  def do_apply[P](root: Root[P])(implicit extensions: HierarchicalExtensions[P, _]): Instance[P] = {
    new Instance(extensions.buildInstance(root))
  }
  

  ///** A constructs an [[Instance]] from a [[Definition]]
  //  *
  //  * @param definition the Module being created
  //  * @return an instance of the module definition
  //  */
  //def apply[T <: BaseModule with IsInstantiable](definition: Definition[T]): Instance[T] =
  //  macro InstanceTransform.apply[T]

  ///** A constructs an [[Instance]] from a [[Definition]]
  //  *
  //  * @param definition the Module being created
  //  * @return an instance of the module definition
  //  */
  //def do_apply[T <: BaseModule with IsInstantiable](
  //  definition: Definition[T]
  //)(
  //  implicit sourceInfo: SourceInfo,
  //  compileOptions:      CompileOptions
  //): Instance[T] = {
  //  // Check to see if the module is already defined internally or externally
  //  val existingMod = Builder.components.map {
  //    case c: DefModule if c.id == definition.proto          => Some(c)
  //    case c: DefBlackBox if c.name == definition.proto.name => Some(c)
  //    case _ => None
  //  }.flatten

  //  if (existingMod.isEmpty) {
  //    // Add a Definition that will get emitted as an ExtModule so that FIRRTL
  //    // does not complain about a missing element
  //    val extModName = Builder.importDefinitionMap.getOrElse(
  //      definition.proto.name,
  //      throwException(
  //        "Imported Definition information not found - possibly forgot to add ImportDefinition annotation?"
  //      )
  //    )
  //    class EmptyExtModule extends ExtModule {
  //      override def desiredName: String = extModName
  //      override def generateComponent(): Option[Component] = {
  //        require(!_closed, s"Can't generate $desiredName module more than once")
  //        _closed = true
  //        val firrtlPorts = definition.proto.getModulePorts.map { port => Port(port, port.specifiedDirection) }
  //        val component = DefBlackBox(this, definition.proto.name, firrtlPorts, SpecifiedDirection.Unspecified, params)
  //        Some(component)
  //      }
  //    }
  //    Definition(new EmptyExtModule() {})
  //  }

  //  val ports = experimental.CloneModuleAsRecord(definition.proto)
  //  val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
  //  clone._madeFromDefinition = true

  //  new Instance(Clone(clone))
  //}

}
