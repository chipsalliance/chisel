// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3._
import chisel3.experimental.hierarchy.{InstantiableClone, ModuleClone}
import chisel3.internal.{throwException, BaseBlackBox, Builder}
import chisel3.experimental.{BaseModule, ExtModule, SourceInfo, UnlocatableSourceInfo}
import chisel3.internal.firrtl.ir.{Component, DefBlackBox, DefClass, DefIntrinsicModule, DefModule, Port}
import chisel3.properties.Class
import firrtl.annotations.IsModule

import scala.annotation.nowarn
import chisel3.experimental.BaseIntrinsicModule

/** User-facing Instance type.
  * Represents a unique instance of type `A` which are marked as @instantiable
  * Can be created using Instance.apply method.
  *
  * @param underlying The internal representation of the instance, which may be either be directly the object, or a clone of an object
  */
final case class Instance[+A] private[chisel3] (private[chisel3] val underlying: Underlying[A])
    extends SealedHierarchy[A] {
  underlying match {
    case Proto(p: IsClone[_]) => chisel3.internal.throwException("Cannot have a Proto with a clone!")
    case other                => // Ok
  }

  /** @return the context of any Data's return from inside the instance */
  private[chisel3] def getInnerDataContext: Option[BaseModule] = underlying match {
    case Proto(value: BaseModule)       => Some(value)
    case Proto(value: IsInstantiable)   => None
    case Clone(i: BaseModule)           => Some(i)
    case Clone(i: InstantiableClone[_]) => i.getInnerContext
    case _                              => throw new InternalErrorException(s"Match error: underlying=$underlying")
  }

  /** @return the context this instance. Note that for non-module clones, getInnerDataContext will be the same as getClonedParent */
  private[chisel3] def getClonedParent: Option[BaseModule] = underlying match {
    case Proto(value: BaseModule)       => value._parent
    case Clone(i: BaseModule)           => i._parent
    case Clone(i: InstantiableClone[_]) => i.getInnerContext
    case _                              => throw new InternalErrorException(s"Match error: underlying=$underlying")
  }

  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the instance.
    *
    * Our @instantiable and @public macros generate the calls to this apply method
    *
    * By calling this function, we summon the proper Lookupable typeclass from our implicit scope.
    *
    * @param that a user-specified lookup function
    * @param lookup typeclass which contains the correct lookup function, based on the types of A and B
    * @param macroGenerated a value created in the macro, to make it harder for users to use this API
    */
  def _lookup[B, C](
    that: A => B
  )(
    implicit lookup: Lookupable[B],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): lookup.C = {
    lookup.instanceLookup(that, this)
  }

  /** Returns the definition of this Instance */
  override def toDefinition: Definition[A] = new Definition(Proto(proto))
  override def toInstance:   Instance[A] = this
  private[chisel3] def copy[T](underlying: Underlying[T]) = new Instance(underlying)

}

/** Factory methods for constructing [[Instance]]s */
object Instance extends SourceInfoDoc {
  implicit class InstanceBaseModuleExtensions[T <: BaseModule](i: Instance[T]) {

    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget: IsModule = i.underlying match {
      case Proto(x: BaseModule)                 => x.getTarget
      case Clone(x: IsClone[_] with BaseModule) => x.getTarget
      case _ => throw new InternalErrorException(s"Match error: i.underlying=${i.underlying}")
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = i.underlying match {
      case Proto(x)                             => x.toAbsoluteTarget
      case Clone(x: IsClone[_] with BaseModule) => x.toAbsoluteTarget
      case _ => throw new InternalErrorException(s"Match error: i.underlying=${i.underlying}")
    }

    /** Returns a FIRRTL ReferenceTarget that references this object, relative to an optional root.
      *
      * If `root` is defined, the target is a hierarchical path starting from `root`.
      *
      * If `root` is not defined, the target is a hierarchical path equivalent to `toAbsoluteTarget`.
      *
      * @note If `root` is defined, and has not finished elaboration, this must be called within `atModuleBodyEnd`.
      * @note The NamedComponent must be a descendant of `root`, if it is defined.
      * @note This doesn't have special handling for Views.
      */
    def toRelativeTarget(root: Option[BaseModule]): IsModule = i.underlying match {
      case Proto(x)                             => x.toRelativeTarget(root)
      case Clone(x: IsClone[_] with BaseModule) => x.toRelativeTarget(root)
      case _ => throw new InternalErrorException(s"Match error: i.underlying=${i.underlying}")
    }

    def toRelativeTargetToHierarchy(root: Option[Hierarchy[BaseModule]]): IsModule = i.underlying match {
      case Proto(x)                             => x.toRelativeTargetToHierarchy(root)
      case Clone(x: IsClone[_] with BaseModule) => x.toRelativeTargetToHierarchy(root)
      case _ => throw new InternalErrorException(s"Match error: i.underlying=${i.underlying}")
    }

    def suggestName(name: String): Unit = i.underlying match {
      case Clone(m: BaseModule) => m.suggestName(name)
      case Proto(m)             => m.suggestName(name)
      case x                    => Builder.exception(s"Cannot call .suggestName on $x")(UnlocatableSourceInfo)
    }
  }

  private class ImportedDefinitionExtModule(
    override val desiredName: String,
    val importedDefinition:   Definition[BaseModule with IsInstantiable]
  ) extends ExtModule {
    override private[chisel3] def _isImportedDefinition = true
    override def generateComponent(): Option[Component] = {
      require(!_closed, s"Can't generate $desiredName module more than once")
      evaluateAtModuleBodyEnd()
      _closed = true
      val firrtlPorts = importedDefinition.proto.getModulePortsAndLocators.map { case (port, sourceInfo) =>
        Port(port, port.specifiedDirection, sourceInfo): @nowarn // Deprecated code allowed for internal use
      }
      val component =
        DefBlackBox(
          this,
          importedDefinition.proto.name,
          firrtlPorts,
          SpecifiedDirection.Unspecified,
          params,
          importedDefinition.proto.layers
        )
      Some(component)
    }
  }

  /** A constructs an [[Instance]] from a [[Definition]]
    *
    * @param definition the Module being created
    * @return an instance of the module definition
    */
  def apply[T <: BaseModule with IsInstantiable](
    definition: Definition[T]
  )(
    implicit sourceInfo: SourceInfo
  ): Instance[T] = {
    // Check to see if the module is already defined internally or externally
    val existingMod = Builder.definitions.view.map(_.proto).exists {
      case c: Class                       => c == definition.proto
      case c: RawModule                   => c == definition.proto
      case c: ImportedDefinitionExtModule => c.importedDefinition == definition
      case c: BaseBlackBox                => c.name == definition.proto.name
      case c: BaseIntrinsicModule         => c.name == definition.proto.name
      case _ => false
    }

    if (!existingMod) {
      // Add a Definition that will get emitted as an ExtModule so that FIRRTL
      // does not complain about a missing element
      val extModName = Builder.importedDefinitionMap.getOrElse(
        definition.proto.name,
        throwException(
          s"Imported Definition information not found for ${definition.proto.name} - possibly forgot to add ImportDefinition annotation?"
        )
      )
      Definition(new ImportedDefinitionExtModule(extModName, definition))
    }

    val ports = experimental.CloneModuleAsRecord(definition.proto)
    val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
    clone._madeFromDefinition = true

    // The definition may have known layers that are not yet known to the
    // Builder.  Add them here.
    definition.proto.layers.foreach(layer.addLayer)

    new Instance(Clone(clone))
  }
  private[chisel3] def apply[T](underlying: Underlying[T]) = new Instance(underlying)
}
