// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}
import chisel3.internal.{Builder, HasId}
import chisel3.internal.firrtl.{GroupDefBegin, GroupDefEnd, Node}
import chisel3.util.simpleClassName
import scala.collection.mutable.LinkedHashSet

/** This object contains Chisel language features for creating optional groups.
  * Optional groups are collections of hardware that are not always present in
  * the circuit.  Optional groups are intended to be used to hold verification
  * or debug code.
  */
object group {

  /** Enumerations of different optional group conventions.  A group convention
    * says how a given group should be lowered to Verilog.
    */
  object Convention {
    sealed trait Type

    /** Internal type used as the parent of all groups. */
    private[chisel3] case object Root extends Type

    /** The group should be lowered to a SystemVerilog `bind`. */
    case object Bind extends Type
  }

  /** A declaration of an optional group.
    *
    * @param convention how this optional group should be lowered
    * @param _parent the parent group, if any
    */
  abstract class Declaration(val convention: Convention.Type)(implicit _parent: Declaration, _sourceInfo: SourceInfo) {
    self: Singleton =>

    /** This establishes a new implicit val for any nested groups. */
    protected final implicit val thiz: Declaration = this

    private[chisel3] def parent: Declaration = _parent

    private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

    private[chisel3] def name: String = simpleClassName(this.getClass())
  }

  object Declaration {
    private[chisel3] case object Root extends Declaration(Convention.Root)(null, UnlocatableSourceInfo)
    implicit val rootDeclaration: Declaration = Root
  }

  /** Add a declaration and all of its parents to the Builder.  This lets the
    * Builder know that this group was used and should be emitted in the FIRRTL.
    */
  private[chisel3] def addDeclarations(declaration: Declaration) = {
    var currentDeclaration: Declaration = declaration
    while (currentDeclaration != Declaration.Root && !Builder.groups.contains(currentDeclaration)) {
      val decl = currentDeclaration
      val parent = decl.parent

      Builder.groups += decl
      currentDeclaration = parent
    }
  }

  /** Create a new optional group definition.
    *
    * @param declaration the optional group declaration this definition is associated with
    * @param block the Chisel code that goes into the group
    * @param sourceInfo a source locator
    */
  def apply[A](
    declaration: Declaration
  )(block:       => A
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    Builder.pushCommand(GroupDefBegin(sourceInfo, declaration))
    addDeclarations(declaration)
    require(
      Builder.groupStack.head == declaration.parent,
      s"nested group '${declaration.name}' must be wrapped in parent group '${declaration.parent.name}'"
    )
    Builder.groupStack = declaration :: Builder.groupStack
    block
    Builder.pushCommand(GroupDefEnd(sourceInfo))
    Builder.groupStack = Builder.groupStack.tail
  }

}
