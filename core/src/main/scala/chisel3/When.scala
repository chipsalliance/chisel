// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl.ir._
import chisel3.experimental.{SourceInfo, UnlocatableSourceInfo}

object when extends when$Intf {

  private[chisel3] def _applyImpl(
    cond: => Bool
  )(block: => Any)(
    implicit sourceInfo: SourceInfo
  ): WhenContext = {
    new WhenContext(sourceInfo, () => cond, block, Nil)
  }

  /** Returns the current `when` condition
    *
    * This is the conjunction of conditions for all `whens` going up the call stack
    * {{{
    * when (a) {
    *   when (b) {
    *     when (c) {
    *     }.otherwise {
    *       when.cond // this is equal to: a && b && !c
    *     }
    *   }
    * }
    * }}}
    */
  def cond: Bool = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val whens = Builder.whenStack
    whens.foldRight(true.B) { case (ctx, acc) =>
      acc && ctx.localCond
    }
  }
}

private object Scope {
  sealed trait Type

  /** Indicates that the `WhenContext` is in the "if" clause. */
  object If extends Type {
    override def toString: String = "if"
  }

  /** Indicates that the `WhenContext` is in the "else" clause. */
  object Else extends Type {
    override def toString: String = "else"
  }
}

/**  A WhenContext may represent a when, and elsewhen, or an
  *  otherwise. Since FIRRTL does not have an "elsif" statement,
  *  alternatives must be mapped to nested if-else statements inside
  *  the alternatives of the preceeding condition. In order to emit
  *  proper FIRRTL, it is necessary to keep track of the depth of
  *  nesting of the FIRRTL whens. Due to the "thin frontend" nature of
  *  Chisel3, it is not possible to know if a when or elsewhen has a
  *  succeeding elsewhen or otherwise; therefore, this information is
  *  added by preprocessing the command queue.
  */
final class WhenContext private[chisel3] (
  _sourceInfo: SourceInfo,
  cond:        () => Bool,
  block:       => Any,
  // For capturing conditions from prior whens or elsewhens
  altConds: List[() => Bool]
) extends WhenContext$Intf {

  /** Indicate if the `WhenContext` is "closed" (`None`) or if this is writing to
    * the "if" or "else" region.
    */
  private var scope: Option[Scope.Type] = None

  private[chisel3] def sourceInfo: SourceInfo = _sourceInfo

  /** Returns the local condition, inverted for an otherwise */
  private[chisel3] def localCond: Bool = {
    implicit val sourceInfo = UnlocatableSourceInfo
    val alt = altConds.foldRight(true.B) { case (c, acc) =>
      acc & !c()
    }
    scope match {
      case Some(Scope.If)   => alt && cond()
      case Some(Scope.Else) => alt && !cond()
      case None             => alt
    }
  }

  private[chisel3] def _elsewhenImpl(
    elseCond: => Bool
  )(block: => Any)(
    implicit sourceInfo: SourceInfo
  ): WhenContext = {
    Builder.forcedUserModule.withRegion(whenCommand.elseRegion) {
      new WhenContext(sourceInfo, () => elseCond, block, cond :: altConds)
    }
  }

  private[chisel3] def _otherwiseImpl(block: => Any)(implicit sourceInfo: SourceInfo): Unit = {
    Builder.pushWhen(this)
    scope = Some(Scope.Else)
    Builder.forcedUserModule.withRegion(whenCommand.elseRegion) {
      block
    }
    scope = None
    Builder.popWhen()
  }

  /** Return true if this `WhenContext` is currently constructing operations. */
  def active: Boolean = scope.isDefined

  // Create the `When` operation and run the `block` thunk inside the
  // `ifRegion`.  Any commands that this thunk creates will be put inside this
  // block.
  private val whenCommand = pushCommand(new When(sourceInfo, cond().ref(sourceInfo)))
  Builder.pushWhen(this)
  scope = Some(Scope.If)
  try {
    Builder.forcedUserModule.withRegion(whenCommand.ifRegion) {
      block
    }
  } catch {
    case _: scala.runtime.NonLocalReturnControl[_] =>
      throwException(
        "Cannot exit from a when() block with a \"return\"!" +
          " Perhaps you meant to use Mux or a Wire as a return value?"
      )
  }
  scope = None
  Builder.popWhen()
}
