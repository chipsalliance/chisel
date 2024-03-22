// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir.Scope

/** A value that can have Memoized blocks */
sealed trait HasMemoized {
  private[chisel3] def attach(memoize: Memoize[_]): Unit
}

/** HasMemoized is sealed so users can't extend it but we still want to extend it in Chisel */
private[chisel3] trait HasMemoizedImpl extends HasMemoized

final class Memoize[A] private (thunk: () => A)(implicit parent: HasMemoized, sourceInfo: SourceInfo) {

  parent.attach(this)

  private val _scope: Scope = {
    // We must eagerly reserve our spot in the Module's commands
    val scope = new Scope(sourceInfo, isLazy = true)
    Builder.forcedUserModule._currentScope.addCommand(scope)
    scope
  }

  lazy val value: A = {
    _scope.markUsed()

    val module = Builder.forcedUserModule
    val oldScope = module._currentScope

    module._currentScope = _scope

    val result = thunk()

    module._currentScope = oldScope
    result
  }
}

object Memoize {
  def apply[A](body: => A)(implicit parent: HasMemoized, sourceInfo: SourceInfo): Memoize[A] = new Memoize(() => body)
}
