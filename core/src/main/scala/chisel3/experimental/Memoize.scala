// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir.Scope

final class Memoize[A] private (wrapped: () => A)(implicit sourceInfo: SourceInfo) {

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

    val result = wrapped()

    module._currentScope = oldScope
    result
  }
}

object Memoize {
  def apply[A](body: => A)(implicit sourceInfo: SourceInfo): Memoize[A] = new Memoize(() => body)
}
