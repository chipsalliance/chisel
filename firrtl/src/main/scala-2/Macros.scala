// SPDX-License-Identifier: Apache-2.0

package firrtl.macros

import scala.reflect

object Macros {
  def isSingletonImpl(obj: AnyRef): Boolean = {
    reflect.runtime.currentMirror.reflect(obj).symbol.isModuleClass
  }
}
