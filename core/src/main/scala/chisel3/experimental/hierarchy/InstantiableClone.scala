// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule

/** @note If we are cloning a non-module, we need another object which has the proper _parent set!
  */
trait InstantiableClone[T <: IsInstantiable] extends core.IsClone[T] {
  private[chisel3] def _innerContext: Hierarchy[_]
  private[chisel3] def getInnerContext: Option[BaseModule] = _innerContext.getInnerDataContext
}
