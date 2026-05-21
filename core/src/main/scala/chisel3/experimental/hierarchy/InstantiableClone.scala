// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.experimental.BaseModule

/** @note If we are cloning a non-module, we need another object which has the proper _parent set!
  *
  * The bound is intentionally unrestricted to allow Scala 3 lookups against types marked
  * `@instantiable` without requiring the type to also extend `IsInstantiable`; the Scala 2
  * `@instantiable` macro adds the parent automatically, but the Scala 3 cross-compile path
  * relies on this trait being usable directly.
  */
trait InstantiableClone[T] extends core.IsClone[T] {
  private[chisel3] def _innerContext:   Hierarchy[_]
  private[chisel3] def getInnerContext: Option[BaseModule] = _innerContext.getInnerDataContext
}
