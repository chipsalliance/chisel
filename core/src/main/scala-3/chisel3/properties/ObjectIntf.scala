// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.experimental.SourceInfo

private[chisel3] trait DynamicObjectObjIntf { self: DynamicObject.type =>

  /** A wrapper method to wrap Class instantiations and return a DynamicObject.
    *
    * This is necessary to help Chisel track internal state. This can be used instead of `Definition.apply` if a
    * DynamicObject is required. If possible, it is safer to user `Definition.apply` and StaticObject.
    *
    * @param bc the Class being created
    *
    * @return a DynamicObject representing an instance of the Class
    */
  def apply[T <: Class](bc: => T)(using SourceInfo): DynamicObject = _applyImpl(bc)
}
