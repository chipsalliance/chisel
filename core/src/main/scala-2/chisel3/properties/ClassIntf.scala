// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.experimental.SourceInfo

private[chisel3] trait Class$Intf { self: Class.type =>

  /** Helper to create a DynamicObject for a Class of a given name.
    *
    * *WARNING*: It is the caller's resonsibility to ensure the Class exists, this is not checked automatically.
    */
  def unsafeGetDynamicObject(className: String)(implicit sourceInfo: SourceInfo): DynamicObject =
    _unsafeGetDynamicObjectImpl(className)
}
