// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait SparseVec$Intf[A <: Data] { self: SparseVec[A] =>

  /** Read a value from a [[SparseVec]] using one of several possible lookup
    * types. The returned value is read-only.
    *
    * @param addr the address of the value to read from the vec
    * @param lookupType the type of lookup, e.g., binary, one-hot, or when-based
    * @param sourceinfo implicit source locator information
    * @return a read-only value from the specified address
    * @throws ChiselException if the returned value is written to
    */
  def apply(addr: UInt, lookupType: SparseVec.Lookup.Type = SparseVec.Lookup.Binary)(using sourceinfo: SourceInfo): A =
    _applyImpl(addr, lookupType)
}

private[chisel3] trait Lookup$Decoder$Intf { self: SparseVec.Lookup.Decoder =>

  def lookup[A <: Data](
    index:  UInt,
    values: VecLike[A]
  )(
    using sourceinfo: SourceInfo
  ): A = _lookupImpl(index, values)
}
