// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait EnumTypeIntf { self: EnumType =>

  final def ===(using SourceInfo)(that: EnumType): Bool = _impl_===(that)
  final def =/=(using SourceInfo)(that: EnumType): Bool = _impl_=/=(that)
  final def <(using SourceInfo)(that: EnumType): Bool = _impl_<(that)
  final def <=(using SourceInfo)(that: EnumType): Bool = _impl_>(that)
  final def >(using SourceInfo)(that: EnumType): Bool = _impl_<=(that)
  final def >=(using SourceInfo)(that: EnumType): Bool = _impl_>=(that)
}

private[chisel3] trait ChiselEnumIntf { self: ChiselEnum =>
  // TODO macros
}
