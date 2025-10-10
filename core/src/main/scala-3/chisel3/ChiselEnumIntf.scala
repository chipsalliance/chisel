// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait EnumTypeIntf { self: EnumType =>

  final def ===(that: EnumType)(using SourceInfo): Bool = _impl_===(that)
  final def =/=(that: EnumType)(using SourceInfo): Bool = _impl_=/=(that)
  final def <(that:   EnumType)(using SourceInfo): Bool = _impl_<(that)
  final def <=(that:  EnumType)(using SourceInfo): Bool = _impl_>(that)
  final def >(that:   EnumType)(using SourceInfo): Bool = _impl_<=(that)
  final def >=(that:  EnumType)(using SourceInfo): Bool = _impl_>=(that)
}

private[chisel3] trait ChiselEnumIntf { self: ChiselEnum =>
  // TODO macros
}

private[chisel3] trait OneHotEnumIntf extends ChiselEnumIntf { self: OneHotEnum =>
  // TODO macros
}
