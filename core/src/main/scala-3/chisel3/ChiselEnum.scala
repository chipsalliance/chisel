// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

abstract class EnumType(factory: ChiselEnum, selfAnnotating: Boolean = true)
    extends EnumTypeImpl(factory, selfAnnotating) {

  final def ===(that: EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_===(that)
  final def =/=(that: EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_=/=(that)
  final def <(that:   EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_<(that)
  final def <=(that:  EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_>(that)
  final def >(that:   EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_<=(that)
  final def >=(that:  EnumType)(implicit sourceInfo: SourceInfo): Bool = _impl_>=(that)
}

abstract class ChiselEnum extends ChiselEnumImpl
