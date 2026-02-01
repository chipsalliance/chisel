// SPDX-License-Identifier: Apache-2.0

package chisel3
package connectable

import chisel3.experimental.SourceInfo

private[chisel3] trait ConnectableOpExtensionIntf[T <: Data] { self: Connectable.ConnectableOpExtension[T] =>

  final def :<=[S <: Data](lProducer: => S)(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonLessEqDataImpl(lProducer)

  final def :<=[S <: Data](producer: Connectable[S])(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonLessEqConnectableImpl(producer)

  final def :>=[S <: Data](lProducer: => S)(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonGreaterEqDataImpl(lProducer)

  final def :>=[S <: Data](producer: Connectable[S])(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonGreaterEqConnectableImpl(producer)

  final def :<>=[S <: Data](lProducer: => S)(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonLessGreaterEqDataImpl(lProducer)

  final def :<>=[S <: Data](producer: Connectable[S])(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonLessGreaterEqConnectableImpl(producer)

  final def :#=[S <: Data](lProducer: => S)(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonHashEqDataImpl(lProducer)

  final def :#=[S <: Data](producer: Connectable[S])(using evidence: T =:= S, sourceInfo: SourceInfo): Unit =
    _colonHashEqConnectableImpl(producer)

  final def :<=(producer: DontCare.type)(using SourceInfo): Unit =
    _colonLessEqDontCareImpl(producer)

  final def :>=(producer: DontCare.type)(using SourceInfo): Unit =
    _colonGreaterEqDontCareImpl(producer)

  final def :<>=(producer: DontCare.type)(using SourceInfo): Unit =
    _colonLessGreaterEqDontCareImpl(producer)

  final def :#=(producer: DontCare.type)(using SourceInfo): Unit =
    _colonHashEqDontCareImpl(producer)
}
