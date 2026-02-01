// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.Builder
import chisel3.internal.firrtl.ir
import chisel3.experimental.SourceInfo

package object domain extends domain$Intf {

  /** Add a [[Domain]] kind to Chisel's runtime Builder so that it will be
    * unconditionally emitted during FIRRTL emission.
    *
    * @param domain the kind of domain to add
    */
  def addDomain(domain: Domain) = {
    Builder.domains += domain
  }

  private[chisel3] def _defineImpl[A <: domain.Type](sink: A, source: A)(implicit sourceInfo: SourceInfo): Unit = {
    Builder.pushCommand(ir.DomainDefine(sourceInfo, sink.lref, source.ref))
  }

  private[chisel3] def _unsafeCastImpl[A <: Data, B <: domain.Type](source: A, domains: B*)(
    implicit sourceInfo: SourceInfo
  ): A = {
    Builder.pushOp(
      ir.DefPrim(sourceInfo, source.cloneType, ir.PrimOp.UnsafeDomainCast, source.ref +: domains.map(_.ref): _*)
    )
  }

}
