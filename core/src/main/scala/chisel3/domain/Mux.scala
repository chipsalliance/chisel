// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3._
import chisel3.experimental.SourceInfo

/** A (glitchy) domain-aware mux
  *
  * Muxes any `Data` and associates the muxed output with a _new_ domain
  * supplied by the caller (e.g. `ClockDomain("_muxed")`).  This is intended to
  * be used as a very basic domain mux, e.g., as a clock mux.  However, this can
  * also be used to mux multiple signals at once, e.g., to mux between both a
  * clock/reset pair and have the resulting pair be on the same domain.
  *
  * @param sel       the mux select
  * @param outT      the value selected when `sel` is true
  * @param outF      the value selected when `sel` is false
  * @param outDomain the output domain to associate the muxed output with
  * @return the muxed output data and the output domain it was associated with
  *
  * @note This is a glitchy!  This is not safe to use if `sel` will change while
  * `outT` or `outF` are toggling.
  */
object Mux {
  def apply[A <: Data](
    sel:        Bool,
    outT:       A,
    outF:       A,
    outDomain:  domain.Type,
    outDomains: domain.Type*
  )(implicit sourceInfo: SourceInfo): A = {
    val out = Wire(chiselTypeOf(outT))
    val domains = outDomain +: outDomains
    Module.currentModule.get.associate(out, domains: _*)
    out := chisel3.Mux(
      domain.unsafeCast(sel, domains:  _*),
      domain.unsafeCast(outT, domains: _*),
      domain.unsafeCast(outF, domains: _*)
    )
    out
  }
}
