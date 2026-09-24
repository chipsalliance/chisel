// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.requireIsHardware
import chisel3.internal.Builder
import chisel3.internal.firrtl.ir
import chisel3.experimental.SourceInfo
import chisel3.reflect.DataMirror

package object domain {

  /** Add a [[Domain]] kind to Chisel's runtime Builder so that it will be
    * unconditionally emitted during FIRRTL emission.
    *
    * @param domain the kind of domain to add
    */
  def addDomain(domain: Domain) = {
    Builder.domains += domain
  }

  /** Forward a domain from a source to a sink.
    *
    * @param sink the destination of the forward
    * @param source the source of the forward
    */
  def define[A <: domain.Type](sink: A, source: A)(implicit sourceInfo: SourceInfo): Unit = {
    Builder.pushCommand(ir.DomainDefine(sourceInfo, sink.lref, source.ref))
  }

  /** Unsafe cast to a variadic list of domains.
    *
    * This is an advanced API that is typically only used when building
    * synchronizer libraries.  E.g., if you are writing a clock domain
    * synchronizer, you need to use this.  If you are using this to work around
    * FIRRTL compilation errors, you may be indavertently hiding bugs.
    *
    * @param source the source Data that should be casted
    * @param domains variadic list of domains to cast to
    */
  def unsafeCast[A <: Data, B <: domain.Type](source: A, domains: B*)(implicit sourceInfo: SourceInfo): A = {
    Builder.pushOp(
      ir.DefPrim(sourceInfo, source.cloneType, ir.PrimOp.UnsafeDomainCast, source.ref +: domains.map(_.ref): _*)
    )
  }

  /** For a hardware type [[Data]], get the domain type of that wire for a given
    * domain kind.
    *
    * @param a some hardware
    * @param kind a kind of domain
    * @return the domain type of `a`
    */
  def domainOf(a: Data, kind: => domain.Type)(implicit sourceInfo: SourceInfo): domain.Type = {
    requireIsHardware(a)
    val _Domain = Wire(kind)
    val _a = WireInit(a)
    Module.currentModule.get.associate(_a, _Domain)
    _Domain
  }

  /** Connect the aligned elements of [[source]] to the aligned elements of [[sink]], with each
    * aligned [[source]] element cast to [[domains]].
    *
    * This is the aligned-direction half of a bidirectional unsafe domain cast.  Pair with
    * [[unsafeConnectFlipped]] to replace `sink :<>= source` across a domain boundary:
    * {{{
    * domain.unsafeConnectAligned(y, x, B)   // aligned: x -> y, cast to B
    * domain.unsafeConnectFlipped(y, x, A)   // flipped: y -> x, cast to A
    * }}}
    *
    * @param sink    the Data whose aligned leaves are driven
    * @param source  the Data whose aligned leaves are the sources (each cast to domains)
    * @param domains variadic list of domains to cast each source element to
    */
  def unsafeConnectAligned[A <: Data](
    sink:    A,
    source:  A,
    domains: domain.Type*
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val sinkElts = DataMirror.collectAlignedDeep(sink) { case e: Element => e }
    val sourceElts = DataMirror.collectAlignedDeep(source) { case e: Element => e }
    require(
      sinkElts.length == sourceElts.length,
      s"unsafeConnectAligned: sink has ${sinkElts.length} aligned element(s) but source has ${sourceElts.length}"
    )
    sinkElts.zip(sourceElts).foreach { case (lhs, rhs) => lhs :<= domain.unsafeCast(rhs, domains: _*) }
  }

  /** Connect the flipped elements of [[source]] to the flipped elements of [[sink]], with each
    * flipped [[sink]] element cast to [[domains]].
    *
    * This is the flipped-direction half of a bidirectional unsafe domain cast.  Pair with
    * [[unsafeConnectAligned]] to replace `sink :<>= source` across a domain boundary:
    * {{{
    * domain.unsafeConnectAligned(y, x, B)   // aligned: x -> y, cast to B
    * domain.unsafeConnectFlipped(y, x, A)   // flipped: y -> x, cast to A
    * }}}
    *
    * @param sink    the Data whose flipped leaves are the sources (each cast to domains)
    * @param source  the Data whose flipped leaves are driven
    * @param domains variadic list of domains to cast each flipped sink element to
    */
  def unsafeConnectFlipped[A <: Data](
    sink:    A,
    source:  A,
    domains: domain.Type*
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val sinkElts = DataMirror.collectFlippedDeep(sink) { case e: Element => e }
    val sourceElts = DataMirror.collectFlippedDeep(source) { case e: Element => e }
    require(
      sinkElts.length == sourceElts.length,
      s"unsafeConnectFlipped: sink has ${sinkElts.length} flipped element(s) but source has ${sourceElts.length}"
    )
    sinkElts.zip(sourceElts).foreach { case (rhs, lhs) => lhs :<= domain.unsafeCast(rhs, domains: _*) }
  }

}
