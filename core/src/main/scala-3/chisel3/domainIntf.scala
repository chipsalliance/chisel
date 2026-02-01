// SPDX-License-Identifier: Apache-2.0

package chisel3.domain

import chisel3._
import chisel3.experimental.SourceInfo

// Empty polyfill for trait needed by Scala 2
private[chisel3] trait domain$Intf

/** Forward a domain from a source to a sink.
  *
  * @param sink the destination of the forward
  * @param source the source of the forward
  */
def define[A <: domain.Type](sink: A, source: A)(using SourceInfo): Unit =
  chisel3.domain._defineImpl(sink, source)

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
def unsafeCast[A <: Data, B <: domain.Type](source: A, domains: B*)(using SourceInfo): A =
  chisel3.domain._unsafeCastImpl(source, domains: _*)
