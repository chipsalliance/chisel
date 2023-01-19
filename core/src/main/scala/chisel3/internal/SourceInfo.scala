// SPDX-License-Identifier: Apache-2.0

// This file contains macros for adding source locators at the point
// of invocation.
//
// This is not part of coreMacros to disallow this macro from being
// implicitly invoked in Chisel frontend (and generating source
// locators in Chisel core), which is almost certainly a bug.
//
// As of Chisel 3.6, these methods are deprecated in favor of the
// public API in chisel3.experimental.

package chisel3.internal

package object sourceinfo {
  import chisel3.experimental._
  import scala.language.experimental.macros
  import scala.reflect.macros.blackbox.Context

  @deprecated(
    "APIs in chisel3.internal are not intended to be public. Use chisel3.experimental.SourceInfo",
    "Chisel 3.6"
  )
  type SourceInfo = chisel3.experimental.SourceInfo

  @deprecated(
    "APIs in chisel3.internal are not intended to be public. Use chisel3.experimental.NoSourceInfo",
    "Chisel 3.6"
  )
  type NoSourceInfo = chisel3.experimental.NoSourceInfo

  @deprecated(
    "APIs in chisel3.internal are not intended to be public. Use chisel3.experimental.UnlocatableSourceInfo",
    "Chisel 3.6"
  )
  val UnlocatableSourceInfo = chisel3.experimental.UnlocatableSourceInfo

  @deprecated(
    "APIs in chisel3.internal are not intended to be public. Use chisel3.experimental.DeprecatedSourceInfo",
    "Chisel 3.6"
  )
  val DeprecatedSourceInfo = chisel3.experimental.DeprecatedSourceInfo

  @deprecated(
    "APIs in chisel3.internal are not intended to be public. Use chisel3.experimental.SourceLine",
    "Chisel 3.6"
  )
  type SourceLine = chisel3.experimental.SourceLine

  @deprecated(
    "APIs in chisel3.internal are not intended to be public. Use chisel3.experimental.SourceInfo",
    "Chisel 3.6"
  )
  object SourceInfo {
    implicit def materialize: SourceInfo = macro SourceInfoMacro.generate_source_info
  }

}
