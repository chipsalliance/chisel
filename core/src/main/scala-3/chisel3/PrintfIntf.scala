// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.experimental.SourceInfo
import chisel3.PrintfMacrosCompat._

import scala.language.experimental.macros

private[chisel3] trait PrintfIntf { self: printf.type =>
  // TODO add printf with format String macro
}
