// SPDX-License-Identifier: Apache-2.0

/** The Chisel compatibility package allows legacy users to continue using the `Chisel` (capital C) package name
  *  while moving to the more standard package naming convention `chisel3` (lowercase c).
  */
import chisel3._ // required for implicit conversions.
import chisel3.util.random.FibonacciLFSR
import chisel3.stage.{phases, ChiselCircuitAnnotation, ChiselOutputFileAnnotation}

import circt.stage.ChiselStage

import scala.annotation.nowarn
