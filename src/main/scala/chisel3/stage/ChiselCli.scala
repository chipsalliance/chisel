// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import firrtl.options.Shell
import scala.annotation.nowarn

@nowarn("cat=deprecation&msg=WarnReflectiveNamingAnnotation")
trait ChiselCli { this: Shell =>
  parser.note("Chisel Front End Options")
  Seq(
    NoRunFirrtlCompilerAnnotation,
    PrintFullStackTraceAnnotation,
    ThrowOnFirstErrorAnnotation,
    WarnReflectiveNamingAnnotation,
    ChiselOutputFileAnnotation,
    ChiselGeneratorAnnotation
  )
    .foreach(_.addOptions(parser))
}
