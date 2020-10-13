// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import firrtl.options.Shell

trait ChiselCli { this: Shell =>
  parser.note("Chisel Front End Options")
  Seq( NoRunFirrtlCompilerAnnotation,
       PrintFullStackTraceAnnotation,
       ChiselOutputFileAnnotation,
       ChiselGeneratorAnnotation )
    .foreach(_.addOptions(parser))
}
