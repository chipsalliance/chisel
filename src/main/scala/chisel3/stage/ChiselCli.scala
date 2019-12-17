// See LICENSE for license details.

package chisel3.stage

import firrtl.options.Shell

trait ChiselCli { this: Shell =>
  parser.note("Chisel Front End Options")
  Seq( NoRunFirrtlCompilerAnnotation,
       PrintFullStackTraceAnnotation,
       ChiselGeneratorAnnotation )
    .foreach(_.addOptions(parser))
}
