// See LICENSE for license details.

package firrtl.passes.memlib

import firrtl._
import firrtl.options.{RegisteredLibrary, ShellOption}
import scopt.OptionParser

class MemLibOptions extends RegisteredLibrary {
  val name: String = "MemLib Options"

  val options: Seq[ShellOption[_]] = Seq( new InferReadWrite,
                     new ReplSeqMem )
    .flatMap(_.options)

}
