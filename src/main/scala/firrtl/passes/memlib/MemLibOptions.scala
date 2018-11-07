// See LICENSE for license details.

package firrtl.passes.memlib

import firrtl._
import firrtl.options.RegisteredLibrary
import scopt.OptionParser

class MemLibOptions extends RegisteredLibrary {
  val name: String = "MemLib Options"
  def addOptions(p: OptionParser[AnnotationSeq]): Unit =
    Seq( new InferReadWrite,
         new ReplSeqMem )
      .map(_.addOptions(p))
}
