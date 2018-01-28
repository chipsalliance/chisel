// See LICENSE for license details.

package chisel3.testers2

object Context {
  class Instance(val backend: TesterBackend) {

  }

  def apply(): Instance = context.get

  private var context: Option[Instance] = None
}


