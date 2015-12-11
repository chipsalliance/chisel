// See LICENSE for license details.

package Chisel

import internal._
import internal.Builder.pushCommand
import firrtl._

object assert {
  def apply(cond: Bool, message: String="") {
    when(!cond) {
      if (message.isEmpty()) {
        printf(s"Assertion failed: (TODO: code / lineno)")
      } else {
        printf(s"Assertion failed: (TODO: code / lineno): $message")
      }
      pushCommand(Stop(Node(Builder.dynamicContext.currentModule.get.clock), 1))
    }
  }
}

object printf {
  def apply(fmt: String, data: Bits*) {
    pushCommand(Printf(Node(Builder.dynamicContext.currentModule.get.clock),
        fmt, data.map(Node(_))))
  }
}
