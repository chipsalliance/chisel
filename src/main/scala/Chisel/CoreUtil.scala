// See LICENSE for license details.

package Chisel

import internal._
import internal.Builder.pushCommand
import ir.firrtl._

object assert {
  /** Checks for a condition to be valid in the circuit at all times. If the
    * condition evaluates to false, the circuit simulation stops with an error.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using assert make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional message to print when the assertion fires
    */
  def apply(cond: Bool, message: String) {
    when (!Builder.dynamicContext.currentModule.get.reset) {
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

  /** A workaround for default-value overloading problems in Scala, just
    * 'assert(cond, "")' */
  def apply(cond: Bool) {
    assert(cond, "")
  }

  /** An elaboration-time assertion, otherwise the same as the above run-time
    * assertion. */
  def apply(cond: Boolean, message: String) {
    Predef.assert(cond, message)
  }

  /** A workaround for default-value overloading problems in Scala, just
    * 'assert(cond, "")' */
  def apply(cond: Boolean) {
    Predef.assert(cond, "")
  }
}

object printf {
  /** Prints a message in simulation.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using printf make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param fmt printf format string
    * @param data format string varargs containing data to print
    */
  def apply(fmt: String, data: Bits*) {
    when (!Builder.dynamicContext.currentModule.get.reset) {
      pushCommand(Printf(Node(Builder.dynamicContext.currentModule.get.clock),
          fmt, data.map((d: Bits) => d.ref)))
    }
  }
}
