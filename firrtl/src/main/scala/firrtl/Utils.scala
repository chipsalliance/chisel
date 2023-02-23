// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.ir._

import _root_.logger.LazyLogging

object Utils extends LazyLogging {

  /** Unwind the causal chain until we hit the initial exception (which may be the first).
    *
    * @param maybeException - possible exception triggering the error,
    * @param first - true if we want the first (eldest) exception in the chain,
    * @return first or last Throwable in the chain.
    */
  def getThrowable(maybeException: Option[Throwable], first: Boolean): Throwable = {
    maybeException match {
      case Some(e: Throwable) => {
        val t = e.getCause
        if (t != null) {
          if (first) {
            getThrowable(Some(t), first)
          } else {
            t
          }
        } else {
          e
        }
      }
      case None | null => null
    }
  }

  /** Throw an internal error, possibly due to an exception.
    *
    * @param message - possible string to emit,
    * @param exception - possible exception triggering the error.
    */
  def throwInternalError(message: String = "", exception: Option[Throwable] = None) = {
    // We'll get the first exception in the chain, keeping it intact.
    val first = true
    val throwable = getThrowable(exception, true)
    val string = if (message.nonEmpty) message + "\n" else message
    error(
      "Internal Error! %sPlease file an issue at https://github.com/ucb-bar/firrtl/issues".format(string),
      throwable
    )
  }

  def time[R](block: => R): (Double, R) = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val timeMillis = (t1 - t0) / 1000000.0
    (timeMillis, result)
  }

  def getUIntWidth(u: BigInt): Int = u.bitLength
  val BoolType = UIntType(IntWidth(1))
  val one = UIntLiteral(1)
  val zero = UIntLiteral(0)

  def sub_type(v: Type): Type = v match {
    case vx: VectorType => vx.tpe
    case vx => UnknownType
  }
  def field_type(v: Type, s: String): Type = v match {
    case vx: BundleType =>
      vx.fields.find(_.name == s) match {
        case Some(f) => f.tpe
        case None    => UnknownType
      }
    case vx => UnknownType
  }

// =================================
  def error(str: String, cause: Throwable = null) = throw new FirrtlInternalException(str, cause)

}
