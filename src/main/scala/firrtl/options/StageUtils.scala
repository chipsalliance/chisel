// SPDX-License-Identifier: Apache-2.0

package firrtl.options

/** Utilities related to working with a [[Stage]] */
object StageUtils {

  /** Print a warning message (in yellow)
    * @param message error message
    */
  def dramaticWarning(message: String): Unit = {
    println(Console.YELLOW + "-" * 78)
    println(s"Warning: $message")
    println("-" * 78 + Console.RESET)
  }

  /** Print an error message (in red)
    * @param message error message
    * @note This does not stop the Driver.
    */
  def dramaticError(message: String): Unit = {
    println(Console.RED + "-" * 78)
    println(s"Error: $message")
    println("-" * 78 + Console.RESET)
  }

  /** Generate a message suggesting that the user look at the usage text.
    * @param message the error message
    */
  def dramaticUsageError(message: String): Unit =
    dramaticError(s"""|$message
                      |Try --help for more information.""".stripMargin)

}
