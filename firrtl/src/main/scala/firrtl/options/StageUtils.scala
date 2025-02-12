// SPDX-License-Identifier: Apache-2.0

package firrtl.options

/** Utilities related to working with a [[Stage]] */
object StageUtils {

  /** Construct a message with an optional header and body.  Demarcate the body
    * with separators appropriate for most terminals.
    *
    * @param header an optional header to include before the separatot
    * @param body the body of the message
    * @return a string containing the complete message
    */
  def dramaticMessage(header: Option[String], body: String): String = {
    s"""|${header.map(_ + "\n").getOrElse("")}${"-" * 78}
        |$body
        |${"-" * 78}""".stripMargin
  }

  /** Print a warning message (in yellow)
    * @param message error message
    */
  def dramaticWarning(message: String): Unit = {
    println(Console.YELLOW + dramaticMessage(header=None, body=s"Warning: $message") + Console.RESET)
  }

  /** Print an error message (in red)
    * @param message error message
    * @note This does not stop the Driver.
    */
  def dramaticError(message: String): Unit = {
    println(Console.RED + dramaticMessage(header=None, body=s"Error: $message") + Console.RESET)
  }

  /** Generate a message suggesting that the user look at the usage text.
    * @param message the error message
    */
  def dramaticUsageError(message: String): Unit =
    dramaticError(s"""|$message
                      |Try --help for more information.""".stripMargin)

}
