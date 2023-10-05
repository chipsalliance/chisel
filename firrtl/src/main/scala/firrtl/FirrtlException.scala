// SPDX-License-Identifier: Apache-2.0

package firrtl

import scala.util.control.NoStackTrace

/** Exception indicating user error
  *
  * These exceptions indicate a problem due to bad input and thus do not include a stack trace.
  * This can be extended by custom transform writers.
  */
class FirrtlUserException(message: String, cause: Throwable = null)
    extends RuntimeException(message, cause)
    with NoStackTrace

/** Exception indicating something went wrong *within* Firrtl itself
  *
  * These exceptions indicate a problem inside the compiler and include a stack trace to help
  * developers debug the issue.
  *
  * This class is private because these are issues within Firrtl itself. Exceptions thrown in custom
  * transforms are treated differently and should thus have their own structure
  */
private[firrtl] class FirrtlInternalException(message: String, cause: Throwable = null)
    extends Exception(message, cause)
