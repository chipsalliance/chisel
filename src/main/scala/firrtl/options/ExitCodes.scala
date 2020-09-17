// SPDX-License-Identifier: Apache-2.0

package firrtl.options

/** The supertype of all exit codes */
sealed trait ExitCode { val number: Int }

/** [[ExitCode]] indicating success */
object ExitSuccess extends ExitCode { val number = 0 }

/** An [[ExitCode]] indicative of failure. This must be non-zero and should not conflict with a reserved exit code. */
sealed trait ExitFailure extends ExitCode

/** An exit code indicating a general, non-specific error */
object GeneralError extends ExitFailure { val number = 1 }
