// SPDX-License-Identifier: Apache-2.0

package firrtl.options

/** The supertype of all exit codes */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed trait ExitCode { val number: Int }

/** [[ExitCode]] indicating success */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object ExitSuccess extends ExitCode { val number = 0 }

/** An [[ExitCode]] indicative of failure. This must be non-zero and should not conflict with a reserved exit code. */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed trait ExitFailure extends ExitCode

/** An exit code indicating a general, non-specific error */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object GeneralError extends ExitFailure { val number = 1 }
