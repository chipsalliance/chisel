// See LICENSE for license details.

package firrtl.options

/** Indicate a generic error in a [[Phase]]
  * @param message exception message
  * @param cause an underlying Exception that this wraps
  */
class PhaseException(val message: String, cause: Throwable = null) extends RuntimeException(message, cause)

/** Indicate an error related to a bad [[firrtl.annotations.Annotation Annotation]] or it's command line option
  * equivalent. This exception is always caught and converted to an error message by a [[Stage]]. Do not use this for
  * communicating generic exception information.
  */
class OptionsException(val message: String, cause: Throwable = null) extends IllegalArgumentException(message, cause)

/** Indicates that a [[Phase]] is missing some mandatory information. This likely occurs either if a user ran something
  * out of order or if the compiler did not run things in the correct order.
  */
class PhasePrerequisiteException(message: String, cause: Throwable = null) extends PhaseException(message, cause)
