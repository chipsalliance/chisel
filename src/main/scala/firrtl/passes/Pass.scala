package firrtl.passes

import firrtl.Utils.error
import firrtl.ir.Circuit
import firrtl.{CircuitForm, CircuitState, FirrtlUserException, Transform, UnknownForm}

/** [[Pass]] is simple transform that is generally part of a larger [[Transform]]
  * Has an [[UnknownForm]], because larger [[Transform]] should specify form
  */
trait Pass extends Transform {
  def inputForm: CircuitForm = UnknownForm
  def outputForm: CircuitForm = UnknownForm
  def run(c: Circuit): Circuit
  def execute(state: CircuitState): CircuitState = {
    val result = (state.form, inputForm) match {
      case (_, UnknownForm) => run(state.circuit)
      case (UnknownForm, _) => run(state.circuit)
      case (x, y) if x > y =>
        error(s"[$name]: Input form must be lower or equal to $inputForm. Got ${state.form}")
      case _ => run(state.circuit)
    }
    CircuitState(result, outputForm, state.annotations, state.renames)
  }
}

// Error handling
class PassException(message: String) extends FirrtlUserException(message)
class PassExceptions(val exceptions: Seq[PassException]) extends FirrtlUserException("\n" + exceptions.mkString("\n"))
class Errors {
  val errors = collection.mutable.ArrayBuffer[PassException]()
  def append(pe: PassException) = errors.append(pe)
  def trigger() = errors.size match {
    case 0 =>
    case 1 => throw errors.head
    case _ =>
      append(new PassException(s"${errors.length} errors detected!"))
      throw new PassExceptions(errors)
  }
}
