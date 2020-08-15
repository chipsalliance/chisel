package firrtl.passes

import firrtl.DependencyAPIMigration
import firrtl.ir.Circuit
import firrtl.{CircuitState, FirrtlUserException, Transform}

/** [[Pass]] is simple transform that is generally part of a larger [[Transform]]
  * Has an [[UnknownForm]], because larger [[Transform]] should specify form
  */
trait Pass extends Transform with DependencyAPIMigration {
  def run(c:         Circuit): Circuit
  def execute(state: CircuitState): CircuitState = state.copy(circuit = run(state.circuit))
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
      throw new PassExceptions(errors.toSeq)
  }
}
