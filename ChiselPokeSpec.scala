// See LICENSE for license details.

package chisel3.iotesters.experimental

import org.scalatest._

import chisel3._
import chisel3.iotesters._

sealed trait TesterBackend {
  def create[T <: Module](dutGen: () => T, options: TesterOptionsManager): (T, Backend)
}
case object FirrtlInterpreterBackend extends TesterBackend {
  override def create[T <: Module](dutGen: () => T, options: TesterOptionsManager): (T, Backend) = {
    setupFirrtlTerpBackend(dutGen, options)
  }
}
case object VerilatorBackend extends TesterBackend {
  override def create[T <: Module](dutGen: () => T, options: TesterOptionsManager): (T, Backend) = {
    setupVerilatorBackend(dutGen, options)
  }
}
case object VcsBackend extends TesterBackend {
  override def create[T <: Module](dutGen: () => T, options: TesterOptionsManager): (T, Backend) = {
    setupVCSBackend(dutGen, options)
  }
}

trait ChiselPokeTesterUtils extends Assertions {
  class InnerTester(val backend: Backend, val options: TesterOptionsManager) {
    // Implicit configuration options for backend
    implicit val logger = System.out  // TODO: this should be parsed in OptionsManager
    implicit val verbose = options.testerOptions.isVerbose
    implicit val displayBase = options.testerOptions.displayBase

    // Circuit state
    private var currCycle = 0

    // TODO: statically-typed Bundle constructors
    // Map-based Bundle expect/pokes currently not supported because those don't compile-time check

    def expect(ref: Bits, value: BigInt, msg: String="") {
      val actualValue = backend.peek(ref, None)
      val postfix = if (msg != "") s": $msg" else ""
      assert(actualValue == value, s"(cycle $currCycle: expected ${ref.instanceName} == $value, got $actualValue$postfix)")
    }

    /** Write a value into the circuit.
      */
    def poke(ref: Bits, value: BigInt) {
      assert(!ref.isLit, s"(attempted to poke literal ${ref.instanceName})")
      backend.poke(ref, value, None)
      val verifyVal = backend.peek(ref, None)
      assert(verifyVal == value, s"(poke failed on ${ref.instanceName} <= $value, read back $verifyVal)")
    }

    /** Steps the circuit by the specified number of clock cycles.
      */
    def step(cycles: Int = 1) {
      require(cycles > 0)
      backend.step(cycles)
      currCycle += cycles
    }

    /** Hold the circuit in reset for the specified number of clock cycles.
      */
    def reset(cycles: Int = 1) {
      require(cycles > 0)
      backend.reset(cycles)
      currCycle += cycles
    }

    private[iotesters] def finish {
      try {
        backend.finish
      } catch {
        case e: TestApplicationException => assert(e.exitVal == 0, s"nonzero simulator exit code ${e.exitVal}")
      }
    }

    // Explicitly no peek is given to allow generation of static testbenches.
    // Dynamic testbenches may be a specialized option later.
    /** Internal: read a value into the circuit.
      */
    private[iotesters] def peek(ref: Bits): BigInt = {
      backend.peek(ref, None)
    }
  }

  /** Instantiates a tester from a module generator, using default Tester options.
    */
  protected def runTester[T <: Module](dutGen: => T, testerBackend: TesterBackend, options: TesterOptionsManager)(block: (InnerTester, T) => Unit) {
    val dutGenShim: () => T = () => dutGen
    val (dut, backend) = testerBackend.create(dutGenShim, options)
    val innerTester = new InnerTester(backend, options)
    try {
      block(innerTester, dut)
    } catch { case e: Throwable =>
      innerTester.finish
      throw e
    }
    innerTester.finish
  }
}

/** Basic peek-poke test system where failures are handled and reported within ScalaTest.
  */
trait PokeTester extends ChiselPokeTesterUtils {
  def test[T <: Module](dutGen: => T, testerBackend: TesterBackend, options: TesterOptionsManager)(block: (InnerTester, T) => Unit) {
    runTester(dutGen, testerBackend, options) { (tester, dut) => block(tester, dut) }
  }

  def test[T <: Module](dutGen: => T, testerBackend: TesterBackend=FirrtlInterpreterBackend)(block: (InnerTester, T) => Unit) {
    val options = new TesterOptionsManager
    test(dutGen, FirrtlInterpreterBackend, options)(block)
  }
}

/** EXPERIMENTAL test system that provides a more ScalaTest-ey way of specifying tests, making
  * heavy use of implicits to reduce boilerplate.
  *
  * API very subject to change.
  */
trait ImplicitPokeTester extends ChiselPokeTesterUtils {
  /** Pokes a value into the circuit.
    */
  def poke(ref: Bits, value: BigInt)(implicit t: InnerTester) {
    t.poke(ref, value)
  }

  // Wrapper for poke with Bool/Boolean types.
  def poke(ref: Bool, value: Boolean)(implicit t: InnerTester) {
    poke(ref, boolean2BigInt(value))
  }

  // Wrapper for check when no explicit message is passed in.
  // Scala doesn't allow multiple overloaded functions with default arguments.
  def check(ref: Bits, value: BigInt)(implicit t: InnerTester) {
    check(ref, value, "")
  }
  /** Asserts that the node's simulation value is equal to the given value.
    */
  def check(ref: Bits, value: BigInt, msg: String)(implicit t: InnerTester) {
    t.expect(ref, value, msg)
  }

  // Wrapper for check with Bool/Boolean with no explicit message.
  // Scala doesn't allow multiple overloaded functions with default arguments.
  def check(ref: Bool, value: Boolean)(implicit t: InnerTester) {
    check(ref, value, "")
  }
  // Wrapper for check with Bool/Boolean
  // Scala doesn't allow multiple overloaded functions with default arguments.
  def check(ref: Bool, value: Boolean, msg: String)(implicit t: InnerTester) {
    check(ref, boolean2BigInt(value), msg)
  }

  def boolean2BigInt(in: Boolean) = in match {
    case true => BigInt(1)
    case false => BigInt(0)
  }

  /** Steps the top-level clock by some number (default 1) of clock cycles.
    */
  def step(cycles: Int = 1)(implicit t: InnerTester) {
    t.step(cycles)
  }
  /** Holds the design in reset for some number (default 1) of clock cycles.
    */
  def reset(cycles: Int = 1)(implicit t: InnerTester) {
    t.reset(cycles)
  }

  /** The advanced version of test, allowing custom options and requiring a custom backend.
   */
  def test[T <: Module](dutGen: => T, testerBackend: TesterBackend, options: TesterOptionsManager)(block: InnerTester => (T => Unit)) {
    runTester(dutGen, testerBackend, options) { (tester, dut) => block(tester)(dut) }
  }

  /** Runs a test: runs the DUT generator, compiles it down to the requested backend, and runs the
    * test sequence.
    *
    * This is the simple version, which uses default options.
    *
    * @example {{{
    * test(new MyDut) {implicit t => c =>
    *   poke(c.io.in, 0x41)
    *   step()
    *   check(c.io.out, 0x42)
    * }
    * }}}
    */
  def test[T <: Module](dutGen: => T, testerBackend: TesterBackend=FirrtlInterpreterBackend)(block: InnerTester => (T => Unit)) {
    val options = new TesterOptionsManager
    test(dutGen, testerBackend, options)(block)
  }
}
