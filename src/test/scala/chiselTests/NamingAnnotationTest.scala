// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.internal.InstanceId
import chisel3.experimental.{chiselName, dump, MultiIOModule}
import org.scalatest._
import org.scalatest.prop._
import chisel3.testers.BasicTester

import scala.collection.mutable.ListBuffer

trait NamedModuleTester extends MultiIOModule {
  val expectedNameMap = ListBuffer[(InstanceId, String)]()
  val expectedModuleNameMap = ListBuffer[(Module, String)]()

  /** Expects some name for a node that is propagated to FIRRTL.
    * The node is returned allowing this to be called inline.
    */
  def expectName[T <: InstanceId](node: T, fullName: String): T = {
    expectedNameMap += ((node, fullName))
    node
  }

  /** Expects some name for a module declaration that is propagated to FIRRTL.
    * The node is returned allowing this to be called inline.
    */
  def expectModuleName[T <: Module](node: T, fullName: String): T = {
    expectedModuleNameMap += ((node, fullName))
    node
  }

  /** After this module has been elaborated, returns a list of (node, expected name, actual name)
    * that did not match expectations.
    * Returns an empty list if everything was fine.
    */
  def getNameFailures(): List[(InstanceId, String, String)] = {
    val failures = ListBuffer[(InstanceId, String, String)]()
    for ((ref, expectedName) <- expectedNameMap) {
      if (ref.instanceName != expectedName) {
        failures += ((ref, expectedName, ref.instanceName))
      }
    }
    for ((mod, expectedModuleName) <- expectedModuleNameMap) {
      if (mod.name != expectedModuleName) {
        failures += ((mod, expectedModuleName, mod.name))
      }
    }
    failures.toList
  }
}

@chiselName
class NamedModule extends NamedModuleTester {
  @chiselName
  def FunctionMockupInner(): UInt = {
    val my2A = 1.U
    val my2B = expectName(my2A +& 2.U, "test_myNested_my2B")
    val my2C = my2B +& 3.U  // should get named at enclosing scope
    my2C
  }

  @chiselName
  def FunctionMockup(): UInt = {
    val myNested = expectName(FunctionMockupInner(), "test_myNested")
    val myA = expectName(1.U + myNested, "test_myA")
    val myB = expectName(myA +& 2.U, "test_myB")
    val myC = expectName(myB +& 3.U, "test_myC")
    myC +& 4.U  // named at enclosing scope
  }

  // chiselName "implicitly" applied
  def ImplicitlyNamed(): UInt = {
    val implicitA = expectName(1.U + 2.U, "test3_implicitA")
    val implicitB = expectName(implicitA + 3.U, "test3_implicitB")
    implicitB + 2.U  // named at enclosing scope
  }

  // Ensure this applies a partial name if there is no return value
  def NoReturnFunction() {
    val noreturn = expectName(1.U + 2.U, "noreturn")
  }


  val test = expectName(FunctionMockup(), "test")
  val test2 = expectName(test +& 2.U, "test2")
  val test3 = expectName(ImplicitlyNamed(), "test3")

  // Test that contents of for loops are named
  for (i <- 0 until 1) {
    val forInner = expectName(test3 + i.U, "forInner")
  }

  // Test that contents of anonymous functions are named
  Seq((0, "anonInner"), (1, "anonInner_1"), (2, "anonInner_2")).foreach { case (in, name) =>
    val anonInner = expectName(test3 + in.U, name)
  }

  NoReturnFunction()
}

@chiselName
class NameCollisionModule extends NamedModuleTester {
  @chiselName
  def repeatedCalls(id: Int): UInt = {
     val test = expectName(1.U + 3.U, s"test_$id")  // should disambiguate by invocation order
     test + 2.U
  }

  // chiselName applied by default to this
  def innerNamedFunction() {
    // ... but not this inner function
    def innerUnnamedFunction() {
      val a = repeatedCalls(1)
      val b = repeatedCalls(2)
    }

    innerUnnamedFunction()
  }

  val test = expectName(1.U + 2.U, "test")
  innerNamedFunction()
}

/** Ensure no crash happens if a named function is enclosed in a non-named module
  */
class NonNamedModule extends NamedModuleTester {
  @chiselName
  def NamedFunction(): UInt = {
    val myVal = 1.U + 2.U
    myVal
  }

  val test = NamedFunction()
}

/** Ensure no crash happens if a named function is enclosed in a non-named function in a named
  * module.
  */
object NonNamedHelper {
  @chiselName
  def NamedFunction(): UInt = {
    val myVal = 1.U + 2.U
    myVal
  }

  def NonNamedFunction() : UInt = {
    val myVal = NamedFunction()
    myVal
  }
}

@chiselName
class NonNamedFunction extends NamedModuleTester {
  val test = NonNamedHelper.NamedFunction()
}

/** Ensure broken links in the chain are simply dropped
  */
@chiselName
class PartialNamedModule extends NamedModuleTester {
  // Create an inner function that is the extent of the implicit naming
  def innerNamedFunction(): UInt = {
    def innerUnnamedFunction(): UInt = {
      @chiselName
      def disconnectedNamedFunction(): UInt = {
        val a = expectName(1.U + 2.U, "test_a")
        val b = expectName(a + 2.U, "test_b")
        b
      }
      disconnectedNamedFunction()
    }
    innerUnnamedFunction() + 1.U
  }

  val test = innerNamedFunction()
}


/** A simple test that checks the recursive function val naming annotation both compiles and
  * generates the expected names.
  */
class NamingAnnotationSpec extends ChiselPropSpec {
  property("NamedModule should have function hierarchical names") {
    // TODO: clean up test style
    var module: NamedModule = null
    elaborate { module = new NamedModule; module }
    assert(module.getNameFailures() == Nil)
  }

  property("NameCollisionModule should disambiguate collisions") {
    // TODO: clean up test style
    var module: NameCollisionModule = null
    elaborate { module = new NameCollisionModule; module }
    assert(module.getNameFailures() == Nil)
  }

  property("PartialNamedModule should have partial names") {
    // TODO: clean up test style
    var module: PartialNamedModule = null
    elaborate { module = new PartialNamedModule; module }
    assert(module.getNameFailures() == Nil)
  }

  property("NonNamedModule should elaborate") {
    elaborate { new NonNamedModule }
  }

  property("NonNamedFunction should elaborate") {
    elaborate { new NonNamedFunction }
  }
}
