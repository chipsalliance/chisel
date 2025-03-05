// SPDX-License-Identifier: Apache-2.0

package chisel3.testing.scalatest

import org.scalatest.{ConfigMap, TestSuite, TestSuiteMixin}
import scala.util.DynamicVariable

/** A Scalatest test suite mix-in that provides access to command line options.
  *
  * This expose any keys/values passed using the `-D<key>=<value` command line
  * option to the test via a `Map`.
  *
  * For example, you can invoke Scalatest passing the `foo=bar` option like so:
  * {{{
  * ./mill 'chisel[2.13.16].test.testOnly' 'fooTest' -Dfoo=bar
  * }}}
  *
  * Inside your test, if the `configMap` member function is accessed this will
  * return:
  * {{{
  * Map(foo -> bar)
  * }}}
  *
  * This is intended to be a building block of more complicated tests that want
  * to customize their execution via the command line.  It is advisable to use
  * this sparingly as tests are generally not intended to change when you run
  * them.
  */
trait HasConfigMap extends TestSuiteMixin { self: TestSuite =>

  // Implement this via a `DynamicVariable` pattern that will be set via the
  // `withFixture` method.  The `super.withFixture` function must be called to
  // make this mix-in "stackable" with other mix-ins.
  private val _configMap: DynamicVariable[ConfigMap] = new DynamicVariable[ConfigMap](ConfigMap.empty)
  abstract override def withFixture(test: NoArgTest) = {
    _configMap.withValue(test.configMap) {
      super.withFixture(test)
    }
  }

  /** Return command line options passed to the test as a `Map`. */
  def configMap: Map[String, Any] = _configMap.value

}
