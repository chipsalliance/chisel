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
  * ./mill 'chisel[].test.testOnly' fooTest -Dfoo=bar
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
  private val _configMap: DynamicVariable[Option[ConfigMap]] = new DynamicVariable[Option[ConfigMap]](None)
  abstract override def withFixture(test: NoArgTest) = {
    _configMap.withValue(Some(test.configMap)) {
      super.withFixture(test)
    }
  }

  /** Return the config map which contains all command line options passed to Scalatest.
    *
    * This is only valid during a test.  It will be `None` if used outside a
    * test.
    *
    * @throws RuntimeException if called outside a Scalatest test
    */
  def configMap: ConfigMap = _configMap.value.getOrElse {
    throw new RuntimeException("configMap may only be accessed inside a Scalatest test")
  }

}
