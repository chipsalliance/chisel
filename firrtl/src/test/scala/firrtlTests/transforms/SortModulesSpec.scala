// SPDX-License-Identifier: Apache-2.0

package firrtlTests.transforms

import firrtl.{ir, CircuitState, Parser}
import firrtl.transforms.SortModules
import firrtl.traversals.Foreachers._

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.collection.mutable

class SortModulesSpec extends AnyFlatSpec with Matchers {

  private def collectModules(names: mutable.ArrayBuffer[String], module: ir.DefModule): Unit = names += module.name

  behavior.of("SortModules")

  it should "enforce define before use of modules" in {

    val input =
      """|circuit Foo:
         |  module Foo:
         |    inst bar of Bar
         |  module Bar:
         |    inst baz of Baz
         |  extmodule Baz:
         |    input a: UInt<1>
         |""".stripMargin

    val state = CircuitState(Parser.parse(input), Seq.empty)
    val moduleNames = mutable.ArrayBuffer.empty[String]

    (new SortModules)
      .execute(state)
      .circuit
      .foreach(collectModules(moduleNames, _: ir.DefModule))

    (moduleNames should contain).inOrderOnly("Baz", "Bar", "Foo")
  }

}
