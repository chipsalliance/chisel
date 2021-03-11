// SPDX-License-Identifier: Apache-2.0

package firrtlTests
package transforms

import firrtl._
import firrtl.testutils._
import firrtl.stage.TransformManager
import firrtl.options.Dependency
import firrtl.transforms.CSESubAccesses

class CSESubAccessesSpec extends FirrtlFlatSpec {
  def compile(input: String): String = {
    val manager = new TransformManager(Dependency[CSESubAccesses] :: Nil)
    val result = manager.execute(CircuitState(parse(input), Nil))
    result.circuit.serialize
  }
  def circuit(body: String): String = {
    """|circuit Test :
       |  module Test :
       |""".stripMargin + body.stripMargin.split("\n").mkString("    ", "\n    ", "\n")
  }

  behavior.of("CSESubAccesses")

  it should "hoist a single RHS subaccess" in {
    val input = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |out <= in[idx]"""
    )
    val expected = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |node _in_idx = in[idx]
          |out <= _in_idx"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "be idempotent" in {
    val input = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |out <= in[idx]"""
    )
    val expected = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |node _in_idx = in[idx]
          |out <= _in_idx"""
    )
    val first = compile(input)
    val second = compile(first)
    first should be(second)
    first should be(parse(expected).serialize)
  }

  it should "hoist a redundant RHS subaccess" in {
    val input = circuit(
      s"""|input in : { foo : UInt<8>, bar : UInt<8> }[4]
          |input idx : UInt<2>
          |output out : { foo : UInt<8>, bar : UInt<8> }
          |out.foo <= in[idx].foo
          |out.bar <= in[idx].bar"""
    )
    val expected = circuit(
      s"""|input in : { foo : UInt<8>, bar : UInt<8> }[4]
          |input idx : UInt<2>
          |output out : { foo : UInt<8>, bar : UInt<8> }
          |node _in_idx = in[idx]
          |out.foo <= _in_idx.foo
          |out.bar <= _in_idx.bar"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "correctly place hosited subaccess after last declaration it depends on" in {
    val input = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |out is invalid
          |when UInt(1) :
          |  node nidx = not(idx)
          |  out <= in[nidx]
          |"""
    )
    val expected = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |out is invalid
          |when UInt(1) :
          |  node nidx = not(idx)
          |  node _in_nidx = in[nidx]
          |  out <= _in_nidx
          |"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "support complex expressions" in {
    val input = circuit(
      s"""|input clock : Clock
          |input in : UInt<8>[4]
          |input idx : UInt<2>
          |input sel : UInt<1>
          |output out : UInt<8>
          |reg r : UInt<2>, clock
          |out <= in[mux(sel, r, idx)]
          |r <= not(idx)"""
    )
    val expected = circuit(
      s"""|input clock : Clock
          |input in : UInt<8>[4]
          |input idx : UInt<2>
          |input sel : UInt<1>
          |output out : UInt<8>
          |reg r : UInt<2>, clock
          |node _in_mux = in[mux(sel, r, idx)]
          |out <= _in_mux
          |r <= not(idx)"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "support nested subaccesses" in {
    val input = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>[4]
          |input jdx : UInt<2>
          |output out : UInt<8>
          |out <= in[idx[jdx]]"""
    )
    val expected = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>[4]
          |input jdx : UInt<2>
          |output out : UInt<8>
          |node _idx_jdx = idx[jdx]
          |node _in_idx = in[_idx_jdx]
          |out <= _in_idx"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "avoid name collisions" in {
    val input = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |out <= in[idx]
          |node _in_idx = not(idx)"""
    )
    val expected = circuit(
      s"""|input in : UInt<8>[4]
          |input idx : UInt<2>
          |output out : UInt<8>
          |node _in_idx_0 = in[idx]
          |out <= _in_idx_0
          |node _in_idx = not(idx)"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "have no effect on LHS SubAccesses" in {
    val input = circuit(
      s"""|input in : UInt<8>
          |input idx : UInt<2>
          |output out : UInt<8>[4]
          |out[idx] <= in"""
    )
    val expected = circuit(
      s"""|input in : UInt<8>
          |input idx : UInt<2>
          |output out : UInt<8>[4]
          |out[idx] <= in"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "ignore flipped LHS SubAccesses" in {
    val input = circuit(
      s"""|input in : { foo : UInt<8> }
          |input idx : UInt<1>
          |input out : { flip foo : UInt<8> }[2]
          |out[0].foo <= UInt(0)
          |out[1].foo <= UInt(0)
          |out[idx].foo <= in.foo"""
    )
    val expected = circuit(
      s"""|input in : { foo : UInt<8> }
          |input idx : UInt<1>
          |input out : { flip foo : UInt<8> }[2]
          |out[0].foo <= UInt(0)
          |out[1].foo <= UInt(0)
          |out[idx].foo <= in.foo"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "ignore SubAccesses of bidirectional aggregates" in {
    val input = circuit(
      s"""|input in : { flip foo : UInt<8>, bar : UInt<8> }
          |input idx : UInt<2>
          |output out : { flip foo : UInt<8>, bar : UInt<8> }[4]
          |out[idx] <= in"""
    )
    val expected = circuit(
      s"""|input in : { flip foo : UInt<8>, bar : UInt<8> }
          |input idx : UInt<2>
          |output out : { flip foo : UInt<8>, bar : UInt<8> }[4]
          |out[idx] <= in"""
    )
    compile(input) should be(parse(expected).serialize)
  }

}
