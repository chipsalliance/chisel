// SPDX-License-Identifier: Apache-2.0

package firrtlTests
package passes

import firrtl._
import firrtl.testutils._
import firrtl.stage.TransformManager
import firrtl.options.Dependency
import firrtl.passes._

class RemoveAccessesSpec extends FirrtlFlatSpec {
  def compile(input: String): String = {
    val manager = new TransformManager(Dependency(RemoveAccesses) :: Nil)
    val result = manager.execute(CircuitState(parse(input), Nil))
    val checks = List(
      CheckHighForm,
      CheckTypes,
      CheckFlows
    )
    for (check <- checks) { check.run(result.circuit) }
    result.circuit.serialize
  }
  def circuit(body: String): String = {
    """|circuit Test :
       |  module Test :
       |""".stripMargin + body.stripMargin.split("\n").mkString("    ", "\n    ", "\n")
  }

  behavior.of("RemoveAccesses")

  it should "handle a simple RHS subaccess" in {
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
          |wire _in_idx : UInt<8>
          |_in_idx is invalid
          |when eq(UInt<1>("h0"), idx) :
          |  _in_idx <= in[0]
          |when eq(UInt<1>("h1"), idx) :
          |  _in_idx <= in[1]
          |when eq(UInt<2>("h2"), idx) :
          |  _in_idx <= in[2]
          |when eq(UInt<2>("h3"), idx) :
          |  _in_idx <= in[3]
          |out <= _in_idx"""
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
          |wire _in_mux : UInt<8>
          |_in_mux is invalid
          |when eq(UInt<1>("h0"), mux(sel, r, idx)) :
          |  _in_mux <= in[0]
          |when eq(UInt<1>("h1"), mux(sel, r, idx)) :
          |  _in_mux <= in[1]
          |when eq(UInt<2>("h2"), mux(sel, r, idx)) :
          |  _in_mux <= in[2]
          |when eq(UInt<2>("h3"), mux(sel, r, idx)) :
          |  _in_mux <= in[3]
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
          |wire _idx_jdx : UInt<2>
          |_idx_jdx is invalid
          |when eq(UInt<1>("h0"), jdx) :
          |  _idx_jdx <= idx[0]
          |when eq(UInt<1>("h1"), jdx) :
          |  _idx_jdx <= idx[1]
          |when eq(UInt<2>("h2"), jdx) :
          |  _idx_jdx <= idx[2]
          |when eq(UInt<2>("h3"), jdx) :
          |  _idx_jdx <= idx[3]
          |wire _in_idx_jdx : UInt<8>
          |_in_idx_jdx is invalid
          |when eq(UInt<1>("h0"), _idx_jdx) :
          |  _in_idx_jdx <= in[0]
          |when eq(UInt<1>("h1"), _idx_jdx) :
          |  _in_idx_jdx <= in[1]
          |when eq(UInt<2>("h2"), _idx_jdx) :
          |  _in_idx_jdx <= in[2]
          |when eq(UInt<2>("h3"), _idx_jdx) :
          |  _in_idx_jdx <= in[3]
          |out <= _in_idx_jdx"""
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
          |wire _in_idx_0 : UInt<8>
          |_in_idx_0 is invalid
          |when eq(UInt<1>("h0"), idx) :
          |  _in_idx_0 <= in[0]
          |when eq(UInt<1>("h1"), idx) :
          |  _in_idx_0 <= in[1]
          |when eq(UInt<2>("h2"), idx) :
          |  _in_idx_0 <= in[2]
          |when eq(UInt<2>("h3"), idx) :
          |  _in_idx_0 <= in[3]
          |out <= _in_idx_0
          |node _in_idx = not(idx)"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "handle a simple LHS subaccess" in {
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
          |wire _out_idx : UInt<8>
          |when eq(UInt<1>("h0"), idx) :
          |  out[0] <= _out_idx
          |when eq(UInt<1>("h1"), idx) :
          |  out[1] <= _out_idx
          |when eq(UInt<2>("h2"), idx) :
          |  out[2] <= _out_idx
          |when eq(UInt<2>("h3"), idx) :
          |  out[3] <= _out_idx
          |_out_idx <= in"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "linearly expand RHS subaccesses of aggregate-typed vecs" in {
    val input = circuit(
      s"""|input in : { foo : UInt<8>, bar : UInt<8> }[4]
          |input idx : UInt<2>
          |output out : { foo : UInt<8>, bar : UInt<8> }
          |out.foo <= in[idx].foo
          |out.bar <= in[idx].bar"""
    )
    val expected = circuit(
      s"""|input in : { foo : UInt<8>, bar : UInt<8>}[4]
          |input idx : UInt<2>
          |output out : { foo : UInt<8>, bar : UInt<8>}
          |wire _in_idx_foo : UInt<8>
          |_in_idx_foo is invalid
          |when eq(UInt<1>("h0"), idx) :
          |  _in_idx_foo <= in[0].foo
          |when eq(UInt<1>("h1"), idx) :
          |  _in_idx_foo <= in[1].foo
          |when eq(UInt<2>("h2"), idx) :
          |  _in_idx_foo <= in[2].foo
          |when eq(UInt<2>("h3"), idx) :
          |  _in_idx_foo <= in[3].foo
          |out.foo <= _in_idx_foo
          |wire _in_idx_bar : UInt<8>
          |_in_idx_bar is invalid
          |when eq(UInt<1>("h0"), idx) :
          |  _in_idx_bar <= in[0].bar
          |when eq(UInt<1>("h1"), idx) :
          |  _in_idx_bar <= in[1].bar
          |when eq(UInt<2>("h2"), idx) :
          |  _in_idx_bar <= in[2].bar
          |when eq(UInt<2>("h3"), idx) :
          |  _in_idx_bar <= in[3].bar
          |out.bar <= _in_idx_bar"""
    )
    compile(input) should be(parse(expected).serialize)
  }

  it should "linearly expand LHS subaccesses of aggregate-typed vecs" in {
    val input = circuit(
      s"""|input in : { foo : UInt<8>, bar : UInt<8> }
          |input idx : UInt<2>
          |output out : { foo : UInt<8>, bar : UInt<8> }[4]
          |out[idx].foo <= in.foo
          |out[idx].bar <= in.bar"""
    )
    val expected = circuit(
      s"""|input in : { foo : UInt<8>, bar : UInt<8> }
          |input idx : UInt<2>
          |output out : { foo : UInt<8>, bar : UInt<8> }[4]
          |wire _out_idx_foo : UInt<8>
          |when eq(UInt<1>("h0"), idx) :
          |  out[0].foo <= _out_idx_foo
          |when eq(UInt<1>("h1"), idx) :
          |  out[1].foo <= _out_idx_foo
          |when eq(UInt<2>("h2"), idx) :
          |  out[2].foo <= _out_idx_foo
          |when eq(UInt<2>("h3"), idx) :
          |  out[3].foo <= _out_idx_foo
          |_out_idx_foo <= in.foo
          |wire _out_idx_bar : UInt<8>
          |when eq(UInt<1>("h0"), idx) :
          |  out[0].bar <= _out_idx_bar
          |when eq(UInt<1>("h1"), idx) :
          |  out[1].bar <= _out_idx_bar
          |when eq(UInt<2>("h2"), idx) :
          |  out[2].bar <= _out_idx_bar
          |when eq(UInt<2>("h3"), idx) :
          |  out[3].bar <= _out_idx_bar
          |_out_idx_bar <= in.bar"""
    )
    compile(input) should be(parse(expected).serialize)
  }
}
