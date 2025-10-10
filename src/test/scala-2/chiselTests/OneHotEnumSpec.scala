// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.{is, switch, Counter}
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import chisel3.util.Decoupled
import chisel3.util.isPow2

object OneHotEnumExample extends OneHotEnum {
  val A, B, C, D, E = Value
}

object OtherOneHotEnum extends OneHotEnum {
  val W, X, Y, Z = Value
}

class OneHotEnumSafeCast extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(OneHotEnumExample.getWidth.W))
    val out = Output(OneHotEnumExample())
    val valid = Output(Bool())
  })

  val (enumVal, valid) = OneHotEnumExample.safe(io.in)
  io.out :#= enumVal
  io.valid := valid
}

class OneHotEnumSafeCastTester extends Module {
  for ((enumVal, i) <- OneHotEnumExample.all.zipWithIndex) {
    val lit = (1 << i).U(OneHotEnumExample.getWidth.W)
    val mod = Module(new OneHotEnumSafeCast)
    mod.io.in :#= lit
    assert(mod.io.out === enumVal)
    assert(mod.io.valid === true.B)
  }

  val invalid_values =
    (1 until (1 << OneHotEnumExample.getWidth)).filter(!isPow2(_)).map(_.U)

  for (invalid_val <- invalid_values) {
    val mod = Module(new SafeCastFromNonLit)
    mod.io.in := invalid_val

    assert(mod.io.valid === false.B)
  }

  stop()
}

class OneHotEnumFSM extends Module {

  object State extends OneHotEnum {
    val Idle, One, Two, Three = Value
  }

  object State2 extends ChiselEnum {
    val Idle, One, Two, Three = Value
  }

  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
    val out2 = Output(UInt(8.W))
    val other_out = Output(UInt(5.W))
    val state = Output(UInt(4.W))
  })

  import State._

  val state = RegInit(Idle)
  val state2 = RegInit(State2.Idle)

  assert(state.getWidth == State.all.length)

  assert(state.isValid)

  assert((1.U << state2.asUInt) === state.asUInt)

  io.state :#= state.asUInt
  io.out2 := DontCare

  printf(cf"state is $state (${state.asUInt}%b)\n")

  when(state2 === State2.Idle) {
    assert(state.is(State.Idle))
    assert(state.next.is(State.One))
    assert(state.next === State.One)
    assert(state2.next === State2.One)
    io.out2 := 0.U
    state2 :#= State2.One
  }

  when(state2 === State2.One) {
    io.out2 := 1.U
    state2 :#= State2.Two

    assert(state.next.is(State.Two))
    assert(state2.next === State2.Two)
  }

  when(state2.isOneOf(State2.Two)) {
    io.out2 := 2.U
    state2 :#= State2.Three
    assert(state.next.is(State.Three))
    assert(state2.next === State2.Three)
  }

  when(state2.isOneOf(State2.Three)) {
    io.out2 :#= io.in
    state2 :#= State2.Idle
    assert(state.next.is(State.Idle))
    assert(state2.next === State2.Idle)
  }

  assert(state.isOneOf(State.all))

  for (s <- State.all) {
    assert(state.is(s) === (state === s))
    assert(state.is(s) === state.isOneOf(s))
  }

  assert(State.Idle.next === State.One)
  assert(State.One.next === State.Two)
  assert(State.Two.next === State.Three)
  assert(State.Three.next === State.Idle)

  state :#= state.select(
    Idle -> One,
    One -> Two,
    Two -> Three,
    Three -> Idle
  )

  io.out :#= state.select(
    Idle -> 0x00.U,
    One -> 0x01.U,
    Two -> 0x02.U,
    Three -> io.in
  )

  assert(io.out === io.out2)

  io.other_out :#= state.select(
    Idle -> 0x10.U,
    One -> 0x11.U,
    Two -> 0x12.U,
    Three -> 0x13.U
  )
}

class OneHotEnumFSMTester extends Module {
  val mod = Module(new OneHotEnumFSM)
  val counter = Counter(9)

  val expectedStateIndex = counter.value % mod.State.all.length.U

  mod.io.in := counter.value

  assert(mod.io.state === (1.U << expectedStateIndex))
  assert(mod.io.other_out === (0x10.U + expectedStateIndex))

  switch(expectedStateIndex) {
    is(0.U) {
      assert(mod.io.out2 === 0.U)
    }
    is(1.U) {
      assert(mod.io.out2 === 1.U)
    }
    is(2.U) {
      assert(mod.io.out2 === 2.U)
    }
    is(3.U) {
      assert(mod.io.out2 === counter.value)
    }
  }

  when(counter.inc()) {
    stop()
  }
}

class OneHotEnumSequenceDetector extends Module {

  object State extends OneHotEnum {
    val Idle, Saw1, Saw10, Saw101 = Value
  }

  val io = IO(new Bundle {
    val in = Input(Bool())
    val detect = Output(Bool())
    val state = Output(UInt(State.all.length.W))
  })

  import State._

  val state = RegInit(Idle)

  assert(state.getWidth == State.all.length)
  assert(state.isValid)

  assert(state.isOneOf(State.all))

  for (s <- State.all) {
    assert(state.is(s) === (state === s))
    assert(state.is(s) === state.isOneOf(s))
  }

  assert(State.Idle.next === State.Saw1)
  assert(State.Saw1.next === State.Saw10)
  assert(State.Saw10.next === State.Saw101)
  assert(State.Saw101.next === State.Idle)

  io.detect := state.is(Saw101)

  state :#= state.select(
    Idle -> Mux(io.in, Saw1, Idle),
    Saw1 -> Mux(io.in, Saw1, Saw10),
    Saw10 -> Mux(io.in, Saw101, Idle),
    Saw101 -> Mux(io.in, Saw1, Saw10)
  )

  io.state :#= state.asUInt

  printf(cf"state is $state (${state.asUInt}%b), in is ${io.in}, detect is ${io.detect}\n")
}

class OneHotEnumSequenceDetectorTester extends Module {
  val mod = Module(new OneHotEnumSequenceDetector)

  import mod.State._

  val symbols = VecInit(Seq(1, 0, 1, 0, 1, 1, 0, 1, 0, 0).map(_.B))
  val expectedHits = VecInit(Seq(0, 0, 0, 1, 0, 1, 0, 0, 1, 0).map(_.B))

  val expectedStates = VecInit(
    Seq(Idle, Saw1, Saw10, Saw101, Saw10, Saw101, Saw1, Saw10, Saw101, Saw10, Idle)
  )

  val counter = Counter(symbols.length)

  mod.io.in := symbols(counter.value)

  assert(
    mod.io.detect === expectedHits(counter.value),
    cf"mismatch at ${counter.value}: got ${mod.io.detect}, expected ${expectedHits(counter.value)}"
  )
  assert(mod.io.state === expectedStates(counter.value).asUInt)

  when(counter.inc()) {
    stop()
  }
}

object VendingMachineState extends OneHotEnum {
  val Idle, Have5, Have10, Vend = Value
}

class OneHotEnumVendingMachine extends Module {

  val io = IO(new Bundle {
    val coin = Flipped(Decoupled(UInt(2.W))) // 0: none, 1: nickel, 2: dime
    val vend = Output(Bool())
    val change = Output(Bool())
    val state = Output(VendingMachineState())
  })

  import VendingMachineState._

  val state = RegInit(Idle)

  assert(state.getWidth == VendingMachineState.all.length)
  assert(state.isValid)
  assert(state.isOneOf(VendingMachineState.all))
  for (s <- VendingMachineState.all) {
    assert(state.is(s) === (state === s))
    assert(state.is(s) === state.isOneOf(s))
  }

  val change = RegInit(false.B)

  val insertFive = io.coin.valid && io.coin.bits === 1.U
  val insertTen = io.coin.valid && io.coin.bits === 2.U

  io.coin.ready := state.isOneOf(Idle, Have5, Have10)

  val nextState = state.select(
    Idle -> Mux(insertTen, Have10, Mux(insertFive, Have5, Idle)),
    Have5 -> Mux(insertTen, Vend, Mux(insertFive, Have10, Have5)),
    Have10 -> Mux(insertFive || insertTen, Vend, Have10),
    Vend -> Idle
  )

  state :#= nextState
  change := state.is(Have10) && insertTen

  io.state :#= state
  io.vend := state.is(Vend)
  io.change := change
}

class OneHotEnumVendingMachineTester extends Module {
  val mod = Module(new OneHotEnumVendingMachine)

  import VendingMachineState._

  val coins = VecInit(Seq(0, 1, 0, 2, 0, 0, 2, 1, 0, 2, 2, 0).map(_.U(2.W)))
  val expectedStates = VecInit(
    Seq(Idle, Idle, Have5, Have5, Vend, Idle, Idle, Have10, Vend, Idle, Have10, Vend)
  )
  val expectedVend = VecInit(Seq(0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1).map(_.B))
  val expectedChange = VecInit(Seq(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1).map(_.B))

  val counter = Counter(coins.length)

  mod.io.coin.bits :#= coins(counter.value)
  mod.io.coin.valid := true.B

  assert(mod.io.state.isOneOf(Idle, Have5, Have10) === mod.io.coin.ready)

  assert(mod.io.state === expectedStates(counter.value))
  assert(mod.io.vend === expectedVend(counter.value))
  assert(mod.io.change === expectedChange(counter.value))

  when(counter.inc()) {
    stop()
  }
}

class OneHotEnumSpec extends AnyFlatSpec with Matchers with LogUtils with ChiselSim with FileCheck {
  behavior of "OneHotEnum"

  it should "maintain Scala-level type-safety" in {
    def foo(e: OneHotEnumExample.Type): Unit = {}

    "foo(OneHotEnumExample.A); foo(OneHotEnumExample.A.next); foo(OneHotEnumExample.E.next)" should compile
    "foo(OtherOneHotEnum.otherEnum)" shouldNot compile
    "foo(EnumExample.otherEnum)" shouldNot compile
    "foo(OtherEnum.otherEnum)" shouldNot compile
  }

  it should "prevent enums from being declared without names" in {
    "object UnnamedEnum1 extends OneHotEnum { Value }" shouldNot compile
  }

  it should "prevent enums from being declared with custom values" in {
    "object UnnamedEnum2 extends OneHotEnum { A = Value(1.U) }" shouldNot compile
  }

  it should "safely cast non-literal UInts to enums correctly and detect illegal casts" in {
    simulate(new OneHotEnumSafeCastTester)(RunUntilFinished(3))
  }

  "OneHotEnumFSM" should "work" in {
    simulate(new OneHotEnumFSMTester)(RunUntilFinished(10))
  }

  "OneHotEnumSequenceDetector" should "work" in {
    simulate(new OneHotEnumSequenceDetectorTester)(RunUntilFinished(12))
  }

  "OneHotEnumVendingMachine" should "work" in {
    simulate(new OneHotEnumVendingMachineTester)(RunUntilFinished(16))
  }

}
