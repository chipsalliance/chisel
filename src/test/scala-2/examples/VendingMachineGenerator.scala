// SPDX-License-Identifier: Apache-2.0

package examples

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{log2Ceil, Counter}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import VendingMachineUtils._

class VendingMachineIO(val legalCoins: Seq[Coin]) extends Bundle {
  require(legalCoins.size >= 1, "The vending machine must accept at least 1 coin!")
  // Order of coins by value
  val coins: Seq[Coin] = legalCoins.sortBy(_.value)
  // Map of coin names to their relative position in value (ie. index in inputs)
  val indexMap: Map[String, Int] = coins.map(_.name).zipWithIndex.toMap

  require(
    coins.map(_.value % coins.head.value == 0).reduce(_ && _),
    "All coins must be a multiple of the lowest value coin!"
  )

  val inputs = Input(Vec(legalCoins.size, Bool()))
  val dispense = Output(Bool())

  def apply(coin: String): Unit = {
    val idx = indexMap(coin)
    inputs(idx) := true.B
  }
}

// Superclass for parameterized vending machines
abstract class ParameterizedVendingMachine(legalCoins: Seq[Coin], val sodaCost: Int) extends Module {
  val io = IO(new VendingMachineIO(legalCoins))
  // Enforce one hot
  if (io.inputs.size > 1) {
    for (input <- io.inputs) {
      when(input) {
        assert(io.inputs.filterNot(_ == input).map(!_).reduce(_ && _), "Only 1 coin can be input in a given cycle!")
      }
    }
  }
}

class VendingMachineGenerator(legalCoins: Seq[Coin], sodaCost: Int)
    extends ParameterizedVendingMachine(legalCoins, sodaCost) {
  require(sodaCost > 0, "Sodas must actually cost something!")

  // All coin values are normalized to a multiple of the minimum coin value
  val minCoin = io.coins.head.value
  val maxCoin = io.coins.last.value
  val maxValue = (sodaCost + maxCoin - minCoin) / minCoin // normalize to minimum value

  val width = log2Ceil(maxValue + 1).W
  val value = RegInit(0.asUInt(width))
  val incValue = WireDefault(0.asUInt(width))
  val doDispense = value >= (sodaCost / minCoin).U

  when(doDispense) {
    value := 0.U // No change given
  }.otherwise {
    value := value + incValue
  }

  for ((coin, index) <- io.coins.zipWithIndex) {
    when(io.inputs(index)) { incValue := (coin.value / minCoin).U }
  }
  io.dispense := doDispense
}

class ParameterizedVendingMachineTester(mod: => ParameterizedVendingMachine, testLength: Int) extends Module {
  require(testLength > 0, "Test length must be positive!")

  // Construct the module
  val dut = Module(mod)
  val coins = dut.io.coins

  // Inputs and expected results
  // Do random testing
  private val _rand = scala.util.Random
  val inputs: Seq[Option[Coin]] = Seq.fill(testLength)(coins.lift(_rand.nextInt(coins.size + 1))) :+ None

  val expected: Seq[Boolean] = getExpectedResults(inputs, dut.sodaCost)

  val inputVec: Vec[UInt] = VecInit(inputs.map {
    case Some(coin) => (1 << dut.io.indexMap(coin.name)).asUInt(coins.size.W)
    case None       => 0.asUInt(coins.size.W)
  })
  val expectedVec: Vec[Bool] = VecInit(expected.map(_.B))

  val (idx, done) = Counter(true.B, testLength + 1)
  when(done) { stop() }

  dut.io.inputs := inputVec(idx).asBools

  // There must not be an out-of-bounds index.  If we OOB, then the check below will be wrong.
  assert(idx < expectedVec.size.U, "There was an out-of-bounds index")

  // Check that we dispense a soda as expected.
  assert(dut.io.dispense === expectedVec(idx))
}

class VendingMachineGeneratorSpec extends AnyFlatSpec with Matchers with ChiselSim {
  behavior.of("The vending machine generator")

  it should "generate a vending machine that accepts only nickels and dimes and costs $0.20" in {
    val coins = Seq(Nickel, Dime)
    simulate {
      new ParameterizedVendingMachineTester(new VendingMachineGenerator(coins, 20), 100)
    }(RunUntilFinished(100 + 3))
  }
  it should "generate a vending machine that only accepts one kind of coin" in {
    val coins = Seq(Nickel)
    simulate {
      new ParameterizedVendingMachineTester(new VendingMachineGenerator(coins, 30), 100)
    }(RunUntilFinished(100 + 2))
  }
  it should "generate a more realistic vending machine that costs $1.50" in {
    val coins = Seq(Penny, Nickel, Dime, Quarter)
    simulate {
      new ParameterizedVendingMachineTester(new VendingMachineGenerator(coins, 150), 100)
    }(RunUntilFinished(150 + 2))
  }
  it should "generate a Harry Potter themed vending machine" in {
    val coins = Seq(Knut, Sickle) // Galleons are worth too much
    simulate {
      new ParameterizedVendingMachineTester(new VendingMachineGenerator(coins, Galleon.value), 100)
    }(RunUntilFinished(100 + 2))
  }
}
