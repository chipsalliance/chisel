// See LICENSE for license details.

package examples

import chiselTests.ChiselFlatSpec
import chisel3.testers.BasicTester
import chisel3._
import chisel3.util._
import VendingMachineUtils._
import chisel3.stage.{DefaultGeneratorPackage, DefaultGeneratorPackageCreator}
import firrtl.AnnotationSeq

class VendingMachineTester(machine: Template[ParameterizedVendingMachine],
                           testLength: Int) extends RawModule {
  require(testLength > 0, "Test length must be positive!")

  // Construct the module
  val dut = machine.instantiate()
  val coins = dut(_.io.coins)

  // Inputs and expected results
  // Do random testing
  private val _rand = scala.util.Random
  val inputs: Seq[Option[Coin]] = Seq.fill(testLength)(coins.lift(_rand.nextInt(coins.size + 1)))
  val expected: Seq[Boolean] = getExpectedResults(inputs, dut(_.sodaCost))

  val inputVec: Vec[UInt] = VecInit(inputs map {
    case Some(coin) => (1 << dut(_.io.indexMap(coin.name))).asUInt(coins.size.W)
    case None => 0.asUInt(coins.size.W)
  })
  val expectedVec: Vec[Bool] = VecInit(expected map (_.B))

  val (idx, done) = Counter(true.B, testLength + 1)
  when (done) { stop(); stop() } // Two stops for Verilator

  dut(_.io.inputs) := inputVec(idx).asBools
  assert(dut(_.io.dispense) === expectedVec(idx))
}

class TesterApplier() extends RawModule {

}

class VendingMachine2GeneratorSpec extends ChiselFlatSpec {
  behavior of "The vending machine generator"

  it should "generate a vending machine that accepts only nickels and dimes and costs $0.20" in {
    object VendingMachinePackage extends DefaultGeneratorPackageCreator[VendingMachineGenerator] {
      override val packge = "PVM"
      override val version = "snapshot"

      val coins = Seq(Nickel, Dime)
      def gen() = new VendingMachineGenerator(coins, 20)

      override def createPackage(top: VendingMachineGenerator, annos: AnnotationSeq): DefaultGeneratorPackage[VendingMachineGenerator] = {
        DefaultGeneratorPackage[VendingMachineGenerator](top, annos, VendingMachinePackage)
      }
    }

    val coins = Seq(Nickel, Dime)
      new ParameterizedVendingMachineTester(new VendingMachineGenerator(coins, 20), 100)
  }
}
