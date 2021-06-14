// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester



//object Scratchpad {
//  def f(a: AddOne): Record
//  def g(a: AddOne): AddOneIO
//  def g(a: AddOne): AddOneInterface
//  magnolia?
//
//}
//@template[Blah]
class AddOne extends MultiIOModule {
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))

  val r = RegNext(in + 1.asUInt)
  out := r
}


class AddOneTester extends BasicTester {
  val plusModuleTemplate = Template(new AddOne)
  val i0 = Instance(plusModuleTemplate) { _.r }
  val i1 = Instance(plusModuleTemplate)
  i0(_.in) := 42.U
  i1(_.in) := i0(_.out)
  assert(i1(_.out).asInstanceOf[UInt] === 44.U)
  stop()
}

class TemplateSpec extends ChiselFlatSpec with Utils {
  "Template/Instance" should "work" in {
    println((new ChiselStage).emitChirrtl(gen = new AddOneTester))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "work with aspects" in {
    import chisel3.aop.injecting._
    import chisel3.aop._
    val plus2aspect = InjectingAspect(
      {dut: AddOneTester => Select.collectDeep(dut, InstanceContext(Seq(dut))) {
        case (t: AddOne, context) =>
          println(context.toTarget)
          println(s"HERE: ${t.in.absoluteTarget(context)}")
          t
      }},
      {dut: AddOne =>
        println("HERE")
        dut.out := dut.in + 1.asUInt
      }
    )
    println((new ChiselStage).emitChirrtl(gen = new AddOneTester))
    assertTesterPasses( new AddOneTester, Nil, Seq(plus2aspect))
  }
}
