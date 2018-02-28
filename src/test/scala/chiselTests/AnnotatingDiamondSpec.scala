// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{annotate, ChiselAnnotation, RunFirrtlTransform}
import chisel3.internal.InstanceId
import chisel3.testers.BasicTester
import firrtl.{CircuitState, LowForm, Transform}
import firrtl.annotations.{
  Annotation,
  SingleTargetAnnotation,
  ModuleName,
  Named
}
import org.scalatest._

/** These annotations and the IdentityTransform class serve as an example of how to write a
  * Chisel/Firrtl library
  */
case class IdentityAnnotation(target: Named, value: String) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named) = this.copy(target = n)
}
/** ChiselAnnotation that corresponds to the above FIRRTL annotation */
case class IdentityChiselAnnotation(target: InstanceId, value: String)
    extends ChiselAnnotation with RunFirrtlTransform {
  def toFirrtl = IdentityAnnotation(target.toNamed, value)
  def transformClass = classOf[IdentityTransform]
}
object identify {
  def apply(component: InstanceId, value: String): Unit = {
    val anno = IdentityChiselAnnotation(component, value)
    annotate(anno)
  }
}

class IdentityTransform extends Transform {
  def inputForm = LowForm
  def outputForm = LowForm

  def execute(state: CircuitState): CircuitState = {
    val annosx = state.annotations.map {
      case IdentityAnnotation(t, value) => IdentityAnnotation(t, value + ":seen")
      case other => other
    }
    state.copy(annotations = annosx)
  }
}

/** A diamond circuit Top instantiates A and B and both A and B instantiate C
  * Illustrations of annotations of various components and modules in both
  * relative and absolute cases
  *
  * This is currently not much of a test, read the printout to see what annotations look like
  */
/**
  * This class has parameterizable widths, it will generate different hardware
  * @param widthC io width
  */
class ModC(widthC: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(widthC.W))
    val out = Output(UInt(widthC.W))
  })
  io.out := io.in

  identify(this, s"ModC($widthC)")

  identify(io.out, s"ModC(ignore param)")
}

/**
  * instantiates a C of a particular size, ModA does not generate different hardware
  * based on it's parameter
  * @param annoParam  parameter is only used in annotation not in circuit
  */
class ModA(annoParam: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt())
    val out = Output(UInt())
  })
  val modC = Module(new ModC(16))
  modC.io.in := io.in
  io.out := modC.io.out

  identify(this, s"ModA(ignore param)")

  identify(io.out, s"ModA.io.out($annoParam)")
  identify(io.out, s"ModA.io.out(ignore_param)")
}

class ModB(widthB: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(widthB.W))
    val out = Output(UInt(widthB.W))
  })
  val modC = Module(new ModC(widthB))
  modC.io.in := io.in
  io.out := modC.io.out

  identify(io.in, s"modB.io.in annotated from inside modB")
}

class TopOfDiamond extends Module {
  val io = IO(new Bundle {
    val in   = Input(UInt(32.W))
    val out  = Output(UInt(32.W))
  })
  val x = Reg(UInt(32.W))
  val y = Reg(UInt(32.W))

  val modA = Module(new ModA(64))
  val modB = Module(new ModB(32))

  x := io.in
  modA.io.in := x
  modB.io.in := x

  y := modA.io.out + modB.io.out
  io.out := y

  identify(this, s"TopOfDiamond\nWith\nSome new lines")

  identify(modB.io.in, s"modB.io.in annotated from outside modB")
}

class DiamondTester extends BasicTester {
  val dut = Module(new TopOfDiamond)

  stop()
}

class AnnotatingDiamondSpec extends FreeSpec with Matchers {

  """
    |Diamond is an example of a module that has two sub-modules A and B who both instantiate their
    |own instances of module C.  This highlights the difference between specific and general
    |annotation scopes
  """.stripMargin - {

    """
      |annotations are not resolved at after circuit elaboration,
      |that happens only after emit has been called on circuit""".stripMargin in {

      Driver.execute(Array("--target-dir", "test_run_dir"), () => new TopOfDiamond) match {
        case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
          val annos = circuit.annotations.map(_.toFirrtl)
          annos.count(_.isInstanceOf[IdentityAnnotation]) should be (10)

          annos.count {
            case IdentityAnnotation(ModuleName("ModC", _), "ModC(16)") => true
            case _ => false
          } should be (1)

          annos.count {
            case IdentityAnnotation(ModuleName("ModC_1", _), "ModC(32)") => true
            case _ => false
          } should be (1)
        case _ =>
          assert(false)
      }
    }
  }
}
