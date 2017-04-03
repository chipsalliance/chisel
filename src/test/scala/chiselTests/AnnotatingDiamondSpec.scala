// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.ChiselAnnotation
import chisel3.internal.InstanceId
import chisel3.testers.BasicTester
import firrtl.{CircuitForm, CircuitState, LowForm, Transform}
import firrtl.annotations.{Annotation, ModuleName, Named}
import org.scalatest._

//scalastyle:off magic.number
/**
  * This and the Identity transform class are a highly schematic implementation of a
  * library implementation of   (i.e. code outside of firrtl itself)
  */
object IdentityAnnotation {
  def apply(target: Named, value: String): Annotation = Annotation(target, classOf[IdentityTransform], value)

  def unapply(a: Annotation): Option[(Named, String)] = a match {
    case Annotation(named, t, value) if t == classOf[IdentityTransform] => Some((named, value))
    case _ => None
  }
}

class IdentityTransform extends Transform {
  override def inputForm: CircuitForm = LowForm

  override def outputForm: CircuitForm = LowForm

  override def execute(state: CircuitState): CircuitState = {
    getMyAnnotations(state) match {
      case Nil => state
      case myAnnotations =>
        state
    }
  }
}

trait IdentityAnnotator {
  self: Module =>
  def identify(component: InstanceId, value: String): Unit = {
    annotate(ChiselAnnotation(component, classOf[IdentityTransform], value))
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
class ModC(widthC: Int) extends Module with IdentityAnnotator {
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
class ModA(annoParam: Int) extends Module with IdentityAnnotator {
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

class ModB(widthB: Int) extends Module with IdentityAnnotator{
  val io = IO(new Bundle {
    val in = Input(UInt(widthB.W))
    val out = Output(UInt(widthB.W))
  })
  val modC = Module(new ModC(widthB))
  modC.io.in := io.in
  io.out := modC.io.out

  identify(io.in, s"modB.io.in annotated from inside modB")
}

class TopOfDiamond extends Module with IdentityAnnotator {
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
  def findAnno(as: Seq[Annotation], name: String): Option[Annotation] = {
    as.find { a => a.targetString == name }
  }

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
          val annos = circuit.annotations
          annos.length should be (10)

          annos.count {
            case Annotation(ModuleName(name, _), _, annoValue) => name == "ModC" && annoValue == "ModC(16)"
            case _ => false
          } should be (1)

          annos.count {
            case Annotation(ModuleName(name, _), _, annoValue) => name == "ModC_1" && annoValue == "ModC(32)"
            case _ => false
          } should be (1)
        case _ =>
          assert(false)
      }
    }
  }
}