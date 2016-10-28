// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.internal.InstanceId
import chisel3.testers.BasicTester
import org.scalatest._

//scalastyle:off magic.number

/** A diamond circuit Top instantiates A and B and both A and B instantiate C
  * Illustrations of annotations of various components and modules in both
  * relative and absolute cases
  *
  * This is currently not much of a test, read the printout to see what annotations look like
  */
case class GeneralAnno(component: InstanceId, value: String) extends Annotation with Annotation.Scope.General
case class SpecificAnno(component: InstanceId, value: String) extends Annotation with Annotation.Scope.Specific

class ModC(param1: Int, param2: Int) extends Module {
  val io = new Bundle {
    val in = UInt(INPUT)
    val out = UInt(OUTPUT)
  }
  io.out := io.in

  annotate(SpecificAnno(this, s"ModC Absolute"))
  annotate(GeneralAnno(this, s"ModC Relative"))

  annotate(SpecificAnno(io.in, s"ModuleC($param1,$param2) width < $param1"))
  annotate(GeneralAnno(io.out, s"ModuleC($param1,$param2) width < $param2"))
}

class ModA(aParam1: Int, aParam2: Int) extends Module {
  val io = new Bundle {
    val in = UInt().flip()
    val out = UInt()
  }
  val modC = Module(new ModC(16, 32))
  modC.io.in := io.in
  io.out := modC.io.out

  annotate(SpecificAnno(this, s"ModA Absolute"))
  annotate(GeneralAnno(this, s"ModA Relative"))

  annotate(SpecificAnno(io.in, s"ModuleA($aParam1,$aParam2) width < $aParam1"))
  annotate(GeneralAnno(io.out, s"ModuleA($aParam1,$aParam2) width < $aParam2"))
}

class ModB(bParam1: Int, bParam2: Int) extends Module {
  val io = new Bundle {
    val in = UInt().flip()
    val out = UInt()
  }
  val modC = Module(new ModC(42, 77))
  modC.io.in := io.in
  io.out := modC.io.out
  annotate(SpecificAnno(io.in, s"ModuleB($bParam1,$bParam2) width < $bParam1"))
  annotate(GeneralAnno(io.out, s"ModuleB($bParam1,$bParam2) width < $bParam2"))
  annotate(SpecificAnno(modC.io.in, s"ModuleB.c.io.in absolute"))
  annotate(GeneralAnno(modC.io.in, s"ModuleB.c.io.in relative"))
}

class TopOfDiamond extends Module {
  val io = new Bundle {
    val in   = UInt(INPUT, 32)
    val out  = UInt(INPUT, 32)
  }
  val x = Reg(UInt(width = 32))
  val y = Reg(UInt(width = 32))

  val modA = Module(new ModA(64, 64))
  val modB = Module(new ModB(32, 48))

  x := io.in
  modA.io.in := x
  modB.io.in := x

  y := modA.io.out + modB.io.out
  io.out := y

  annotate(SpecificAnno(this, s"TopOfDiamond Absolute"))
  annotate(GeneralAnno(this, s"TopOfDiamond Relative"))

  annotate(SpecificAnno(modB.io.in, s"TopOfDiamond.moduleB.io.in"))
  annotate(GeneralAnno(modB.io.in, s"TopOfDiamond.moduleB.io.in"))
}

class DiamondTester extends BasicTester {
  val dut = Module(new TopOfDiamond)

  stop()
}

class AnnotatingDiamondSpec extends FreeSpec with Matchers {
  def getValue(a: Annotation): String = {
    a match {
      case SpecificAnno(_, value1) => value1
      case GeneralAnno(_, value2) => value2
    }
  }
  def findAnno(as: Seq[Annotation], name: String): Option[Annotation] = {
    as.find { a => a.fullName == name }
  }

  """
    |Diamond is an example of a module that has two submoduesl A and B who both instantiate their
    |own instances of module C.  This highlights the difference between specific and general
    |annotation scopes
  """.stripMargin - {

    """
      |annotations are not resolved at after circuit elaboration,
      |that happens only after emit has been called on circuit""".stripMargin in {
      val circuit = Driver.elaborate { () => new DiamondTester }
      // val e = Driver.emit(circuit)
      val annotations =  circuit.annotations

      annotations.forall { a =>
        !a.isResolved } should be (true)
    }
    "dumpFirrtl invokes the emitter so now annotations should be resolved" in {
      val circuit = Driver.elaborate { () => new DiamondTester }
      val annotations =  circuit.annotations
      Driver.dumpFirrtlWithAnnotations(circuit)

      annotations.forall { _.isResolved } should be (true)
    }

    "a bunch of annotations are created" - {
      val circuit = Driver.elaborate { () => new DiamondTester }
      val annotations =  circuit.annotations
      Driver.dumpFirrtlWithAnnotations(circuit)

      """
        |a general annotation in module c, will appear multiple times in list
        |and thus is a bad way to try and capture module parameterization
      """.stripMargin in {
        annotations.count{ a => a.fullName == "ModC.io.out"} should be > 1
      }
      "a specific annotation in module C will only appear once" in {
        annotations.count { a => a.fullName == "DiamondTester.dut.modA.modC.io.in" } should be(1)
      }

      """
        |Although specific can appear twice, in this example B annotated Cs out specifically.
        |in addition to C's doing it internally""".stripMargin in {
        annotations.count{ a => a.fullName == "DiamondTester.dut.modB.modC.io.in"} should be (2)

      }
      //scalastyle:off regex
      //  "show the list" in {
      //    println(circuit.annotations.map { a => s"${a.firrtlInstanceName} ${getValue(a)}" }.mkString("\n"))
      //  }
      //scalastyle:on regex
    }
  }
}