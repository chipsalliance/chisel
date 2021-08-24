// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.{BaseSim, ChiselAnnotation}
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import firrtl.annotations.{ReferenceTarget, SingleTargetAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File

/** Dummy [[printf]] annotation.
  * @param target target of component to be annotated
  */
case class PrintfAnnotation(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
  def duplicate(n: ReferenceTarget): PrintfAnnotation = this.copy(target = n)
}

object PrintfAnnotation {
  /** Create annotation for a given [[printf]].
    * @param c component to be annotated
    */
  def annotate(c: BaseSim): Unit = {
    chisel3.experimental.annotate(new ChiselAnnotation {
      def toFirrtl: PrintfAnnotation = PrintfAnnotation(c.toTarget)
    })
  }
}

/* Printable Tests */
class PrintableSpec extends AnyFlatSpec with Matchers {
  // This regex is brittle, it specifically finds the clock and enable signals followed by commas
  private val PrintfRegex = """\s*printf\(\w+, [^,]+,(.*)\).*""".r
  private val StringRegex = """([^"]*)"(.*?)"(.*)""".r
  private case class Printf(str: String, args: Seq[String])
  private def getPrintfs(firrtl: String): Seq[Printf] = {
    def processArgs(str: String): Seq[String] =
      str split "," map (_.trim) filter (_.nonEmpty)
    def processBody(str: String): (String, Seq[String]) = {
      str match {
        case StringRegex(_, fmt, args) =>
          (fmt, processArgs(args))
        case _ => fail(s"Regex to process Printf should work on $str!")
      }
    }

    firrtl split "\n" collect {
      case PrintfRegex(matched) =>
        val (str, args) = processBody(matched)
        Printf(str, args)
    }
  }

  behavior of "Printable & Custom Interpolator"

  it should "pass exact strings through" in {
    class MyModule extends BasicTester {
      printf(p"An exact string")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("An exact string", Seq())) =>
      case e => fail()
    }
  }
  it should "handle Printable and String concatination" in {
    class MyModule extends BasicTester {
      printf(p"First " + PString("Second ") + "Third")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("First Second Third", Seq())) =>
      case e => fail()
    }
  }
  it should "call toString on non-Printable objects" in {
    class MyModule extends BasicTester {
      val myInt = 1234
      printf(p"myInt = $myInt")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("myInt = 1234", Seq())) =>
      case e => fail()
    }
  }
  it should "generate proper printf for simple Decimal printing" in {
    class MyModule extends BasicTester {
      val myWire = WireDefault(1234.U)
      printf(p"myWire = ${Decimal(myWire)}")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("myWire = %d", Seq("myWire"))) =>
      case e => fail()
    }
  }
  it should "handle printing literals" in {
    class MyModule extends BasicTester {
      printf(Decimal(10.U(32.W)))
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("%d", Seq(lit))) =>
        assert(lit contains "UInt<32>")
      case e => fail()
    }
  }
  it should "correctly escape percent" in {
    class MyModule extends BasicTester {
      printf(p"%")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("%%", Seq())) =>
      case e => fail()
    }
  }
  it should "correctly emit tab" in {
    class MyModule extends BasicTester {
      printf(p"\t")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("\\t", Seq())) =>
      case e => fail()
    }
  }
  it should "support names of circuit elements including submodule IO" in {
    // Submodule IO is a subtle issue because the Chisel element has a different
    // parent module
    class MySubModule extends Module {
      val io = IO(new Bundle {
        val fizz = UInt(32.W)
      })
    }
    class MyBundle extends Bundle {
      val foo = UInt(32.W)
      override def cloneType: this.type = (new MyBundle).asInstanceOf[this.type]
    }
    class MyModule extends BasicTester {
      override def desiredName: String = "MyModule"
      val myWire = Wire(new MyBundle)
      val myInst = Module(new MySubModule)
      printf(p"${Name(myWire.foo)}")
      printf(p"${FullName(myWire.foo)}")
      printf(p"${FullName(myInst.io.fizz)}")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("foo", Seq()),
               Printf("myWire.foo", Seq()),
               Printf("myInst.io.fizz", Seq())) =>
      case e => fail()
    }
  }
  it should "handle printing ports of submodules" in {
    class MySubModule extends Module {
      val io = IO(new Bundle {
        val fizz = UInt(32.W)
      })
    }
    class MyModule extends BasicTester {
      val myInst = Module(new MySubModule)
      printf(p"${myInst.io.fizz}")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("%d", Seq("myInst.io.fizz"))) =>
      case e => fail()
    }
  }
  it should "print UInts and SInts as Decimal by default" in {
    class MyModule extends BasicTester {
      val myUInt = WireDefault(0.U)
      val mySInt = WireDefault(-1.S)
      printf(p"$myUInt & $mySInt")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("%d & %d", Seq("myUInt", "mySInt"))) =>
      case e => fail()
    }
  }
  it should "print Vecs like Scala Seqs by default" in {
    class MyModule extends BasicTester {
      val myVec = Wire(Vec(4, UInt(32.W)))
      myVec foreach (_ := 0.U)
      printf(p"$myVec")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("Vec(%d, %d, %d, %d)",
               Seq("myVec[0]", "myVec[1]", "myVec[2]", "myVec[3]"))) =>
      case e => fail()
    }
  }
  it should "print Bundles like Scala Maps by default" in {
    class MyModule extends BasicTester {
      val myBun = Wire(new Bundle {
        val foo = UInt(32.W)
        val bar = UInt(32.W)
      })
      myBun.foo := 0.U
      myBun.bar := 0.U
      printf(p"$myBun")
    }
    val firrtl = ChiselStage.emitChirrtl(new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("AnonymousBundle(foo -> %d, bar -> %d)",
               Seq("myBun.foo", "myBun.bar"))) =>
      case e => fail()
    }
  }
  it should "get emitted with a name and annotated" in {

    /** Test circuit containing annotated and renamed [[printf]]s. */
    class PrintfAnnotationTest extends Module {
      val myBun = Wire(new Bundle {
        val foo = UInt(32.W)
        val bar = UInt(32.W)
      })
      myBun.foo := 0.U
      myBun.bar := 0.U
      val howdy = printf(p"hello ${myBun}")
      PrintfAnnotation.annotate(howdy)
      PrintfAnnotation.annotate(printf(p"goodbye $myBun"))
      PrintfAnnotation.annotate(printf(p"adieu $myBun").suggestName("farewell"))
    }

    // compile circuit
    val testDir = new File("test_run_dir", "PrintfAnnotationTest")
    (new ChiselStage).emitSystemVerilog(
      gen = new PrintfAnnotationTest,
      args = Array("-td", testDir.getPath)
    )

    // read in annotation file
    val annoFile = new File(testDir, "PrintfAnnotationTest.anno.json")
    annoFile should exist
    val annoLines = scala.io.Source.fromFile(annoFile).getLines.toList

    // check for expected annotations
    exactly(3, annoLines) should include ("chiselTests.PrintfAnnotation")
    exactly(1, annoLines) should include ("~PrintfAnnotationTest|PrintfAnnotationTest>farewell")
    exactly(1, annoLines) should include ("~PrintfAnnotationTest|PrintfAnnotationTest>SIM")
    exactly(1, annoLines) should include ("~PrintfAnnotationTest|PrintfAnnotationTest>howdy")

    // read in FIRRTL file
    val firFile = new File(testDir, "PrintfAnnotationTest.fir")
    firFile should exist
    val firLines = scala.io.Source.fromFile(firFile).getLines.toList

    // check that verification components have expected names
    exactly(1, firLines) should include ("""printf(clock, UInt<1>("h1"), "hello AnonymousBundle(foo -> %d, bar -> %d)", myBun.foo, myBun.bar) : howdy""")
    exactly(1, firLines) should include ("""printf(clock, UInt<1>("h1"), "goodbye AnonymousBundle(foo -> %d, bar -> %d)", myBun.foo, myBun.bar) : SIM""")
    exactly(1, firLines) should include ("""printf(clock, UInt<1>("h1"), "adieu AnonymousBundle(foo -> %d, bar -> %d)", myBun.foo, myBun.bar) : farewell""")
  }
}
