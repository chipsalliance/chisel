// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import circt.stage.ChiselStage
import org.scalactic.source.Position
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/* Printable Tests */
class PrintableSpec extends AnyFlatSpec with Matchers with Utils {
  // This regex is brittle, it specifically finds the clock and enable signals followed by commas
  private val PrintfRegex = """\s*printf\(\w+, [^,]+,(.*)\).*""".r
  private val StringRegex = """([^"]*)"(.*?)"(.*)""".r
  private case class Printf(str: String, args: Seq[String])
  private def getPrintfs(firrtl: String): Seq[Printf] = {
    def processArgs(str: String): Seq[String] =
      str.split(",").toIndexedSeq.map(_.trim).filter(_.nonEmpty)
    def processBody(str: String): (String, Seq[String]) = {
      str match {
        case StringRegex(_, fmt, args) =>
          (fmt, processArgs(args))
        case _ => fail(s"Regex to process Printf should work on $str!")
      }
    }
    firrtl.split("\n").toIndexedSeq.collect {
      case PrintfRegex(matched) =>
        val (str, args) = processBody(matched)
        Printf(str, args)
    }
  }

  // Generates firrtl, gets Printfs
  // Calls fail() if failed match; else calls the partial function which could have its own check
  private def generateAndCheck(gen: => RawModule)(check: PartialFunction[Seq[Printf], Unit])(implicit pos: Position) = {
    val firrtl = ChiselStage.emitCHIRRTL(gen)
    val printfs = getPrintfs(firrtl)
    if (!check.isDefinedAt(printfs)) {
      fail()
    } else {
      check(printfs)
    }
  }

  behavior.of("Printable & Custom Interpolator")

  it should "pass exact strings through" in {
    class MyModule extends BasicTester {
      printf(p"An exact string")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("An exact string", Seq())) =>
    }
  }
  it should "handle Printable and String concatenation" in {
    class MyModule extends BasicTester {
      printf(p"First " + PString("Second ") + "Third")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("First Second Third", Seq())) =>
    }
  }
  it should "call toString on non-Printable objects" in {
    class MyModule extends BasicTester {
      val myInt = 1234
      printf(p"myInt = $myInt")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("myInt = 1234", Seq())) =>
    }
  }
  it should "generate proper printf for simple Decimal printing" in {
    class MyModule extends BasicTester {
      val myWire = WireDefault(1234.U)
      printf(p"myWire = ${Decimal(myWire)}")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("myWire = %d", Seq("myWire"))) =>
    }
  }
  it should "handle printing literals" in {
    class MyModule extends BasicTester {
      printf(Decimal(10.U(32.W)))
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("%d", Seq(lit))) =>
        assert(lit contains "UInt<32>")
    }
  }
  it should "correctly escape percent" in {
    class MyModule extends BasicTester {
      printf(p"%")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("%%", Seq())) =>
    }
  }
  it should "correctly emit tab" in {
    class MyModule extends BasicTester {
      printf(p"\t")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("\\t", Seq())) =>
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
    }
    class MyModule extends BasicTester {
      override def desiredName: String = "MyModule"
      val myWire = Wire(new MyBundle)
      val myInst = Module(new MySubModule)
      printf(p"${Name(myWire.foo)}")
      printf(p"${FullName(myWire.foo)}")
      printf(p"${FullName(myInst.io.fizz)}")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("foo", Seq()), Printf("myWire.foo", Seq()), Printf("myInst.io.fizz", Seq())) =>
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
    generateAndCheck(new MyModule) {
      case Seq(Printf("%d", Seq("myInst.io.fizz"))) =>
    }
  }
  it should "print UInts and SInts as Decimal by default" in {
    class MyModule extends BasicTester {
      val myUInt = WireDefault(0.U)
      val mySInt = WireDefault(-1.S)
      printf(p"$myUInt & $mySInt")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("%d & %d", Seq("myUInt", "mySInt"))) =>
    }
  }
  it should "print Vecs like Scala Seqs by default" in {
    class MyModule extends BasicTester {
      val myVec = Wire(Vec(4, UInt(32.W)))
      myVec.foreach(_ := 0.U)
      printf(p"$myVec")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("Vec(%d, %d, %d, %d)", Seq("myVec[0]", "myVec[1]", "myVec[2]", "myVec[3]"))) =>
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
    generateAndCheck(new MyModule) {
      case Seq(Printf("AnonymousBundle(foo -> %d, bar -> %d)", Seq("myBun.foo", "myBun.bar"))) =>
    }
  }

  it should "print use the sanitized names of Bundle elements" in {
    class MyModule extends Module {
      class UnsanitaryBundle extends Bundle {
        val `a-x` = UInt(8.W)
      }
      val myBun = Wire(new UnsanitaryBundle)
      myBun.`a-x` := 0.U
      printf(p"$myBun")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("UnsanitaryBundle(aminusx -> %d)", Seq("myBun.aminusx"))) =>
    }
  }

  // Unit tests for cf
  it should "print regular scala variables with cf format specifier" in {

    class MyModule extends BasicTester {
      val f1 = 20.4517
      val i1 = 10
      val str1 = "String!"
      printf(
        cf"F1 = $f1 D1 = $i1 F1 formatted = $f1%2.2f str1 = $str1%s i1_str = $i1%s i1_hex=$i1%x"
      )

    }

    generateAndCheck(new MyModule) {
      case Seq(Printf("F1 = 20.4517 D1 = 10 F1 formatted = 20.45 str1 = String! i1_str = 10 i1_hex=a", Seq())) =>
    }
  }

  it should "print chisel bits with cf format specifier" in {

    class MyBundle extends Bundle {
      val foo = UInt(32.W)
      val bar = UInt(32.W)
      override def toPrintable: Printable = {
        cf"Bundle : " +
          cf"Foo : $foo%x Bar : $bar%x"
      }
    }
    class MyModule extends BasicTester {
      val b1 = 10.U
      val w1 = Wire(new MyBundle)
      w1.foo := 5.U
      w1.bar := 10.U
      printf(cf"w1 = $w1")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("w1 = Bundle : Foo : %x Bar : %x", Seq("w1.foo", "w1.bar"))) =>
    }
  }

  it should "support names of circuit elements using format specifier including submodule IO with cf format specifier" in {
    // Submodule IO is a subtle issue because the Chisel element has a different
    // parent module
    class MySubModule extends Module {
      val io = IO(new Bundle {
        val fizz = UInt(32.W)
      })
    }
    class MyBundle extends Bundle {
      val foo = UInt(32.W)
    }
    class MyModule extends BasicTester {
      override def desiredName: String = "MyModule"
      val myWire = Wire(new MyBundle)
      val myInst = Module(new MySubModule)
      printf(cf"${myWire.foo}%n")
      printf(cf"${myWire.foo}%N")
      printf(cf"${myInst.io.fizz}%N")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("foo", Seq()), Printf("myWire.foo", Seq()), Printf("myInst.io.fizz", Seq())) =>
    }
  }

  it should "correctly print strings after modifier" in {
    class MyModule extends BasicTester {
      val b1 = 10.U
      printf(cf"This is here $b1%x!!!! And should print everything else")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("This is here %x!!!! And should print everything else", Seq("UInt<4>(0ha)"))) =>
    }
  }

  it should "correctly print strings with a lot of literal %% and different format specifiers for Wires" in {
    class MyModule extends BasicTester {
      val b1 = 10.U
      val b2 = 20.U
      printf(cf"%%  $b1%x%%$b2%b = ${b1 % b2}%d %%%% Tail String")
    }

    generateAndCheck(new MyModule) {
      case Seq(Printf("%%  %x%%%b = %d %%%% Tail String", Seq(lita, litb, _))) =>
        assert(lita.contains("UInt<4>") && litb.contains("UInt<5>"))
    }
  }

  it should "not allow unescaped % in the message" in {
    class MyModule extends BasicTester {
      printf(cf"This should error out for sure because of % - it should be %%")
    }
    a[java.util.UnknownFormatConversionException] should be thrownBy {
      extractCause[java.util.UnknownFormatConversionException] {
        ChiselStage.emitCHIRRTL { new MyModule }
      }
    }
  }

  it should "allow Printables to be expanded and used" in {
    class MyModule extends BasicTester {
      val w1 = 20.U
      val f1 = 30.2
      val i1 = 14
      val pable = cf"w1 = $w1%b f1 = $f1%2.2f"
      printf(cf"Trying to expand printable $pable and mix with i1 = $i1%d")
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("Trying to expand printable w1 = %b f1 = 30.20 and mix with i1 = 14", Seq(lit))) =>
        assert(lit.contains("UInt<5>"))
    }
  }

  it should "fail with a single  % in the message" in {
    class MyModule extends BasicTester {
      printf(cf"%")
    }
    a[java.util.UnknownFormatConversionException] should be thrownBy {
      extractCause[java.util.UnknownFormatConversionException] {
        ChiselStage.emitCHIRRTL { new MyModule }
      }
    }
  }

  it should "fail when passing directly to StirngContext.cf a string with  literal \\ correctly escaped  " in {
    a[StringContext.InvalidEscapeException] should be thrownBy {
      extractCause[StringContext.InvalidEscapeException] {
        val s_seq = Seq("Test with literal \\ correctly escaped")
        StringContext(s_seq: _*).cf(Seq(): _*)
      }
    }
  }

  it should "pass correctly escaped \\ when using Printable.pack" in {
    class MyModule extends BasicTester {
      printf(Printable.pack("\\ \\]"))
    }
    generateAndCheck(new MyModule) {
      case Seq(Printf("\\\\ \\\\]", Seq())) =>
    }
  }
}
