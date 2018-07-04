package chiselTests

import org.scalatest.{FlatSpec, Matchers}
import scala.collection.mutable

import chisel3._
import chisel3.testers.BasicTester

/* Printable Tests */
class PrintableSpec extends FlatSpec with Matchers {
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
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("An exact string", Seq())) =>
      case e => fail()
    }
  }
  it should "handle Printable and String concatination" in {
    class MyModule extends BasicTester {
      printf(p"First " + PString("Second ") + "Third")
    }
    val firrtl = Driver.emit(() => new MyModule)
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
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("myInt = 1234", Seq())) =>
      case e => fail()
    }
  }
  it should "generate proper printf for simple Decimal printing" in {
    class MyModule extends BasicTester {
      val myWire = WireInit(1234.U)
      printf(p"myWire = ${Decimal(myWire)}")
    }
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("myWire = %d", Seq("myWire"))) =>
      case e => fail()
    }
  }
  it should "handle printing literals" in {
    class MyModule extends BasicTester {
      printf(Decimal(10.U(32.W)))
    }
    val firrtl = Driver.emit(() => new MyModule)
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
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("%%", Seq())) =>
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
      override def cloneType = (new MyBundle).asInstanceOf[this.type]
    }
    class MyModule extends BasicTester {
      override def desiredName = "MyModule"
      val myWire = Wire(new MyBundle)
      val myInst = Module(new MySubModule)
      printf(p"${Name(myWire.foo)}")
      printf(p"${FullName(myWire.foo)}")
      printf(p"${FullName(myInst.io.fizz)}")
    }
    val firrtl = Driver.emit(() => new MyModule)
    println(firrtl)
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
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("%d", Seq("myInst.io.fizz"))) =>
      case e => fail()
    }
  }
  it should "print UInts and SInts as Decimal by default" in {
    class MyModule extends BasicTester {
      val myUInt = WireInit(0.U)
      val mySInt = WireInit(-1.S)
      printf(p"$myUInt & $mySInt")
    }
    val firrtl = Driver.emit(() => new MyModule)
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
    val firrtl = Driver.emit(() => new MyModule)
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
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("Bundle(foo -> %d, bar -> %d)",
               Seq("myBun.foo", "myBun.bar"))) =>
      case e => fail()
    }
  }
}
