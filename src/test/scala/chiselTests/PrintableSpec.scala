package chiselTests

import org.scalatest.{FlatSpec, Matchers}
import scala.collection.mutable

import chisel3._
import chisel3.testers.BasicTester

/* Printable Tests */
class PrintableSpec extends FlatSpec with Matchers {
  private val PrintfRegex = """\s*printf\((.*)\).*""".r
  // This regex is brittle, it relies on the first two arguments of the printf
  // not containing quotes, problematic if Chisel were to emit UInt<1>("h01")
  // instead of the current UInt<1>(1) for the enable signal
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
      val myWire = Wire(init = UInt(1234))
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
      printf(Decimal(UInt(10, 32)))
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
  it should "support names of circuit elements and the current module" in {
    class MyBundle extends Bundle {
      val foo = UInt(width = 32)
      override def cloneType = (new MyBundle).asInstanceOf[this.type]
    }
    class MyModule extends BasicTester {
      override def desiredName = "MyModule"
      val myWire = Wire(new MyBundle)
      printf(p"${Name(myWire.foo)}")
      printf(p"${FullName(myWire.foo)}")
      printf(p"${FullName(this)}")
    }
    val firrtl = Driver.emit(() => new MyModule)
    getPrintfs(firrtl) match {
      case Seq(Printf("foo", Seq()),
               Printf("myWire.foo", Seq()),
               Printf("MyModule", Seq())) =>
      case e => fail()
    }
  }
  it should "print UInts and SInts as Decimal by default" in {
    class MyModule extends BasicTester {
      val myUInt = Wire(init = UInt(0))
      val mySInt = Wire(init = SInt(-1))
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
      val myVec = Wire(Vec(4, UInt(width = 32)))
      myVec foreach (_ := UInt(0))
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
        val foo = UInt(width = 32)
        val bar = UInt(width = 32)
      })
      myBun.foo := UInt(0)
      myBun.bar := UInt(0)
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
