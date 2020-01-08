// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir.StringLit

import java.io._

import scala.sys.process._
import annotation.tailrec
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.Assertions._
import org.scalacheck._
import org.scalacheck.Arbitrary._

class PrintfSpec extends FirrtlPropSpec {

  property("Printf should correctly print values in each format %x, %d, %b") {
    val prefix = "Printf"
    val testDir = compileFirrtlTest(prefix, "/features")
    val harness = new File(testDir, s"top.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)

    verilogToCpp(prefix, testDir, Seq(), harness) #&&
    cppToExe(prefix, testDir) ! loggingProcessLogger

    // Check for correct Printf:
    // Count up from 0, match decimal, hex, and binary
    // see /features/Print.fir to see what we're matching
    val regex = """\tcount\s+=\s+(\d+)\s+0x(\w+)\s+b([01]+).*""".r
    var done = false
    var expected = 0
    var error = false
    val ret = Process(s"./V${prefix}", testDir) !
      ProcessLogger( line => {
        line match {
          case regex(dec, hex, bin) => {
            if (!done) {
              // Must mark error before assertion or sbt test will pass
              if (Integer.parseInt(dec, 10) != expected) error = true
              assert(Integer.parseInt(dec, 10) == expected)
              if (Integer.parseInt(hex, 16) != expected) error = true
              assert(Integer.parseInt(hex, 16) == expected)
              if (Integer.parseInt(bin, 2) != expected) error = true
              assert(Integer.parseInt(bin, 2) == expected)
              expected += 1
            }
          }
          case _ => // Do Nothing
        }
      })
    if (error) fail()
  }
}

class StringSpec extends FirrtlPropSpec {

  // Whitelist is [0x20 - 0x7e]
  val whitelist =
    """ !\"#$%&\''()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ""" +
    """[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""

  property(s"Character whitelist should be supported: [$whitelist] ") {
    val lit = StringLit.unescape(whitelist)
    // We accept \' but don't bother escaping it ourselves
    val res = whitelist.replaceAll("""\\'""", "'")
    // Check result
    assert(lit.serialize == res)
    assert(lit.verilogEscape.tail.init == res)
  }

  // Valid escapes = \n, \t, \\, \", \'
  val esc = """\\\'\"\t\n"""
  val validEsc = Seq('n', 't', '\\', '"', '\'')
  property(s"Escape characters [$esc] should parse") {
    val lit = StringLit.unescape(esc)
    assert(lit.string(0).toByte == 0x5c)
    assert(lit.string(1).toByte == 0x27)
    assert(lit.string(2).toByte == 0x22)
    assert(lit.string(3).toByte == 0x09)
    assert(lit.string(4).toByte == 0x0a)
    assert(lit.string.length == 5)
  }

  // From IEEE 1364-2001 2.6
  def isValidVerilogString(str: String): Boolean = {
    @tailrec def rec(xs: List[Char]): Boolean = xs match {
      case Nil => true
      case '\\' :: esc =>
        if (Set('n', 't', '\\', '"').contains(esc.head)) rec(esc.tail)
        else { // Check valid octal escape, otherwise illegal
          val next3 = esc.take(3)
          if (next3.size == 3 && next3.forall(('0' to '7').toSet.contains)) rec(esc.drop(3))
          else false
        }
      case char :: tail => // Check Legal ASCII
        if (char.toInt < 256 && char.toInt >= 0) rec(tail)
        else false
    }
    rec(str.toList)
  }
  // From IEEE 1364-2001 17.1.1.2
  val legalFormats = "HhDdOoBbCcLlVvMmSsTtUuZz%".toSet
  def isValidVerilogFormat(str: String): Boolean = str.toSeq.sliding(2).forall {
    case Seq('%', char) if legalFormats contains char => true
    case _ => true
  }

  // Generators for legal Firrtl format strings
  val genFormat = Gen.oneOf("bdxc%").map(List('%', _))
  val genEsc = Gen.oneOf(esc.toSeq).map(List(_))
  val genChar = arbitrary[Char].suchThat(c => (c != '%' && c != '\\')).map(List(_))
  val genFragment = Gen.frequency((10, genChar), (1, genFormat), (1, genEsc)).map(_.mkString)
  val genString = Gen.listOf[String](genFragment).map(_.mkString)

  property ("Firrtl Format Strings with Unicode chars should emit as legal Verilog Strings") {
    forAll (genString) { str =>
      val verilogStr = StringLit(str).verilogFormat.verilogEscape
      assert(isValidVerilogString(verilogStr))
      assert(isValidVerilogFormat(verilogStr))
    }
  }
}
