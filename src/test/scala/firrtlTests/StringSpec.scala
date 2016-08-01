/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtlTests

import firrtl._

import java.io._

import scala.sys.process._
import org.scalatest._
import org.scalatest.prop._
import org.scalatest.Assertions._
import org.scalacheck._

class PrintfSpec extends FirrtlPropSpec {

  property("Printf should correctly print values in each format %x, %d, %b") {
    val prefix = "Printf"
    val testDir = compileFirrtlTest(prefix, "/features")
    val harness = new File(testDir, s"top.cpp")
    copyResourceToFile(cppHarness.toString, harness)

    verilogToCpp(prefix, testDir, Seq(), harness).!
    cppToExe(prefix, testDir).!

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
    """ !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ""" +
    """[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"""
  val whitelistBA: Array[Byte] = Array.range(0x20, 0x7e) map (_.toByte)

  property(s"Character whitelist should be supported: [$whitelist] ") {
    val lit = firrtl.FIRRTLStringLitHandler.unescape(whitelist)
    // Check internal
    lit.array zip whitelistBA foreach { case (b, e) =>
      assert(b == e, s"(${b.toChar} did not equal expected ${e.toChar})")
    }
    // Check result
    assert(lit.serialize == whitelist)
  }

  // Valid escapes = \n, \t, \\, \", \'
  val esc = """\\\'\"\t\n"""
  val validEsc = Seq('n', 't', '\\', '"', '\'')
  property(s"Escape characters [$esc] should parse") {
    val lit = firrtl.FIRRTLStringLitHandler.unescape(esc)
    assert(lit.array(0) == 0x5c) 
    assert(lit.array(1) == 0x27)
    assert(lit.array(2) == 0x22)
    assert(lit.array(3) == 0x09)
    assert(lit.array(4) == 0x0a)
    assert(lit.array.length == 5)
  }

  // Generators for random testing
  val validChar = Gen.oneOf((0x20.toChar to 0x5b.toChar).toSeq ++
                             (0x5e.toChar to 0x7e.toChar).toSeq) // exclude '\\'
  val validCharSeq = Gen.containerOf[Seq, Char](validChar)
  val invalidChar = Gen.oneOf(Gen.choose(0x00.toChar, 0x1f.toChar),
                             Gen.choose(0x7f.toChar, 0xff.toChar))
  val invalidEsc = Gen.oneOf((0x00.toChar to 0xff.toChar).toSeq diff validEsc)

  property("Random invalid strings should fail") {
    forAll(validCharSeq, invalidChar, validCharSeq) { 
      (head: Seq[Char], bad: Char, tail: Seq[Char]) =>
        val str = ((head :+ bad) ++ tail).mkString
        intercept[InvalidStringLitException] {
          firrtl.FIRRTLStringLitHandler.unescape(str)
        }
    }
  }
    
  property(s"Invalid escape characters should fail") {
    forAll(validCharSeq, invalidEsc, validCharSeq) {
      (head: Seq[Char], badEsc: Char, tail: Seq[Char]) =>
        val str = (head ++ Seq('\\', badEsc) ++ tail).mkString
        intercept[InvalidEscapeCharException] {
          firrtl.FIRRTLStringLitHandler.unescape(str)
        }
    }
  }
}
