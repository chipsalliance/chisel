// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.test._
import chisel3.testing.scalatest.FileCheck
import java.io.{ByteArrayOutputStream, PrintStream}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class UnitTestMainSpec extends AnyFlatSpec with Matchers with FileCheck {
  def check(args: Seq[String])(checkOut: String, checkErr: String): Unit = {
    val outStream = new ByteArrayOutputStream()
    val errStream = new ByteArrayOutputStream()
    Console.withOut(new PrintStream(outStream)) {
      Console.withErr(new PrintStream(errStream)) {
        UnitTests.main(args.toArray)
      }
    }
    if (!checkOut.isEmpty) outStream.toString.fileCheck("--allow-empty")(checkOut)
    if (!checkErr.isEmpty) errStream.toString.fileCheck("--allow-empty")(checkErr)
  }

  def checkOutAndErr(args: String*)(checkOut: String, checkErr: String): Unit = check(args)(checkOut, checkErr)

  def checkOut(args: String*)(checkOut: String): Unit = check(args)(checkOut, "")
  def checkErr(args: String*)(checkErr: String): Unit = check(args)("", checkErr)

  it should "print a help page" in {
    checkOut("-h")("""
      // CHECK: Chisel Unit Test Utility
      // CHECK: Usage:
      // CHECK: -h, --help
      """)
  }

  it should "list unit tests" in {
    checkOutAndErr("-l", "-f", "^chiselTests\\.sampleTests\\.")(
      """
      // CHECK-DAG: chiselTests.sampleTests.ClassTest
      // CHECK-DAG: chiselTests.sampleTests.ObjectTest
      // CHECK-DAG: chiselTests.sampleTests.ModuleTest
      """,
      """
      // CHECK-NOT: Hello from
      """
    )
  }

  it should "execute unit test constructors" in {
    checkErr("-f", "^chiselTests\\.sampleTests\\.")("""
      // CHECK-DAG: Hello from class test
      // CHECK-DAG: Hello from object test
      // CHECK-DAG: Hello from module test
      """)
  }

  it should "generate unit test FIRRTL" in {
    checkOut("-f", "^chiselTests\\.sampleTests\\.")("""
      // CHECK: module ModuleTest :
      """)
  }

  it should "support custom runpaths" in {
    val runpathArgs = System
      .getProperty("java.class.path")
      .split(java.io.File.pathSeparator)
      .map(s => if (s.trim.length == 0) "." else s)
      .flatMap(s => Seq("-R", s))
    val args = Seq("-l", "-f", "^chiselTests\\.sampleTests\\.") ++ runpathArgs
    checkOut(args: _*)(
      """
      // CHECK-DAG: chiselTests.sampleTests.ClassTest
      // CHECK-DAG: chiselTests.sampleTests.ObjectTest
      // CHECK-DAG: chiselTests.sampleTests.ModuleTest
      """
    )
  }
}

package sampleTests {
  class ClassTest extends UnitTest {
    Console.err.println("Hello from class test")
  }
  object ObjectTest extends UnitTest {
    Console.err.println("Hello from object test")
  }
  class ModuleTest extends RawModule with UnitTest {
    Console.err.println("Hello from module test")
  }
}
