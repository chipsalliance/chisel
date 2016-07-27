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

import java.io._

import com.typesafe.scalalogging.LazyLogging
import scala.sys.process._
import org.scalatest._
import org.scalatest.prop._
import scala.io.Source

import firrtl._
import firrtl.Annotations.AnnotationMap

// This trait is borrowed from Chisel3, ideally this code should only exist in one location
trait BackendCompilationUtilities {
  /** Create a temporary directory with the prefix name. Exists here because it doesn't in Java 6.
    */
  def createTempDirectory(prefix: String): File = {
    val temp = File.createTempFile(prefix, "")
    if (!temp.delete()) {
      throw new IOException(s"Unable to delete temp file '$temp'")
    }
    if (!temp.mkdir()) {
      throw new IOException(s"Unable to create temp directory '$temp'")
    }
    temp
  }

  /** Copy the contents of a resource to a destination file.
    */
  def copyResourceToFile(name: String, file: File) {
    val in = getClass().getResourceAsStream(name)
    if (in == null) {
      throw new FileNotFoundException(s"Resource '$name'")
    }
    val out = new FileOutputStream(file)
    Iterator.continually(in.read).takeWhile(-1 !=).foreach(out.write)
    out.close()
  }


  def makeHarness(template: String => String, post: String)(f: File): File = {
    val prefix = f.toString.split("/").last
    val vf = new File(f.toString + post)
    val w = new FileWriter(vf)
    w.write(template(prefix))
    w.close()
    vf
  }

  /** Generates a Verilator invocation to convert Verilog sources to C++
    * simulation sources.
    *
    * The Verilator prefix will be V$dutFile, and running this will generate
    * C++ sources and headers as well as a makefile to compile them.
    *
    * Verilator will automatically locate the top-level module as the one among
    * all the files which are not included elsewhere. If multiple ones exist,
    * the compilation will fail.
    *
    * @param dutFile name of the DUT .v without the .v extension
    * @param dir output directory
    * @param vSources list of additional Verilog sources to compile
    * @param cppHarness C++ testharness to compile/link against
    */
  def verilogToCpp(
      dutFile: String,
      dir: File,
      vSources: Seq[File],
      cppHarness: File): ProcessBuilder =

    Seq("verilator",
        "--cc", s"$dutFile.v") ++
        vSources.map(file => Seq("-v", file.toString)).flatten ++
        Seq("--assert",
            "--Wno-fatal",
            "--trace",
            "-O2",
            "--top-module", dutFile,
            "+define+TOP_TYPE=V" + dutFile,
            "-CFLAGS", s"""-Wno-undefined-bool-conversion -O2 -DTOP_TYPE=V$dutFile -include V$dutFile.h""",
            "-Mdir", dir.toString,
            "--exe", cppHarness.toString)

  def cppToExe(prefix: String, dir: File): ProcessBuilder =
    Seq("make", "-C", dir.toString, "-j", "-f", s"V${prefix}.mk", s"V${prefix}")

  def executeExpectingFailure(
      prefix: String,
      dir: File,
      assertionMsg: String = "Assertion failed"): Boolean = {
    var triggered = false
    val e = Process(s"./V${prefix}", dir) !
      ProcessLogger(line => {
        triggered = triggered || line.contains(assertionMsg)
        System.out.println(line)
      })
    triggered
  }

  def executeExpectingSuccess(prefix: String, dir: File): Boolean = {
    !executeExpectingFailure(prefix, dir)
  }
}

trait FirrtlRunners extends BackendCompilationUtilities {
  lazy val cppHarness = new File(s"/top.cpp")
  def compileFirrtlTest(
      prefix: String,
      srcDir: String,
      annotations: AnnotationMap = new AnnotationMap(Seq.empty)): File = {
    val testDir = createTempDirectory(prefix)
    copyResourceToFile(s"${srcDir}/${prefix}.fir", new File(testDir, s"${prefix}.fir"))

    Driver.compile(
      s"$testDir/$prefix.fir",
      s"$testDir/$prefix.v",
      new VerilogCompiler(),
      Parser.IgnoreInfo,
      annotations)
    testDir
  }
  def runFirrtlTest(
      prefix: String,
      srcDir: String,
      annotations: AnnotationMap = new AnnotationMap(Seq.empty)) = {
    val testDir = compileFirrtlTest(prefix, srcDir, annotations)
    val harness = new File(testDir, s"top.cpp")
    copyResourceToFile(cppHarness.toString, harness)

    verilogToCpp(prefix, testDir, Seq(), harness).!
    cppToExe(prefix, testDir).!
    assert(executeExpectingSuccess(prefix, testDir))
  }
}

trait FirrtlMatchers {
  // Replace all whitespace with a single space and remove leading and
  //   trailing whitespace
  // Note this is intended for single-line strings, no newlines
  def normalized(s: String): String = {
    require(!s.contains("\n"))
    s.replaceAll("\\s+", " ").trim
  }
}

class FirrtlPropSpec extends PropSpec with PropertyChecks with FirrtlRunners with LazyLogging

class FirrtlFlatSpec extends FlatSpec with Matchers with FirrtlRunners with FirrtlMatchers with LazyLogging

