/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
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

package Chisel.testers
import Chisel._



object CompilationUtilitiess {
  import scala.util.Properties.envOrElse
  import sys.process.stringSeqToProcess
  protected val CC = envOrElse("CC", "g++" )
  protected val CXX = envOrElse("CXX", "g++" )
  protected val CCFLAGS = envOrElse("CCFLAGS", "")
  protected val CXXFLAGS = envOrElse("CXXFLAGS", "")
  protected val CPPFLAGS = envOrElse("CPPFLAGS", "")
  protected val LDFLAGS = envOrElse("LDFLAGS", "")
  protected val chiselENV = envOrElse("CHISEL", "")

  def run(cmd: String) = {
    val bashCmd = Seq("bash", "-c", cmd)
    val c = bashCmd.!
    ChiselError.info(cmd + " RET " + c)
    c == 0
  }

  def cc(dir: String, name: String, flags: String = "", isCC: Boolean = false) {
    val compiler = if (isCC) CC else CXX
    val cmd = List(compiler, "-c", "-o", dir + name + ".o", flags, dir + name + ".cpp").mkString(" ")
    if (!run(cmd)) throw new Exception("failed to compile " + name + ".cpp")
  }

  def link(dir: String, target: String, objects: Seq[String], isCC: Boolean = false, isLib: Boolean = false) {
    val compiler = if (isCC) CC else CXX
    val shared = if (isLib) "-shared" else ""
    val ac = (List(compiler, LDFLAGS, shared, "-o", dir + target) ++ (objects map (dir + _))).mkString(" ")
    if (!run(ac)) throw new Exception("failed to link " + objects.mkString(", "))
  }
}

abstract class Backend
class FloBackend extends Backend 
class VerilogBackend extends Backend {
  def genHarness(c: Module, name: String) { }
}

object TesterDriver {
  val isVCD = false
  val targetDir = "."
  val backend: Backend  = new VerilogBackend
  val name = "test"
  val circuit = Circuit(Seq(Component("top",Seq(Port(null,null)),Nil)),"main")
  val testCommand: Option[String] = None


  // Setting this to TRUE will initialize the tester's RNG with the
  // seed below.
  //      case "--testerSeed" => {
  //        testerSeedValid = true
  //        testerSeed = args(i+1).toLong }
  var testerSeedValid = false
  var testerSeed = System.currentTimeMillis()

  // Setting this to TRUE will case the test harness to print its
  // standard input stream to a file.
  var dumpTestInput = false

  private def test[T <: Module](mod: T, ftester: T => Tester[T]): Unit = {
    ftester(mod).finish
  }

}
