package firrtlTests.execution

import java.io.File

import firrtl._
import firrtl.ir._
import firrtlTests._

import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlStage}
import firrtl.options.TargetDirAnnotation

/**
  * Mixing in this trait causes a SimpleExecutionTest to be run in Verilog simulation.
  */
trait VerilogExecution extends TestExecution {
  this: SimpleExecutionTest =>
  def runEmittedDUT(c: Circuit, testDir: File): Unit = {
    // Run FIRRTL, emit Verilog file
    val cAnno = FirrtlCircuitAnnotation(c)
    val tdAnno = TargetDirAnnotation(testDir.getAbsolutePath)
    (new FirrtlStage).run(AnnotationSeq(Seq(cAnno, tdAnno)))

    // Copy harness resource to test directory
    val harness = new File(testDir, s"top.cpp")
    copyResourceToFile(cppHarnessResourceName, harness)

    // Make and run Verilog simulation
    verilogToCpp(c.main, testDir, Nil, harness) #&&
    cppToExe(c.main, testDir) ! loggingProcessLogger
    assert(executeExpectingSuccess(c.main, testDir))
  }
}
