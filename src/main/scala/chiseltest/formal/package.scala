// SPDX-License-Identifier: Apache-2.0

package chiseltest

import chisel3._
import circt.stage.ChiselStage
import scala.annotation.compileTimeOnly
import scala.sys.process._
import scala.collection.mutable
import scala.reflect.io._
/**
 * Formal compatibility API placeholders.
 *
 * Formal verification is currently unsupported in this compatibility layer.
 * Any usage should fail at compile time to avoid vacuously passing tests.
 */
package object formal {

    /** Annotation placeholder for source compatibility only. */
  case class BoundedCheck(depth: Int)

  /**
    * Bounded Model Checking Backends supported by `Formal`.
    * Currently only BTORMC is supported.
    */
  abstract class FormalBackendAnnotation(
    val name: String, 
    val kmaxFlag: String,
    val UnsatOutput: String, // string identifying the start of an unsat result
    val SatOutput: String   // string identifying the start of a sat result
  )

  case object BTORMCBackend 
    extends FormalBackendAnnotation("btormc", "-kmax", "", "sat")

  /**
    * counterexample object for a BMC run
    */
  abstract class Witness {
    def serialize : String
  }

  /**
    * Most basic type of witness: simply dumps the ouput from the BMC tool
    */
  case class BasicWitness(body: String) extends Witness {
    override def serialize: String = body
  }

  /**
    * Result of a Bounded Model Checking run
    */
  abstract class BMCResult(
    val message: String,
    val _witness: Option[Witness]
  )

  case object Unsat extends BMCResult(
    "All checks passed: no counterexamples were found!",
    None
  )

  case class Sat[W <: Witness](witness: W) extends BMCResult(
    s"Some checks failed: counter example was found!\n${witness.serialize}",
    Some(witness)
  )

  trait Formal {
    // Converts the design to a formal model using the btor2 backend, then
    // runs the output through some model-checker, e.g. btormc
    def verify[T <: Module](dut: => T, annotations: Seq[Any]): BMCResult = {
      // start by running the btor2 backend
      val btor2DUT: String = ChiselStage.emitBtor2(
        dut, 
        firtoolOpts = Array("-default-layer-specialization=enable")
      )

      // Store the btor2 result to a file
      val workdir: String = sys.props("user.dir")
      val btor2file = File.makeTemp(suffix = ".btor2")
      btor2file writeAll btor2DUT

      // Store absolute path
      val fileAbsPath: String = btor2file.path
      
      // Filter out non formal annotations and keep the first one
      val backanno = annotations.flatMap {
        case fba: FormalBackendAnnotation => Some(fba)
        case _ => None
      }.head

      // Extract the BMC depth
      val kMax = annotations.flatMap {
        case BoundedCheck(depth) => Some(depth)
        case _ => None
      }.head

      // Check if the given backend is in the user's path 
      val executablePath: String = {
        val output = mutable.ArrayBuffer.empty[String]
        val exitCode: Int = List("which", backanno.name).!(ProcessLogger(output += _))
        if (exitCode != 0) {
          throw new Exception(s"${backanno.name} not found on the PATH!\n${output.mkString("\n")}")
        }
        output.head.trim
      }

      /* verify the model using the given backend */

      // Setup our model checker invocation
      val command = List(executablePath, backanno.kmaxFlag, kMax.toString, fileAbsPath) 

      // Run invocation and capture output
      val bmcRes = {
        val output = mutable.ArrayBuffer.empty[String]
        val exitCode: Int = command.!(ProcessLogger(output += _))
        if (exitCode != 0) {
          throw new Exception(s"BMC invocation failed with output:\n${output.mkString("\n")}")
        }

        // output can be empty in some unsat cases
        if (!output.isEmpty) 
          output.foldLeft("\n")((acc, o) => acc + (o + "\n")).trim 
        else 
          ""
      }

      // most basic check on result
      val result = bmcRes match {
        case backanno.UnsatOutput => Unsat  // Note this is only true for BMC
        case s => Sat(BasicWitness(s))
      }

      // print out result
      println(result.message)

      // feed result to user as well
      result
    }
  }

  @compileTimeOnly("chiseltest.formal.past is unsupported in this compatibility layer")
  def past[T <: Data](x: T, delay: Int = 1): T =
    throw new UnsupportedOperationException("chiseltest.formal.past is unsupported")

  @compileTimeOnly("chiseltest.formal.past is unsupported in this compatibility layer")
  def past[T <: Data](x: T): T =
    throw new UnsupportedOperationException("chiseltest.formal.past is unsupported")
}
