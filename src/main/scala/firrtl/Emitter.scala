
package firrtl

import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Paths, Files}
import java.io.Writer
import java.io.Reader

import scala.sys.process._
import scala.io.Source

import Utils._
import firrtl.passes._

trait Emitter extends LazyLogging {
  def run(c: Circuit, w: Writer)
}

object FIRRTLEmitter extends Emitter {
  def run(c: Circuit, w: Writer) = w.write(c.serialize)
}

object VerilogEmitter extends Emitter {
  // Currently just trap into Stanza
  def run(c: Circuit, w: Writer) 
  {
    logger.debug(s"Verilog Emitter is not yet implemented in Scala")
    val toStanza = Files.createTempFile(Paths.get(""), "verilog", ".fir")
    val fromStanza = Files.createTempFile(Paths.get(""), "verilog", ".fir")
    Files.write(toStanza, c.serialize.getBytes)

    val cmd = Seq("firrtl-stanza", "-i", toStanza.toString, "-o", fromStanza.toString, "-b", "verilog")
    logger.debug(cmd.mkString(" "))
    val ret = cmd.!
    // Copy from Stanza output to user requested outputFile (we can't get filename from Writer)
    Source.fromFile(fromStanza.toString) foreach { w.write(_) }

    Files.delete(toStanza)
    Files.delete(fromStanza)
  }
}
