
package firrtl.passes

import com.typesafe.scalalogging.LazyLogging
import java.nio.file.{Paths, Files}

// For calling Stanza 
import scala.sys.process._
import scala.io.Source

import firrtl._
import firrtl.Utils._
import firrtl.PrimOps._

trait Pass extends LazyLogging {
  def name: String
  def run(c: Circuit): Circuit
}

// Trait for migration, trap to Stanza implementation for passes not yet implemented in Scala
trait StanzaPass extends LazyLogging {
  def stanzaPass(c: Circuit, n: String): Circuit = {
    // For migration from Stanza, handle unimplemented Passes
    logger.debug(s"Pass ${n} is not yet implemented in Scala")
    val stanzaPasses = Seq("resolve", n) 
    val toStanza = Files.createTempFile(Paths.get(""), n, ".fir")
    val fromStanza = Files.createTempFile(Paths.get(""), n, ".fir")
    Files.write(toStanza, c.serialize.getBytes)

    val cmd = Seq("firrtl-stanza", "-i", toStanza.toString, "-o", fromStanza.toString, "-b", "firrtl") ++ 
              stanzaPasses.flatMap(x=>Seq("-x", x))
    logger.debug(cmd.mkString(" "))
    val ret = cmd.!
    //println(ret)
    val newC = Parser.parse(fromStanza.toString, Source.fromFile(fromStanza.toString).getLines)
    Files.delete(toStanza)
    Files.delete(fromStanza)
    newC
  }
}

object PassUtils extends LazyLogging {
  val listOfPasses: Seq[Pass] = Seq(ToWorkingIR)
  lazy val mapNameToPass: Map[String, Pass] = listOfPasses.map(p => p.name -> p).toMap

  def executePasses(c: Circuit, passes: Seq[Pass]): Circuit = {
    if (passes.isEmpty) c
    else executePasses(passes.head.run(c), passes.tail)
  }
}

// These should be distributed into separate files
object CheckHighForm extends Pass with StanzaPass {
  def name = "High Form Check"
  def run (c:Circuit): Circuit = stanzaPass(c, "high-form-check")
}

object ToWorkingIR extends Pass with StanzaPass {
  def name = "Working IR"
  def run (c:Circuit): Circuit = stanzaPass(c, "to-working-ir")
}

object Resolve extends Pass with StanzaPass {
  def name = "Resolve"
  def run (c:Circuit): Circuit = stanzaPass(c, "resolve")
}

object ResolveKinds extends Pass with StanzaPass {
  def name = "Resolve Kinds"
  def run (c:Circuit): Circuit = stanzaPass(c, "resolve-kinds")
}

object InferTypes extends Pass with StanzaPass {
  def name = "Infer Types"
  def run (c:Circuit): Circuit = stanzaPass(c, "infer-types")
}

object CheckTypes extends Pass with StanzaPass {
  def name = "Check Types"
  def run (c:Circuit): Circuit = stanzaPass(c, "check-types")
}

object ResolveGenders extends Pass with StanzaPass {
  def name = "Resolve Genders"
  def run (c:Circuit): Circuit = stanzaPass(c, "resolve-genders")
}

object CheckGenders extends Pass with StanzaPass {
  def name = "Check Genders"
  def run (c:Circuit): Circuit = stanzaPass(c, "check-genders")
}

object InferWidths extends Pass with StanzaPass {
  def name = "Infer Widths"
  def run (c:Circuit): Circuit = stanzaPass(c, "infer-widths")
}

object CheckWidths extends Pass with StanzaPass {
  def name = "Width Check"
  def run (c:Circuit): Circuit = stanzaPass(c, "width-check")
}

object PullMuxes extends Pass with StanzaPass {
  def name = "Pull Muxes"
  def run (c:Circuit): Circuit = stanzaPass(c, "pull-muxes")
}

object ExpandConnects extends Pass with StanzaPass {
  def name = "Expand Connects"
  def run (c:Circuit): Circuit = stanzaPass(c, "expand-connects")
}

object RemoveAccesses extends Pass with StanzaPass {
  def name = "Remove Accesses"
  def run (c:Circuit): Circuit = stanzaPass(c, "remove-accesses")
}

object ExpandWhens extends Pass with StanzaPass {
  def name = "Expand Whens"
  def run (c:Circuit): Circuit = stanzaPass(c, "expand-whens")
}

object CheckInitialization extends Pass with StanzaPass {
  def name = "Check Initialization"
  def run (c:Circuit): Circuit = stanzaPass(c, "check-init")
}

object ConstProp extends Pass with StanzaPass {
  def name = "Constant Propogation"
  def run (c:Circuit): Circuit = stanzaPass(c, "const-prop")
}

object VerilogWrap extends Pass with StanzaPass {
  def name = "Verilog Wrap"
  def run (c:Circuit): Circuit = stanzaPass(c, "verilog-wrap")
}

object SplitExp extends Pass with StanzaPass {
  def name = "Split Expressions"
  def run (c:Circuit): Circuit = stanzaPass(c, "split-expressions")
}

object VerilogRename extends Pass with StanzaPass {
  def name = "Verilog Rename"
  def run (c:Circuit): Circuit = stanzaPass(c, "verilog-rename")
}

object LowerTypes extends Pass with StanzaPass {
  def name = "Lower Types"
  def run (c:Circuit): Circuit = stanzaPass(c, "lower-types")
}

