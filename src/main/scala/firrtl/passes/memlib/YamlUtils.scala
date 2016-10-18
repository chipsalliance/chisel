package firrtl.passes
package memlib
import net.jcazevedo.moultingyaml._
import java.io.{File, CharArrayWriter, PrintWriter}

object CustomYAMLProtocol extends DefaultYamlProtocol {
  // bottom depends on top
}

class YamlFileReader(file: String) {
  import CustomYAMLProtocol._
  def parse[A](implicit reader: YamlReader[A]) : Seq[A] = {
    if (new File(file).exists) {
      val yamlString = scala.io.Source.fromFile(file).getLines.mkString("\n")
      yamlString.parseYamls flatMap (x =>
        try Some(reader read x)
        catch { case e: Exception => None }
      )
    }
    else error("Yaml file doesn't exist!")
  }
}

class YamlFileWriter(file: String) {
  import CustomYAMLProtocol._
  val outputBuffer = new CharArrayWriter
  val separator = "--- \n"
  def append(in: YamlValue) {
    outputBuffer append s"$separator${in.prettyPrint}"
  }
  def dump() {
    val outputFile = new PrintWriter(file)
    outputFile write outputBuffer.toString
    outputFile.close()
  }
}
