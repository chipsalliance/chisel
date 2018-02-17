// See LICENSE for license details.

package firrtl.passes
package memlib
import net.jcazevedo.moultingyaml._
import java.io.{CharArrayWriter, File, PrintWriter}

import firrtl.Utils.error

object CustomYAMLProtocol extends DefaultYamlProtocol {
  // bottom depends on top
  implicit val _pin = yamlFormat1(Pin)
  implicit val _source = yamlFormat2(Source)
  implicit val _top = yamlFormat1(Top)
  implicit val _configs = yamlFormat3(Config)
}

case class Pin(name: String)
case class Source(name: String, module: String)
case class Top(name: String)
case class Config(pin: Pin, source: Source, top: Top)


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
    else sys.error("Yaml file doesn't exist!")
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
