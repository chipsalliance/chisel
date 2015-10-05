import java.io._
import firrtl._
import firrtl.Utils._

object Example
{
  // Example use of Scala FIRRTL parser and serialization
  def main(args: Array[String])
  {
    val inputFile = args(0)

    // Parse file
    val ast = firrtl.Parser.parse(inputFile)

    val writer = new PrintWriter(new File(args(1)))
    writer.write(ast.serialize) // serialize returns String
    writer.close()
  }
}
