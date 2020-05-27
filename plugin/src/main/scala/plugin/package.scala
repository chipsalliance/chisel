// See LICENSE for license details.

import java.io.FileWriter

package object plugin {
  /** Write to a file, useful for debugging things at compilation time when you have a million warnings
    * @param filename
    * @param body
    */
  def write(filename: String, body: String): Unit = {
    val fw = new FileWriter(s"$filename.txt")
    fw.write(body)
    fw.close()
  }

  /** Write source code AST to a file
    * @param filename
    * @param body
    */
  def writeAST(filename: String, body: String): Unit = {
    write(filename, stringifyAST(body))
  }

  /** Indents an AST to reflect the AST structure
    * Useful for diffing between source trees
    * @param ast
    * @return indented ast
    */
  def stringifyAST(ast: String): String = {
    var ntabs = 0
    val buf = new StringBuilder
    ast.zipWithIndex.foreach { case (c, idx) =>
      c match {
        case ' ' =>
        case '(' =>
          ntabs += 1
          buf ++= "(\n" + "| " * ntabs
        case ')' =>
          ntabs -= 1
          buf ++= "\n" + "| " * ntabs + ")"
        case ','=> buf ++= ",\n" + "| " * ntabs
        case  c if idx > 0 && ast(idx-1)==')' =>
          buf ++= "\n" + "| " * ntabs + c
        case c => buf += c
      }
    }
    buf.toString
  }
}
