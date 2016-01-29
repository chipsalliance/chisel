
/* TODO
 * - Add support for comments (that being said, current Scopers regex should ignore commented lines)
 * - Add better error messages for illformed FIRRTL 
 * - Add support for files that do not have a circuit (like a module by itself in a file)
 * - Improve performance? Replace regex?
 * - Add proper commnad-line arguments?
 * - Wrap in Reader subclass. This would have less memory footprint than creating a large string
 */

package firrtl

import scala.io.Source
import scala.collection.mutable.Stack
import scala.collection.mutable.StringBuilder
import java.io._


object Translator
{

  def addBrackets(inputIt: Iterator[String]): StringBuilder = {
    def countSpaces(s: String): Int = s.prefixLength(_ == ' ')
    def stripComments(s: String): String = s takeWhile (!";".contains(_))

    val Scopers = """\s*(circuit|module|when|else|mem)(.*)""".r

    // Function start
    val it = inputIt.zipWithIndex 
    var ret = new StringBuilder()

    if( !it.hasNext ) throw new Exception("Empty file!")
    
    // Find circuit before starting scope checks
    var line = it.next 
    while ( it.hasNext && !line._1.contains("circuit") ) {  
      ret ++= line._1 + "\n"
      line = it.next
    }
    ret ++= line._1 + " { \n"
    if( !it.hasNext ) throw new Exception("No circuit in file!")


    val scope = Stack[Int]()
    val lowestScope = countSpaces(line._1)
    scope.push(lowestScope) 
    var newScope = true // indicates if increasing scope spacing is legal on next line

    while( it.hasNext ) {
      it.next match { case (lineText, lineNum) =>
        val text = stripComments(lineText)
        val spaces = countSpaces(text)

        val l = if (text.length > spaces ) { // Check that line has text in it
          if (newScope) { 
            if( spaces <= scope.top ) scope.push(spaces+2) // Hack for one-line scopes
            else scope.push(spaces) 
          }

          // Check if change in current scope
          if( spaces < scope.top ) {
            while( spaces < scope.top ) {
              // Close scopes (adding brackets as we go)
              scope.pop() 
              ret.deleteCharAt(ret.lastIndexOf("\n")) // Put on previous line
              ret ++= " }\n"
            }
            if( spaces != scope.top ) 
              throw new Exception("Spacing does not match scope on line : " + lineNum + " : " + scope.top)
          }
          else if( spaces > scope.top ) 
            throw new Exception("Invalid increase in scope on line " + lineNum)
          
          // Now match on legal scope increasers
          text match {
            case Scopers(keyword, _* ) => {
              newScope = true
              //text + " { "
              text.replaceFirst(":", ": {")
            }
            case _ => { 
              newScope = false
              text
            }
          }
        } // if( text.length > spaces ) 
        else {
          text // empty lines
        }

        ret ++= l + "\n"
      } // it.next match
    } // while( it.hasNext )
    
    // Print any closing braces
    while( scope.top > lowestScope ) {
      scope.pop()
      ret.deleteCharAt(ret.lastIndexOf("\n")) // Put on previous line
      ret ++= " }\n"
    }

    ret
  }

  def main(args: Array[String]) {
   
    try {
      val translation = addBrackets(Source.fromFile(args(0)).getLines)

      val writer = new PrintWriter(new File(args(1)))
      writer.write(translation.result)
      writer.close()
    } catch {
      case e: Exception => {
        throw new Exception("USAGE: Translator <input file> <output file>\n" + e)
      }
    }
  }

}
