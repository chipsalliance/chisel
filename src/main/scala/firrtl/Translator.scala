/*
Copyright (c) 2014 - 2016 The Regents of the University of
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

    val scopers = """(circuit|module|when|else|mem|with)"""
    val MultiLineScope = ("""(.*""" + scopers + """)(.*:\s*)""").r
    val OneLineScope   = ("""(.*""" + scopers + """\s*:\s*)\((.*)\)\s*""").r

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
            case OneLineScope(head, keyword, body) => {
              newScope = false
              head + "{" + body + "}"
            }
            case MultiLineScope(head, keyword, tail) => {
              newScope = true
              //text.replaceFirst(":", ": {")
              text + " { "
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
