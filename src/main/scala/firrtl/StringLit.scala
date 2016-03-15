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

package firrtl

import java.nio.charset.StandardCharsets.UTF_8
import scala.annotation.tailrec

// Default implementations are for valid FIRRTL Strings
trait StringLitHandler {

  // Apply correct escaping for strings in a given language
  def escape(lit: StringLit): String = {
    @tailrec
    def escape(in: Array[Byte], str: String): String = {
      if (in.isEmpty) str
      else {
        val s = in.head match {
          case 0x5c => "\\\\"
          case 0x0a => "\\n"
          case 0x09 => "\\t"
          case 0x22 => "\\\""
          case c => 
            if (c >= 0x20 && c <= 0x7e) new String(Array(c), UTF_8)
            else throw new FIRRTLException(s"Unsupported byte in StringLit: $c")
        }
        escape(in.tail, str + s)
      }
    }
    escape(lit.array, "")
  }

  // Turn raw parsed strings into String Literals
  def unescape(raw: String): StringLit = {
    @tailrec
    def unescape(in: Array[Byte], out: Array[Byte]): Array[Byte] = {
      if (in.isEmpty) out
      else {
        val c = in.head
        if ((c < 0x20) || (c > 0x7e)) { // Range from ' ' to '~'
          val msg = "Unsupported unescaped UTF-8 Byte 0x%02x! ".format(c) + 
                    "Valid range [0x20-0x7e]"
          throw new InvalidStringLitException(msg)
        }
        // Handle escaped characters
        if (c == 0x5c) { // backslash
          val (n: Int, bytes: Array[Byte]) = in(1) match {
            case 0x5c => (2, Array[Byte](0x5c)) // backslash
            case 0x6e => (2, Array[Byte](0x0a)) // newline
            case 0x74 => (2, Array[Byte](0x09)) // tab
            case 0x27 => (2, Array[Byte](0x27)) // single quote
            case 0x22 => (2, Array[Byte](0x22)) // double quote
            // TODO Finalize supported escapes, implement hex2Bytes
            //case 0x78 => (4, hex2Bytes(in.slice(2, 3)))) // hex escape
            //case 0x75 => (6, hex2Bytes(in.slice(2, 5))) // unicode excape 
            case e => { // error
              val msg = s"Invalid escape character ${e.toChar}! " + 
                         "Valid characters [nt'\"\\]"
              throw new InvalidEscapeCharException(msg)
            }
          }
          unescape(in.drop(n), out ++ bytes) // consume n
        } else {
          unescape(in.tail, out :+ in.head)
        }
      }
    }

    val byteArray = raw.getBytes(java.nio.charset.StandardCharsets.UTF_8)
    StringLit(unescape(byteArray, Array()))
  }

  // Translate from FIRRTL formatting to the specific language's formatting
  def format(lit: StringLit): StringLit = ???
  // Translate from the specific language's formatting to FIRRTL formatting
  def unformat(lit: StringLit): StringLit = ???
}

object FIRRTLStringLitHandler extends StringLitHandler

object VerilogStringLitHandler extends StringLitHandler {
  // Change FIRRTL format string to Verilog format
  override def format(lit: StringLit): StringLit = {
    @tailrec
    def format(in: Array[Byte], out: Array[Byte], percent: Boolean): Array[Byte] = {
      if (in.isEmpty) out
      else {
        if (percent && in.head == 'x') format(in.tail, out :+ 'h'.toByte, false)
        else format(in.tail, out :+ in.head, in.head == '%' && !percent)
      }
    }
    StringLit(format(lit.array, Array(), false))
  }
}

