/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
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

package Chisel
import Literal._

object Enum {
  private def makeLit[T <: Bits](nodeType: T, i: Int, n: Int) =
    nodeType.cloneType.makeLit(i, BigInt(n-1).bitLength)

  /** create n enum values of given type */
  def apply[T <: Bits](nodeType: T, n: Int): List[T] = Range(0, n).map(x => makeLit(nodeType, x, n)).toList

  /** create enum values of given type and names */
  def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] = (l.toList zip (Range(0, l.length).map(x => makeLit(nodeType, x, l.length)))).toMap

  /** create enum values of given type and names */
  def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = (l zip (Range(0, l.length).map(x => makeLit(nodeType, x, l.length)))).toMap

}
