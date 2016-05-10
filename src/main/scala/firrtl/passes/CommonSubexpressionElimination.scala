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

package firrtl.passes

import firrtl._
import firrtl.Utils._
import firrtl.Mappers._

import annotation.tailrec

object CommonSubexpressionElimination extends Pass {
  def name = "Common Subexpression Elimination"

  private def cseOnce(s: Statement): (Statement, Long) = {
    var nEliminated = 0L
    val expressions = collection.mutable.HashMap[MemoizedHash[Expression], String]()
    val nodes = collection.mutable.HashMap[String, Expression]()

    def recordNodes(s: Statement): Statement = s match {
      case x: DefNode =>
        nodes(x.name) = x.value
        expressions.getOrElseUpdate(x.value, x.name)
        x
      case _ => s map recordNodes
    }

    def eliminateNodeRef(e: Expression): Expression = e match {
      case WRef(name, tpe, kind, gender) => nodes get name match {
        case Some(expression) => expressions get expression match {
          case Some(cseName) if cseName != name =>
            nEliminated += 1
            WRef(cseName, tpe, kind, gender)
          case _ => e
        }
        case _ => e
      }
      case _ => e map eliminateNodeRef
    }

    def eliminateNodeRefs(s: Statement): Statement = s map eliminateNodeRefs map eliminateNodeRef

    recordNodes(s)
    (eliminateNodeRefs(s), nEliminated)
  }

  @tailrec
  private def cse(s: Statement): Statement = {
    val (res, n) = cseOnce(s)
    if (n > 0) cse(res) else res
  }

  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => Module(m.info, m.name, m.ports, cse(m.body))
    }
    Circuit(c.info, modulesx, c.main)
  }
}
