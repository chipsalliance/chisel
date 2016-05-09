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

object DeadCodeElimination extends Pass {
  def name = "Dead Code Elimination"

  private def dceOnce(s: Stmt): (Stmt, Long) = {
    val referenced = collection.mutable.HashSet[String]()
    var nEliminated = 0L

    def checkExpressionUse(e: Expression): Expression = {
      e match {
        case WRef(name, _, _, _) => referenced += name
        case _ => e map checkExpressionUse
      }
      e
    }

    def checkUse(s: Stmt): Stmt = s map checkUse map checkExpressionUse

    def maybeEliminate(x: Stmt, name: String) =
      if (referenced(name)) x
      else {
        nEliminated += 1
        Empty()
      }

    def removeUnused(s: Stmt): Stmt = s match {
      case x: DefRegister => maybeEliminate(x, x.name)
      case x: DefWire => maybeEliminate(x, x.name)
      case x: DefNode => maybeEliminate(x, x.name)
      case x => s map removeUnused
    }

    checkUse(s)
    (removeUnused(s), nEliminated)
  }

  @tailrec
  private def dce(s: Stmt): Stmt = {
    val (res, n) = dceOnce(s)
    if (n > 0) dce(res) else res
  }

  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => Module(m.info, m.name, m.ports, dce(m.body))
    }
    Circuit(c.info, modulesx, c.main)
  }
}
