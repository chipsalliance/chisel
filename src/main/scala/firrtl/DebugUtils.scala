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

import Utils._

private object DebugUtils {

  implicit class DebugASTUtils(ast: AST) {
    // Is this actually any use?
    /*
    def preOrderTraversal(f: AST => Unit): Unit = {
      f(ast)
      ast match {
        case a: Block => a.stmts.foreach(_.preOrderTraversal(f))
        case a: Assert => a.pred.preOrderTraversal(f)
        case a: When => {
          a.pred.preOrderTraversal(f)
          a.conseq.preOrderTraversal(f)
          a.alt.preOrderTraversal(f)
        }
        case a: BulkConnect => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: Connect => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: OnReset => {
          a.lhs.preOrderTraversal(f)
          a.rhs.preOrderTraversal(f)
        }
        case a: DefAccessor => {
          a.dir.preOrderTraversal(f)
          a.source.preOrderTraversal(f)
          a.index.preOrderTraversal(f)
        }
        case a: DefPoison => a.tpe.preOrderTraversal(f)
        case a: DefNode => a.value.preOrderTraversal(f)
        case a: DefInst => a.module.preOrderTraversal(f)
        case a: DefMemory => {
          a.tpe.preOrderTraversal(f)
          a.clock.preOrderTraversal(f)
        }
        case a: DefReg => {
          a.tpe.preOrderTraversal(f)
          a.clock.preOrderTraversal(f)
          a.reset.preOrderTraversal(f)
        }
        case a: DefWire => a.tpe.preOrderTraversal(f)
        case a: Field => {
          a.dir.preOrderTraversal(f)
          a.tpe.preOrderTraversal(f)
        }
        case a: VectorType => a.tpe.preOrderTraversal(f)
        case a: BundleType => a.fields.foreach(_.preOrderTraversal(f))
        case a: Port => {
          a.dir.preOrderTraversal(f)
          a.tpe.preOrderTraversal(f)
        }
        case a: Module => {
          a.ports.foreach(_.preOrderTraversal(f))
          a.stmt.preOrderTraversal(f)
        }
        case a: Circuit => a.modules.foreach(_.preOrderTraversal(f))
        //case _ => throw new Exception(s"Unsupported FIRRTL node ${ast.getClass.getSimpleName}!")
        case _ =>
      }
    } */
  }
}
