// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

sealed trait SMTCommand
case class Comment(msg: String) extends SMTCommand
case class SetLogic(logic: String) extends SMTCommand
case class DefineFunction(name: String, args: Seq[SMTFunctionArg], e: SMTExpr) extends SMTCommand
case class DeclareFunction(sym: SMTSymbol, args: Seq[SMTFunctionArg]) extends SMTCommand
case class DeclareUninterpretedSort(name: String) extends SMTCommand
case class DeclareUninterpretedSymbol(name: String, tpe: String) extends SMTCommand
