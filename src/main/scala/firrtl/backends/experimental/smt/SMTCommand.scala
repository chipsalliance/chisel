// SPDX-License-Identifier: Apache-2.0
// Author: Kevin Laeufer <laeufer@cs.berkeley.edu>

package firrtl.backends.experimental.smt

private sealed trait SMTCommand
private case class Comment(msg: String) extends SMTCommand
private case class SetLogic(logic: String) extends SMTCommand
private case class DefineFunction(name: String, args: Seq[SMTFunctionArg], e: SMTExpr) extends SMTCommand
private case class DeclareFunction(sym: SMTSymbol, args: Seq[SMTFunctionArg]) extends SMTCommand
private case class DeclareUninterpretedSort(name: String) extends SMTCommand
private case class DeclareUninterpretedSymbol(name: String, tpe: String) extends SMTCommand
