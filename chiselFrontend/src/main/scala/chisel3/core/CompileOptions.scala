// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

trait CompileOptions {
  // Should Record connections require a strict match of fields.
  // If true and the same fields aren't present in both source and sink, a MissingFieldException,
  // MissingLeftFieldException, or MissingRightFieldException will be thrown.
  val connectFieldsMustMatch: Boolean
  // When creating an object that takes a type argument, the argument must be unbound (a pure type).
  val declaredTypeMustBeUnbound: Boolean
  // Module IOs should be wrapped in an IO() to define their bindings before the reset of the module is defined.
  val requireIOWrap: Boolean
  // If a connection operator fails, don't try the connection with the operands (source and sink) reversed.
  val dontTryConnectionsSwapped: Boolean
  // If connection directionality is not explicit, do not use heuristics to attempt to determine it.
  val dontAssumeDirectionality: Boolean
  // Issue a deprecation warning if Data.{flip, asInput,asOutput} is used
  // instead of Flipped, Input, or Output.
  val deprecateOldDirectionMethods: Boolean
  // Check that referenced Data have actually been declared.
  val checkSynthesizable: Boolean
}

object CompileOptions {
  // Provides a low priority Strict default. Can be overridden by importing the NotStrict option.
  implicit def materialize: CompileOptions = chisel3.core.ExplicitCompileOptions.Strict
}

object ExplicitCompileOptions {
  // Collection of "not strict" connection compile options.
  // These provide compatibility with existing code.
  // import chisel3.core.ExplicitCompileOptions.NotStrict
  implicit object NotStrict extends CompileOptions {
    val connectFieldsMustMatch = false
    val declaredTypeMustBeUnbound = false
    val requireIOWrap = false
    val dontTryConnectionsSwapped = false
    val dontAssumeDirectionality = false
    val deprecateOldDirectionMethods = false
    val checkSynthesizable = false
  }

  // Collection of "strict" connection compile options, preferred for new code.
  // import chisel3.core.ExplicitCompileOptions.Strict
  implicit object Strict extends CompileOptions {
    val connectFieldsMustMatch = true
    val declaredTypeMustBeUnbound = true
    val requireIOWrap = true
    val dontTryConnectionsSwapped = true
    val dontAssumeDirectionality = true
    val deprecateOldDirectionMethods = true
    val checkSynthesizable = true
  }
}
