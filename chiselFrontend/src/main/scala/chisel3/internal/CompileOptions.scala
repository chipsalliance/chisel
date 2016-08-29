// See LICENSE for license details.

package chisel3.internal

trait CompileOptions {
  // Should Bundle connections require a strict match of fields.
  // If true and the same fields aren't present in both source and sink, a MissingFieldException,
  // MissingLeftFieldException, or MissingRightFieldException will be thrown.
  val connectFieldsMustMatch: Boolean
  val declaredTypeMustBeUnbound: Boolean
  val requireIOWrap: Boolean
  val dontTryConnectionsSwapped: Boolean
  val dontAssumeDirectionality: Boolean
}

trait ExplicitCompileOptions extends CompileOptions

///** Initialize compilation options from a string map.
//  *
//  * @param optionsMap the map from "option" to "value"
//  */
//class CompileOptions(optionsMap: Map[String, String]) {
//  // The default for settings related to "strictness".
//  val strictDefault: String = optionsMap.getOrElse("strict", "false")
//  // Should Bundle connections require a strict match of fields.
//  // If true and the same fields aren't present in both source and sink, a MissingFieldException,
//  // MissingLeftFieldException, or MissingRightFieldException will be thrown.
//  val connectFieldsMustMatch: Boolean = optionsMap.getOrElse("connectFieldsMustMatch", strictDefault).toBoolean
//  val declaredTypeMustBeUnbound: Boolean = optionsMap.getOrElse("declaredTypeMustBeUnbound", strictDefault).toBoolean
//  val requireIOWrap: Boolean = optionsMap.getOrElse("requireIOWrap", strictDefault).toBoolean
//  val dontTryConnectionsSwapped: Boolean = optionsMap.getOrElse("dontTryConnectionsSwapped", strictDefault).toBoolean
//  val dontAssumeDirectionality: Boolean = optionsMap.getOrElse("dontAssumeDirectionality", strictDefault).toBoolean
//}
