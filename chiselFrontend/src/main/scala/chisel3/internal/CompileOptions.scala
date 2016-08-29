// See LICENSE for license details.

package chisel3.internal

/** Initialize compilation options from a string map.
  *
  * @param optionsMap the map from "option" to "value"
  */
class CompileOptions(optionsMap: Map[String, String]) {
  // The default for settings related to "strictness".
  val strictDefault: String = optionsMap.getOrElse("strict", "false")
  val looseDefault: String = (!(strictDefault.toBoolean)).toString
  // Should Bundle connections require a strict match of fields.
  // If true and the same fields aren't present in both source and sink, a MissingFieldException,
  // MissingLeftFieldException, or MissingRightFieldException will be thrown.
  val connectFieldsMustMatch: Boolean = optionsMap.getOrElse("connectFieldsMustMatch", strictDefault).toBoolean
  val regTypeMustBeUnbound: Boolean = optionsMap.getOrElse("regTypeMustBeUnbound", strictDefault).toBoolean
  val declaredTypeMustBeUnbound: Boolean = optionsMap.getOrElse("declaredTypeMustBeUnbound", strictDefault).toBoolean
  val requireIOWrap: Boolean = optionsMap.getOrElse("requireIOWrap", strictDefault).toBoolean
  val dontTryConnectionsSwapped: Boolean = optionsMap.getOrElse("dontTryConnectionsSwapped", strictDefault).toBoolean
  val dontAssumeDirectionality: Boolean = optionsMap.getOrElse("dontAssumeDirectionality", strictDefault).toBoolean
}
