package chisel3.experimental

import chisel3.Record

/** Wrapper object for a Record alias name. Primarily intended to provide an invocation point for source line locators, but
  * also contains pertinent information to generating FIRRTL alias statements.
  *
  * @param id The desired name to generate an alias statement
  * @param strippedSuffix In the case of forced coersion by [[Input]] or [[Output]], the string to append to the end of the
  *        alias name. Takes the default value of `"_stripped"`
  */
case class BundleAlias private[chisel3] (info: SourceInfo, id: String, strippedSuffix: String = "_stripped")

object BundleAlias {
  def apply(id: String)(implicit info:  SourceInfo): BundleAlias = BundleAlias(info, id)
  def apply(id: String, strippedSuffix: String)(implicit info: SourceInfo): BundleAlias =
    BundleAlias(info, id, strippedSuffix)
}

trait HasTypeAlias {
  this: Record =>

  /** An optional FIRRTL type alias name to give to this [[Record]]. If overrided with a `Some(...)`, for instance
    * `Some(BundleAlias("UserBundle"))`, this causes emission of circuit-level FIRRTL statements that declare that name for
    * this bundle type:
    *
    * @example
    * {{{
    * class MyBundle extends Bundle with HasTypeAlias {
    *   override def aliasName = Some("UserBundle")
    * }
    * }}}
    *
    * {{{
    * circuit Top :
    *   type UserBundle = { ... }
    *   module Top :
    *     // ...
    * }}}
    *
    * This is used as a strong hint for the generated type alias: steps like sanitization and disambiguation
    * may change the resulting alias by necessity, so there is no certain guarantee that the desired name will show up in
    * the generated FIRRTL.
    */
  def aliasName: Option[BundleAlias] = None

  // The final sanitized and disambiguated alias for this bundle, generated when aliasName is a non-empty Option.
  // This is important if sanitization and disambiguation results in a changed alias,
  // as the sanitized name no longer matches the user-specified alias.
  private[chisel3] var finalizedAlias: Option[String] = None
}