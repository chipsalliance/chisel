// SPDX-License-Identifier: Apache-2.0

package chisel3

/** Provides ScalaDoc information for "hidden" `do_*` methods
  *
  * Mix this into classes/objects that have `do_*` methods to get access to the shared `SourceInfoTransformMacro`
  * ScalaDoc group and the lengthy `groupdesc` below.
  *
  * @groupdesc SourceInfoTransformMacro
  *
  * <p>
  *   '''These internal methods are not part of the public-facing API!'''
  *   <br>
  *   <br>
  *
  *   The equivalent public-facing methods do not have the `do_` prefix or have the same name. Use and look at the
  *   documentation for those. If you want left shift, use `<<`, not `do_<<`. If you want conversion to a
  *   [[scala.collection.Seq Seq]] of [[Bool]]s look at the `asBools` above, not the one below. Users can safely ignore
  *   every method in this group! <br> <br>
  *
  *   🐉🐉🐉 '''Here be dragons...''' 🐉🐉🐉
  *   <br>
  *   <br>
  *
  *   These `do_X` methods are used to enable both implicit passing of SourceInfo
  *   while also supporting chained apply methods. In effect all "normal" methods that you, as a user, will use in your
  *   designs, are converted to their "hidden", `do_*`, via macro transformations. Without using macros here, only one
  *   of the above wanted behaviors is allowed (implicit passing and chained applies)---the compiler interprets a
  *   chained apply as an explicit 'implicit' argument and will throw type errors. <br> <br>
  *
  *   The "normal", public-facing methods then take no SourceInfo. However, a macro transforms this public-facing method
  *   into a call to an internal, hidden `do_*` that takes an explicit SourceInfo by inserting an
  *   `implicitly[SourceInfo]` as the explicit argument. </p>
  *
  * @groupprio SourceInfoTransformMacro 1001
  */
trait SourceInfoDoc
