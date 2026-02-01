// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.reflect.DataMirror.internal.chiselTypeClone
import chisel3.experimental.SourceInfo
import chisel3.internal.binding.DynamicIndexBinding

/** Package for experimental features, which may have their API changed, be removed, etc.
  *
  * Because its contents won't necessarily have the same level of stability and support as
  * non-experimental, you must explicitly import this package to use its contents.
  */
package object experimental {
  import scala.language.implicitConversions
  import chisel3.internal.BaseModule

  // Implicit conversions for BlackBox Parameters
  implicit def fromIntToIntParam(x:       Int):    chisel3.IntParam = chisel3.IntParam(BigInt(x))
  implicit def fromLongToIntParam(x:      Long):   chisel3.IntParam = chisel3.IntParam(BigInt(x))
  implicit def fromBigIntToIntParam(x:    BigInt): chisel3.IntParam = chisel3.IntParam(x)
  implicit def fromDoubleToDoubleParam(x: Double): chisel3.DoubleParam = chisel3.DoubleParam(x)
  implicit def fromStringToStringParam(x: String): chisel3.StringParam = chisel3.StringParam(x)

  // Rocket Chip-style clonemodule

  /** A record containing the results of CloneModuleAsRecord
    * The apply method is retrieves the element with the supplied name.
    */
  type ClonePorts = BaseModule.ClonePorts

  object CloneModuleAsRecord extends CloneModuleAsRecord$Intf {

    private[chisel3] def _applyImpl(
      proto: BaseModule
    )(
      implicit sourceInfo: chisel3.experimental.SourceInfo
    ): ClonePorts = {
      BaseModule.cloneIORecord(proto)
    }
  }

  /** Requires that a node is hardware ("bound")
    */
  object requireIsHardware {
    def apply(node: Data, msg: String = ""): Unit = {
      if (!node.isSynthesizable) {
        val prefix = if (msg.nonEmpty) s"$msg " else ""
        throw ExpectedHardwareException(
          s"$prefix'$node' must be hardware, " +
            "not a bare Chisel type. Perhaps you forgot to wrap it in Wire(_) or IO(_)?"
        )
      }
    }
  }

  /** Require that a Data can be annotated. It must be non-literal hardware.
    */
  object requireIsAnnotatable {
    def apply(node: Data, msg: String = ""): Unit = {
      def prefix = if (msg.nonEmpty) s"$msg " else ""
      requireIsHardware(node, msg)
      if (node.isLit) {
        throw ExpectedAnnotatableException(
          s"$prefix'$node' must not be a literal."
        )
      }
      if (node.topBinding.isInstanceOf[DynamicIndexBinding]) {
        throw ExpectedAnnotatableException(
          s"$prefix'$node' must not be a dynamic index into a Vec. Try assigning it to a Wire."
        )
      }
    }
  }

  /** Requires that a node is a chisel type (not hardware, "unbound")
    */
  object requireIsChiselType {
    def apply(node: Data, msg: String = ""): Unit = if (node.isSynthesizable) {
      val prefix = if (msg.nonEmpty) s"$msg " else ""
      throw ExpectedChiselTypeException(s"$prefix'$node' must be a Chisel type, not hardware")
    }
  }

  type Direction = ActualDirection
  val Direction = ActualDirection

  @deprecated("FlatIO has moved to package chisel3", "Chisel 6.0")
  val FlatIO = chisel3.FlatIO

  /** Generate prefixes from values of this type in the Chisel compiler plugin
    *
    * Users can mixin this trait to tell the Chisel compiler plugin to include the names of
    * vals of this type when generating prefixes for naming `Data` and `Mem` instances.
    * This is generally useful whenever creating a `class` that contains `Data`, `Mem`,
    * or `Module` instances but does not itself extend `Data` or `Module`.
    *
    * @see See [[https://www.chisel-lang.org/chisel3/docs/explanations/naming.html the compiler plugin documentation]] for more information on this process.
    *
    * @example {{{
    * import chisel3._
    * import chisel3.experimental.AffectsChiselPrefix
    *
    * class MyModule extends Module {
    *   // Note: This contains a Data but is not a named component itself
    *   class NotAData extends AffectsChiselPrefix {
    *     val value = Wire(Bool())
    *   }
    *
    *   // Name with AffectsChiselPrefix:    "nonData_value"
    *   // Name without AffectsChiselPrefix: "value"
    *   val nonData = new NotAData
    *
    *   // Name with AffectsChiselPrefix:    "nonData2_value"
    *   // Name without AffectsChiselPrefix: "value_1"
    *   val nonData2 = new NotAData
    * }
    * }}}
    */
  trait AffectsChiselPrefix

  /** If mixed in with a user-defined type, Chisel will attempt to name instances of the type
    */
  trait AffectsChiselName

  object BundleLiterals {
    implicit class AddBundleLiteralConstructor[T <: Record](val x: T) extends AddBundleLiteralConstructorIntf[T] {
      private[chisel3] def _LitImpl(elems: (T => (Data, Data))*)(implicit sourceInfo: SourceInfo): T = {
        val fs = elems.map(_.asInstanceOf[Data => (Data, Data)])
        x._makeLit(fs: _*).asInstanceOf[T]
      }
    }
  }

  /** This class provides the `Lit` method needed to define a `Vec` literal
    */
  object VecLiterals {
    implicit class AddVecLiteralConstructor[T <: Data](val x: Vec[T]) extends AddVecLiteralConstructorIntf[T] {

      private[chisel3] def _LitImpl(elems: (Int, T)*)(implicit sourceInfo: SourceInfo): Vec[T] = {
        x._makeLit(elems: _*)
      }
    }

    implicit class AddObjectLiteralConstructor(val x: Vec.type) extends AddObjectLiteralConstructorIntf {

      private[chisel3] def _LitImpl[T <: Data](elems: T*)(implicit sourceInfo: SourceInfo): Vec[T] = {
        val sampleElement = cloneSupertype(elems, s"Vec.Lit(...)")
        val vec: Vec[T] = Vec.apply(elems.length, sampleElement)
        vec.Lit(elems.zipWithIndex.map(_.swap): _*)
      }
    }
  }

  // ****************************** Hardware equivalents of Scala Tuples ******************************
  // These are intended to be used via DataView

  /** [[Data]] equivalent of Scala's [[scala.Tuple2]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple2` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple2[+A <: Data, +B <: Data] private[chisel3] (val _1: A, val _2: B) extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin:   Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple2(chiselTypeClone(_1), chiselTypeClone(_2))
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple3]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple3` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple3[+A <: Data, +B <: Data, +C <: Data] private[chisel3] (val _1: A, val _2: B, val _3: C)
      extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple3(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple4]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple4` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple4[+A <: Data, +B <: Data, +C <: Data, +D <: Data] private[chisel3] (
    val _1: A,
    val _2: B,
    val _3: C,
    val _4: D
  ) extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple4(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple5]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple5` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple5[+A <: Data, +B <: Data, +C <: Data, +D <: Data, +E <: Data] private[chisel3] (
    val _1: A,
    val _2: B,
    val _3: C,
    val _4: D,
    val _5: E
  ) extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple5(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4),
      chiselTypeClone(_5)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4,
      "_5" -> _5
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple6]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple6` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple6[+A <: Data, +B <: Data, +C <: Data, +D <: Data, +E <: Data, +F <: Data] private[chisel3] (
    val _1: A,
    val _2: B,
    val _3: C,
    val _4: D,
    val _5: E,
    val _6: F
  ) extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple6(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4),
      chiselTypeClone(_5),
      chiselTypeClone(_6)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4,
      "_5" -> _5,
      "_6" -> _6
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple7]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple7` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple7[
    +A <: Data,
    +B <: Data,
    +C <: Data,
    +D <: Data,
    +E <: Data,
    +F <: Data,
    +G <: Data
  ] private[chisel3] (val _1: A, val _2: B, val _3: C, val _4: D, val _5: E, val _6: F, val _7: G)
      extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple7(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4),
      chiselTypeClone(_5),
      chiselTypeClone(_6),
      chiselTypeClone(_7)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4,
      "_5" -> _5,
      "_6" -> _6,
      "_7" -> _7
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple8]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple8` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple8[
    +A <: Data,
    +B <: Data,
    +C <: Data,
    +D <: Data,
    +E <: Data,
    +F <: Data,
    +G <: Data,
    +H <: Data
  ] private[chisel3] (val _1: A, val _2: B, val _3: C, val _4: D, val _5: E, val _6: F, val _7: G, val _8: H)
      extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple8(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4),
      chiselTypeClone(_5),
      chiselTypeClone(_6),
      chiselTypeClone(_7),
      chiselTypeClone(_8)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4,
      "_5" -> _5,
      "_6" -> _6,
      "_7" -> _7,
      "_8" -> _8
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple9]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple9` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple9[
    +A <: Data,
    +B <: Data,
    +C <: Data,
    +D <: Data,
    +E <: Data,
    +F <: Data,
    +G <: Data,
    +H <: Data,
    +I <: Data
  ] private[chisel3] (val _1: A, val _2: B, val _3: C, val _4: D, val _5: E, val _6: F, val _7: G, val _8: H, val _9: I)
      extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple9(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4),
      chiselTypeClone(_5),
      chiselTypeClone(_6),
      chiselTypeClone(_7),
      chiselTypeClone(_8),
      chiselTypeClone(_9)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4,
      "_5" -> _5,
      "_6" -> _6,
      "_7" -> _7,
      "_8" -> _8,
      "_9" -> _9
    )
  }

  /** [[Data]] equivalent of Scala's [[scala.Tuple9]]
    *
    * Users may not instantiate this class directly. Instead they should use the implicit conversion from `Tuple9` in
    * `chisel3.experimental.conversions`
    */
  final class HWTuple10[
    +A <: Data,
    +B <: Data,
    +C <: Data,
    +D <: Data,
    +E <: Data,
    +F <: Data,
    +G <: Data,
    +H <: Data,
    +I <: Data,
    +J <: Data
  ] private[chisel3] (
    val _1:  A,
    val _2:  B,
    val _3:  C,
    val _4:  D,
    val _5:  E,
    val _6:  F,
    val _7:  G,
    val _8:  H,
    val _9:  I,
    val _10: J
  ) extends Bundle() {
    // Because this implementation exists in chisel3.core, it cannot compile with the plugin, so we implement the behavior manually
    override protected def _usingPlugin: Boolean = true
    override protected def _cloneTypeImpl: Bundle = new HWTuple10(
      chiselTypeClone(_1),
      chiselTypeClone(_2),
      chiselTypeClone(_3),
      chiselTypeClone(_4),
      chiselTypeClone(_5),
      chiselTypeClone(_6),
      chiselTypeClone(_7),
      chiselTypeClone(_8),
      chiselTypeClone(_9),
      chiselTypeClone(_10)
    )
    override protected def _elementsImpl: Iterable[(String, Any)] = Vector(
      "_1" -> _1,
      "_2" -> _2,
      "_3" -> _3,
      "_4" -> _4,
      "_5" -> _5,
      "_6" -> _6,
      "_7" -> _7,
      "_8" -> _8,
      "_9" -> _9,
      "_10" -> _10
    )
  }
}
