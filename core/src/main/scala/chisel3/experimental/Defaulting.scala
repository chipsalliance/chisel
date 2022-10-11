// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.sourceinfo.SourceInfo

import scala.collection.immutable.SeqMap

/** Data type for representing any type that has a defined default value
  *
  * Uses the underlying opaque type support.
  *
  * @note This API is experimental and subject to change
  */
final class Defaulting[T <: Data] private (
  tpe:          => T,
  defaultValue: => T
)(
  implicit sourceInfo: SourceInfo,
  compileOptions:      CompileOptions)
    extends Record {
  requireIsChiselType(tpe, s"Chisel hardware type $tpe must be a pure type, not bound to hardware.")
  requireIsHardware(defaultValue, s"Default value $defaultValue must be bound to a hardware component")

  /** The underlying hardware component, is either the Chisel data type (if `this` is unbound) or hardware component (if `this` is bound to hardware) */
  lazy val underlying: T = tpe

  /** The default value for this Defaulting */
  lazy val default: T = defaultValue

  val elements = SeqMap("" -> underlying)
  override def opaqueType = elements.size == 1
  override def cloneType: this.type = {
    val freshType = if (tpe.isSynthesizable) chiselTypeOf(tpe) else tpe.cloneType
    (new Defaulting[T](freshType, defaultValue)).asInstanceOf[this.type]
  }
}

/** Object that provides factory methods for [[Defaulting]] objects
  *
  * @note This API is experimental and subject to change
  */
object Defaulting {

  /** Build a Defaulting[T <: Data]
    *
    * @param tpe the Chisel data type
    * @param default the Chisel default value, must be bound to a hardware value
    */
  def apply[T <: Data](
    tpe:     T,
    default: T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Defaulting[T] = new Defaulting(tpe, default)

  /** Build a Defaulting[T <: Data]
    *
    * @param default the Chisel default value, must be bound to a hardware value. The underlying type is pulled from the default value.
    */
  def apply[T <: Data](default: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Defaulting[T] =
    new Defaulting(chiselTypeOf(default), default)

  implicit class DefaultingTypeclass[T <: Data](h: T) {
    def withConnectableDefault(fs: (T => (Data, Data))*)(implicit sourceInfo: SourceInfo): T = {
      fs.foreach { f => f(h) match { case (h, l) => setConnectableDefault(h, l) } }
      h
    }
    def withConnectableDefault(lit: T)(implicit sourceInfo: SourceInfo): T = {
      if(!lit.isLit) internal.Builder.error("Must provide a literal")
      if(h.defaultOrNull != null) internal.Builder.error(s"Cannot set a type's default value twice; setting as $lit, but already set as ${h.defaultOrNull}")
      setConnectableDefault(h, lit)
      h
    }
    def withConnectableDontCare(implicit sourceInfo: SourceInfo): T = {
      if(h.defaultOrNull != null) internal.Builder.error(s"Cannot set a type's default value twice; setting as DontCare, but already set as ${h.defaultOrNull}")
      setConnectableDefault(h, DontCare)
      h
    }
    def hasConnectableDefault: Boolean = h.defaultOrNull != null
    def connectableDefault: T = {
      require(hasConnectableDefault)
      h.defaultOrNull.asInstanceOf[T]
    }
  }

  def buildTrie(data: Data): Trie[String, Data] = {
    val trie = Trie.empty[String, Data]
    def recBuildTrie(path: Vector[String], d: Data): Unit = {
      trie.insert(path, d)
      d match {
        case a: Aggregate if a.getElements.size == 0 =>
        case a: Vec[Data @unchecked] =>
          a.getElements.zipWithIndex.foreach { case (e, i) => recBuildTrie(path :+ ("_$_" + i.toString), e)}
        case a: Record =>
          a.elements.foreach { case (field, e) => recBuildTrie(path :+ field, e) }
        case x =>
      }
    }
    recBuildTrie(Vector.empty[String], data)
    trie
  }

  private[chisel3] def setConnectableDefault(h: Data, lit: Data): Unit = {
    val hwTrie = buildTrie(h)
    val litTrie = buildTrie(lit)
    hwTrie.collectDeep {
      case (path, Some(x)) =>
        val inLit = litTrie.get(path)
        inLit.map { y => 
          require(x.defaultOrNull == null)
          require(y.isLit || y == DontCare)
          x.defaultOrNull = y
        }
    }
  }
}

