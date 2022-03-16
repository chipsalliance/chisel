// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import java.util.IdentityHashMap

sealed trait Proxy[+P] {
  def proto: P
  private[chisel3] def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T]
  def contexts:       Seq[Context[P]]
  def lineageOpt:   Option[Proxy[Any]]
  def toDefinition: Definition[P]
}

sealed trait InstanceProxy[+P] extends Proxy[P] {
  def genesis: Proxy[P]
  def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T] = {
    val genesisContextual = genesis.compute(key, contextual)
    contexts.foldLeft(genesisContextual) { case (c, context) => context.compute(key, c) }
  }
  def proto = genesis.proto
  def lineageOfType[C](pf: PartialFunction[Any, C]): Option[C] = lineageOpt match {
    case Some(a) if pf.isDefinedAt(a) => pf.lift(a)
    case Some(i: InstanceProxy[Any]) => i.lineageOfType[C](pf)
    case other => None
  }
  def toInstance = new Instance(this)
  def toDefinition: Definition[P] = genesis.toDefinition
  def localProxy: InstanceProxy[P] = genesis match {
    case d: DefinitionProxy[P] => this
    case i: InstanceProxy[P]   => i.localProxy
  }
}

trait Clone[+P] extends InstanceProxy[P] {
  def lineageOpt: Option[Proxy[Any]] = None
  def genesis: DefinitionProxy[P]
}
trait Transparent[+P] extends InstanceProxy[P] {
  def lineageOpt: Option[Proxy[Any]] = None
  def genesis: DefinitionProxy[P]
}
trait Mock[+P] extends InstanceProxy[P] {
  def lineage: Proxy[Any]
  def lineageOpt: Option[Proxy[Any]] = Some(lineage)
  def genesis: InstanceProxy[P]
}
trait DefinitionProxy[+P] extends Proxy[P] {
  def contexts: Seq[Context[P]] = Nil
  def compute[T](key: Contextual[T], contextual: Contextual[T]): Contextual[T] = {
    contexts.foldLeft(contextual) { case (c, context) => context.compute(key, c) }
  }
  def lineageOpt: Option[Proxy[Any]] = None
  def toDefinition = new Definition(this)
}

final case class InstantiableDefinition[P](proto: P) extends DefinitionProxy[P]
final case class InstantiableTransparent[P](genesis: InstantiableDefinition[P], contexts: Seq[Context[P]])
    extends InstanceProxy[P] {
  val lineageOpt = None
}
final case class InstantiableMock[P](genesis: InstanceProxy[P], lineage: Proxy[Any], contexts: Seq[Context[P]])
    extends InstanceProxy[P] {
  val lineageOpt = Some(lineage)
}
