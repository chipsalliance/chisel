// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform, WithContextTransform}
import java.util.IdentityHashMap

sealed trait Hierarchy[+P] {
  private[chisel3] val cache = new IdentityHashMap[Any, Any]()

  def open[T](contextual: Contextual[T]): T = proxy.compute(contextual, contextual).value

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  def _lookup[B](
    that: P => B
  )(
    implicit lookupable: Lookupable[B],
    macroGenerated:      chisel3.internal.MacroGenerated
  ): lookupable.R = {
    // TODO: Call to 'that' should be replaced with shapeless to enable deserialized Underlying
    val protoValue = that(proto)
    if (cache.containsKey(protoValue)) cache.get(protoValue).asInstanceOf[lookupable.R]
    else {
      val ret = lookupable(protoValue, this)
      cache.put(protoValue, ret)
      ret
    }
  }

  def getLineageOf[T](pf: PartialFunction[Any, Hierarchy[T]]): Option[Hierarchy[T]] = {
    pf.lift(this)
      .orElse(proxy.lineageOpt.flatMap {
        case d: DefinitionProxy[_] => d.toDefinition.getLineageOf(pf)
        case i: InstanceProxy[_]   => i.toInstance.getLineageOf(pf)
        case other => println(s"NONE!! $other"); None
      })
  }

  def proxyAs[T]: Proxy[P] with T = proxy.asInstanceOf[Proxy[P] with T]

  /** Determine whether proxy proto is of type provided.
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[B], which will fail if B is an inner class.
    * @note IMPORTANT: this function IGNORES type parameters, akin to normal type erasure.
    * @note IMPORTANT: this function relies on Java reflection for proxy proto, but Scala reflection for provided type
    *
    * E.g. isA[List[Int]] will return true, even if proxy proto is of type List[String]
    * @return Whether proxy proto is of provided type (with caveats outlined above)
    */
  def isA[B: TypeTag]: Boolean = {
    val tptag = implicitly[TypeTag[B]]
    // drop any type information for the comparison, because the proto will not have that information.
    val name = tptag.tpe.toString.takeWhile(_ != '[')
    superClasses.contains(name)
  }

  /** @return Return the proxy Definition[P] of this Hierarchy[P] */
  def toDefinition: Definition[P]

  //def toLense = Lense(proxy)

  private[chisel3] def proxy: Proxy[P]
  private[chisel3] def proto: P = proxy.proto
  private lazy val superClasses = Hierarchy.calculateSuperClasses(proto.getClass())
}

object Hierarchy {

  // This code handles a special-case where, within an mdoc context, the type returned from
  //  scala reflection (typetag) looks different than when returned from java reflection.
  //  This function detects this case and reshapes the string to match.
  private def modifyReplString(clz: String): String = {
    if (clz != null) {
      clz.split('.').toList match {
        case "repl" :: "MdocSession" :: app :: rest => s"$app.this." + rest.mkString(".")
        case other                                  => clz
      }
    } else clz
  }
  // Nested objects stick a '$' at the end of the object name, but this does not show up in the scala reflection type string
  // E.g.
  // object Foo {
  //   object Bar {
  //     class Baz()
  //   }
  // }
  // Scala type will be Foo.Bar.Baz
  // Java type will be Foo.Bar$.Baz
  private def modifyNestedObjects(clz: String): String = {
    if (clz != null) { clz.replace("$", "") }
    else clz
  }
  private def calculateSuperClasses(clz: Class[_]): Set[String] = {
    if (clz != null) {
      Set(modifyNestedObjects(modifyReplString(clz.getCanonicalName()))) ++
        clz.getInterfaces().flatMap(i => calculateSuperClasses(i)) ++
        calculateSuperClasses(clz.getSuperclass())
    } else {
      Set.empty[String]
    }
  }
}

final case class Instance[+P] private[chisel3] (private[chisel3] proxy: InstanceProxy[P]) extends Hierarchy[P] {
  def toDefinition = proxy.toDefinition
  override def proxyAs[T]: InstanceProxy[P] with T = proxy.asInstanceOf[InstanceProxy[P] with T]
}

object Instance {
  def apply[P](definition: Definition[P]): Instance[P] =
    macro InstanceTransform.apply[P]
  def do_apply[P](definition: Definition[P])(implicit stampable: ProxyInstancer[P]): Instance[P] = {
    new Instance(stampable(definition, Nil))
  }
  def withContext[P](definition: Definition[P])(fs: (TopLense[P] => Unit)*): Instance[P] =
    macro WithContextTransform.withContext[P]
  def do_withContext[P](
    definition: Definition[P]
  )(fs:         (TopLense[P] => Unit)*
  )(
    implicit stampable: ProxyInstancer[P]
  ): Instance[P] = {
    val lenses = fs.map { f =>
      val lense = TopLense(definition.proxy)
      f(lense)
      lense
    }
    val i = new Instance(stampable(definition, lenses))
    i
  }
}

final case class Definition[+P] private[chisel3] (private[chisel3] proxy: DefinitionProxy[P])
    extends IsLookupable
    with Hierarchy[P] {
  def toDefinition = this
  override def proxyAs[T]: DefinitionProxy[P] with T = proxy.asInstanceOf[DefinitionProxy[P] with T]
}

object Definition {
  def apply[P](proto: => P): Definition[P] =
    macro DefinitionTransform.apply[P]
  def do_apply[P](proto: => P)(implicit buildable: ProxyDefiner[P]): Definition[P] = {
    new Definition(buildable(proto))
  }
}
