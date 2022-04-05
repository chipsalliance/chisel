// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform, WithContextTransform}
import java.util.IdentityHashMap

/** Represents a view of a proto from a specific hierarchical path */
sealed trait Hierarchy[+P] {

  //private[chisel3] val cache = new IdentityHashMap[Any, Any]()

  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[@public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the hierarchy.
    *
    * Our @instantiable and @public macros generate the calls to this apply method
    *
    * By calling this function, we summon the proper Lookupable typeclass from our implicit scope.
    *
    * @param that a user-specified lookup function
    * @param lookup typeclass which contains the correct lookup function, based on the types of A and B
    * @param macroGenerated a value created in the macro, to make it harder for users to use this API
    */
  def _lookup[B](
    that: P => B
  )(
    implicit lookupable: Lookupable[B],
    macroGenerated:      chisel3.internal.MacroGenerated
  ): lookupable.R = {
    // TODO: Call to 'that' should be replaced with shapeless to enable deserialized Underlying
    val protoValue = that(proto)
    proxy.retrieveMeAsHierarchy(protoValue).orElse(proxy.retrieveMe(protoValue)).orElse {
      val retValue = lookupable.apply(protoValue, this)
      proxy.cacheMe(protoValue, retValue)
      Some(retValue)
    }.get.asInstanceOf[lookupable.R]
  }

  /** Finds the closest parent Instance/Hierarchy in proxy's lineage which matches a partial function
    *
    * @param pf selection partial function
    * @return closest matching parent in lineage which matches pf, if one does
    */
  def getLineageOf[T](pf: PartialFunction[Any, Hierarchy[T]]): Option[Hierarchy[T]] = {
    pf.lift(this)
      .orElse(proxy.lineageOpt.flatMap {
        case d: DefinitionProxy[_] => d.toDefinition.getLineageOf(pf)
        case i: InstanceProxy[_]   => i.toInstance.getLineageOf(pf)
        case other => println(s"NONE!! $other"); None
      })
  }

  /** Useful to view underlying proxy as another type it is representing
    *
    * @return proxy with a different type
    */
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

  def isNarrowerOrEquivalentTo[X](other: Hierarchy[X]): Boolean = {
    (this, other, this.proto == other.proto) match {
      case (_,                _,                false) => /*println(0);*/ false
      case (t: Definition[P], o: Definition[P], true)  => /*println(1);*/ true
      case (t: Definition[P], o: Instance[P],   true)  => /*println(2);*/ true
      case (t: Instance[P],   o: Definition[P], true)  => /*println(3);*/ false
      case (t: Instance[P],   o: Instance[P],   true)  => (t.proxy.lineageOpt, o.proxy.lineageOpt) match {
        case (Some(lt), Some(lo)) => /*println(4);*/ lt.toHierarchy.isNarrowerOrEquivalentTo(lo.toHierarchy)
        case (Some(_),     None)  => /*println(5);*/ false
        case (None,     Some(_))  => /*println(6);*/ true
        case (None,  None)        => /*println(7);*/ true
      }
    }
  }

  /** @return Return the proxy Definition[P] of this Hierarchy[P] */
  def toDefinition: Definition[P]

  def toContext: Context[P] = Context(proxy)

  /** Given a proto's contextual, return the contextuals value from this hierarchical path
    * @param contextual proto's contextual
    * @return contextual value from this hierarchical path
    */
  //private[chisel3] def open[T](contextual: Contextual[T]): T = {
  //  val c = proxy.contexts.map { case (context) => context.lookupContextual(contextual) }.collectFirst{case Some(c) => c}.getOrElse(contextual)
  //  c.compute(this).get
  //}



  /** @return Underlying proxy representing a proto in viewed from a hierarchical path */
  private[chisel3] def proxy: Proxy[P]

  /** @return Underlying proto, which is the actual underlying object we are representing */
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

/** Represents an Instance of a proto, from a specific hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class Instance[+P] private[chisel3] (private[chisel3] proxy: InstanceProxy[P]) extends Hierarchy[P] {

  override def toDefinition = proxy.toDefinition

  override def proxyAs[T]: InstanceProxy[P] with T = proxy.asInstanceOf[InstanceProxy[P] with T]
}

object Instance {
  def apply[P](definition: Definition[P]): Instance[P] =
    macro InstanceTransform.apply[P]
  def do_apply[P](definition: Definition[P])(implicit stampable: ProxyInstancer[P]): Instance[P] = {
    new Instance(stampable(definition))
  }
  def withContext[P](definition: Definition[P])(fs: (Context[P] => Unit)*): Instance[P] =
    macro WithContextTransform.withContext[P]
  def do_withContext[P](
    definition: Definition[P]
  )(fs:         (Context[P] => Unit)*
  )(
    implicit stampable: ProxyInstancer[P]
  ): Instance[P] = {
    val i = new Instance(stampable(definition))
    val context = i.proxy.toContext
    fs.foreach { f =>
      f(context)
    }
    //println(i.proxy.edits.values())
    i
  }
}

/** Represents a Definition of a proto, at the root of a hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class Definition[+P] private[chisel3] (private[chisel3] proxy: DefinitionProxy[P])
    extends IsLookupable
    with Hierarchy[P] {

  override def toDefinition = this

  override def proxyAs[T]: DefinitionProxy[P] with T = proxy.asInstanceOf[DefinitionProxy[P] with T]
}

object Definition {
  def apply[P](proto: => P): Definition[P] =
    macro DefinitionTransform.apply[P]
  def do_apply[P](proto: => P)(implicit buildable: ProxyDefiner[P]): Definition[P] = {
    new Definition(buildable(proto))
  }
}
