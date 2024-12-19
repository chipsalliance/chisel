package chisel3.experimental.hierarchy.core

import chisel3._
import scala.reflect.runtime.universe.TypeTag

trait HierarchyProto[+A] {
  private[chisel3] def underlying: Underlying[A]
  private[chisel3] def proto: A = underlying match {
    case Proto(value) => value
    case Clone(i: IsClone[A]) => i.getProto
  }
}

trait HierarchyIsA[+A] extends HierarchyProto[A] {
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

  private lazy val superClasses = calculateSuperClasses(super.proto.getClass())
  private def calculateSuperClasses(clz: Class[_]): Set[String] = {
    if (clz != null) {
      Set(modifyReplString(clz.getCanonicalName())) ++
        clz.getInterfaces().flatMap(i => calculateSuperClasses(i)) ++
        calculateSuperClasses(clz.getSuperclass())
    } else {
      Set.empty[String]
    }
  }
  private def inBaseClasses(clz: String): Boolean = superClasses.contains(clz)

  /** Determine whether underlying proto is of type provided.
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[B], which will fail if B is an inner class.
    * @note IMPORTANT: this function IGNORES type parameters, akin to normal type erasure.
    * @note IMPORTANT: this function relies on Java reflection for underlying proto, but Scala reflection for provided type
    *
    * E.g. isA[List[Int]] will return true, even if underlying proto is of type List[String]
    * @return Whether underlying proto is of provided type (with caveats outlined above)
    */
  def isA[B: TypeTag]: Boolean = {
    val tptag = implicitly[TypeTag[B]]
    // drop any type information for the comparison, because the proto will not have that information.
    val name = tptag.tpe.toString.takeWhile(_ != '[')
    inBaseClasses(name)
  }
}
