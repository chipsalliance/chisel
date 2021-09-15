package chisel3.experimental.hierarchy

import chisel3._
import scala.collection.mutable.{HashMap, HashSet}
import scala.reflect.runtime.universe.WeakTypeTag
import chisel3.internal.BaseModule.IsClone
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.IsModule
import scala.annotation.implicitNotFound
import scala.reflect.runtime.universe.TypeTag

trait Hierarchy[+A] {
  private[chisel3] def cloned: Either[A, IsClone[A]]
  private[chisel3] def getProto: A = cloned match {
    case Left(value: A) => value
    case Right(i: IsClone[A]) => i._proto
  }
  private[chisel3] def protoTypeString: String
  // SCALA Reflection API
  def isA[B : WeakTypeTag]: Boolean


  // JAVA Reflection API
  //private [chisel3] val superClasses = HashSet[String]()
  //private def updateSuperClasses(clz: Class[_]): Unit = {
  //  if(clz != null) {
  //    superClasses += clz.getCanonicalName()
  //    clz.getInterfaces().foreach(i => updateSuperClasses(i))
  //    updateSuperClasses(clz.getSuperclass())
  //    println(s"Type Parameters: " + clz.getTypeParameters().toList.map(_.toString))
  //  }
  //}
  //private[chisel3] def inBaseClasses(clz: String): Boolean = {
  //  if(superClasses.isEmpty) updateSuperClasses(getProto.getClass())
  //  superClasses.contains(clz)
  //}



  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Data, Data]()
  private[chisel3] def getInnerDataContext: Option[BaseModule]


  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[@public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the hierarhcy.
    *
    * Our @instantiable and @public macros generate the calls to this apply method
    *
    * By calling this function, we summon the proper Lookupable typeclass from our implicit scope.
    *
    * @param that a user-specified lookup function
    * @param lookup typeclass which contains the correct lookup function, based on the types of A and B
    * @param macroGenerated a value created in the macro, to make it harder for users to use this API
    */
  def _lookup[B, C](that: A => B)(implicit lookup: Lookupable[B], macroGenerated: chisel3.internal.MacroGenerated): lookup.C
}

//object Hierarchy {
//  implicit class HierarchyReflectiveExtensions[T](i: Hierarchy[T]) {
//    def isA[B : WeakTypeTag]: Boolean = {
//      val tpe = implicitly[WeakTypeTag[B]].tpe
//      val b = tpe.toString
//      val ret = i.inBaseClasses(b)
//      println(i.superClasses)
//      println(b)
//      ret
//    }
//    @implicitNotFound("This is a test")
//    def isAn[B : WeakTypeTag]: Boolean = isA[B]
//  }
//
//}