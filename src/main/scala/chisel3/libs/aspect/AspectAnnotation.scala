package chisel3.libs.aspect

import chisel3.experimental.RunFirrtlTransform
import firrtl.{RenameMap, Transform}
import firrtl.annotations.{Annotation, ComponentName, ModuleName, Named}
import firrtl.ir.DefInstance

import scala.collection.mutable

case class AspectAnnotation(connections: Seq[(ComponentName, ComponentName)], enclosingModule: Named, instance: DefInstance, module: firrtl.ir.DefModule) extends Annotation with RunFirrtlTransform {
  override def toFirrtl: Annotation = this
  override def transformClass: Class[_ <: Transform] = classOf[AspectTransform]
  private val errors = mutable.ArrayBuffer[String]()
  private def rename[T<:Named](n: T, renames: RenameMap): T = (n, renames.get(n)) match {
    case (m: ModuleName, Some(Seq(x: ModuleName))) => x.asInstanceOf[T]
    case (c: ComponentName, Some(Seq(x: ComponentName))) => x.asInstanceOf[T]
    case (_, None) => n
    case (_, other) =>
      errors += s"Bad rename in ${this.getClass}: $n to $other"
      n
  }
  override def update(renames: RenameMap): Seq[Annotation] = {
    val newConnections = connections.map { case (from, to) => (rename(from, renames), rename(to, renames)) }
    val newEncl = rename(enclosingModule, renames)
    if(errors.nonEmpty) {
      throw new Exception(errors.mkString("\n"))
    }
    Seq(AspectAnnotation(newConnections, newEncl, instance, module))
  }
}

