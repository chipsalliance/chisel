package chisel3.libs.aspect

import chisel3.experimental.RunFirrtlTransform
import firrtl.{RenameMap, Transform}
import firrtl.annotations.{Annotation, ComponentName, ModuleName, Named, Component}
import firrtl.ir.DefInstance

import scala.collection.mutable

case class AspectAnnotation(connections: Seq[(Component, Component)], enclosingModule: Component, instance: DefInstance, module: firrtl.ir.DefModule) extends Annotation with RunFirrtlTransform {
  override def toFirrtl: Annotation = this
  override def transformClass: Class[_ <: Transform] = classOf[AspectTransform]
  private val errors = mutable.ArrayBuffer[String]()
  private def rename(c: Component, renames: RenameMap): Component = {
    val ret = (c, renames.get(c)) match {
      case (c: Component, Some(Seq(x: Component))) => x
      case (_, None) => c
      case (_, other) =>
        errors += s"Bad rename in ${this.getClass}: $c to $other"
        c
    }
    ret
  }
  override def update(renames: RenameMap): Seq[Annotation] = {
    import Component.convertComponent2Named
    import Component.convertNamed2Component
    val newConnections = connections.map { case (from, to) => (rename(from, renames), rename(to, renames)) }
    val newEncl = rename(enclosingModule, renames)
    if(errors.nonEmpty) {
      throw new Exception(errors.mkString("\n"))
    }
    Seq(AspectAnnotation(newConnections, newEncl, instance, module))
  }
}

