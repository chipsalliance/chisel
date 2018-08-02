package chisel3.libs.aspect

import chisel3.experimental.RunFirrtlTransform
import chisel3.libs.Component
import firrtl.{RenameMap, Transform}
import firrtl.annotations.{Annotation, ComponentName, ModuleName, Named}
import firrtl.ir.DefInstance

import scala.collection.mutable

case class AspectAnnotation(connections: Seq[(Component, Component)], enclosingModule: Named, instance: DefInstance, module: firrtl.ir.DefModule) extends Annotation with RunFirrtlTransform {
  override def toFirrtl: Annotation = this
  override def transformClass: Class[_ <: Transform] = classOf[AspectTransform]
  private val errors = mutable.ArrayBuffer[String]()
  private def rename(c: Component, renames: RenameMap): Component = {
    val n = Component.convertComponent2Named(c)
    val ret = (n, renames.get(n)) match {
      case (m: ModuleName, Some(Seq(x: ModuleName))) => x
      case (c: ComponentName, Some(Seq(x: ComponentName))) => x
      case (_, None) => n
      case (_, other) =>
        errors += s"Bad rename in ${this.getClass}: $n to $other"
        n
    }
    Component.named2component(ret)
  }
  override def update(renames: RenameMap): Seq[Annotation] = {
    import Component.convertComponent2Named
    import Component.named2component
    val newConnections = connections.map { case (from, to) => (rename(from, renames), rename(to, renames)) }
    val newEncl = rename(enclosingModule, renames)
    if(errors.nonEmpty) {
      throw new Exception(errors.mkString("\n"))
    }
    Seq(AspectAnnotation(newConnections, newEncl, instance, module))
  }
}

