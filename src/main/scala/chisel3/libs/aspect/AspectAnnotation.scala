package chisel3.libs.aspect

import chisel3.experimental.RunFirrtlTransform
import firrtl.{RenameMap, Transform}
import firrtl.annotations._
import firrtl.ir.{DefInstance, Statement}

import scala.collection.mutable

case class AspectAnnotation(ioConnects: Seq[(Component, Component)],
                            instance: String,
                            module: firrtl.ir.DefModule,
                            shouldInline: Boolean
                           ) extends BrittleAnnotation with RunFirrtlTransform {
  override def toFirrtl: Annotation = this
  override def transformClass: Class[_ <: Transform] = classOf[AspectTransform]

  override def targets: Seq[Component] = ioConnects.flatMap(x => Seq(x._1, x._2))
  override def duplicate(targets: Seq[Component]): BrittleAnnotation =
    AspectAnnotation(targets.grouped(2).toSeq.map(x => (x(0), x(1))), instance, module, shouldInline)
}

