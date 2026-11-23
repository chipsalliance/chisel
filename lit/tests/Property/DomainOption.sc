// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s
// XFAIL: *
// TODO: Remove XFAIL once the bundled firtool supports property options.

import chisel3._
import chisel3.domain.{Domain, Field}
import chisel3.properties.{AnyClassType, ClassType, Property}

object RootDomain extends Domain {
  override def fields = Seq("root" -> Field.Option(Field.AnyRef))
  def apply(root: Property[_]): domain.Type = super.apply(root)
}

class DomainOptionTop extends RawModule {
  val empty = Property[Option[AnyClassType]](None)
  val aRef = RootDomain(empty).asAnyClassType
  val b = RootDomain(Property.some(aRef))
  val bRef = b.asAnyClassType
  val root = b.field.root.asInstanceOf[Property[Option[ClassType]]].getOrElse(bRef)
  (root === aRef).assert("populated option must preserve source identity")
  (empty.getOrElse(bRef) === bRef).assert("empty option must return its default")
  (!(aRef === bRef)).assert("distinct domains must have distinct identities")
}

// CHECK: module DomainOptionTop(
println(
  circt.stage.ChiselStage.emitSystemVerilog(
    new DomainOptionTop,
    firtoolOpts = Array("--domain-mode=infer-all", "--strip-domain=false")
  )
)
