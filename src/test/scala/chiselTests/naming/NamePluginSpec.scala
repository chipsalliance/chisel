// See LICENSE for license details.

package chiselTests.naming

import chisel3.experimental.{BaseModule, ChiselAnnotation, annotate, chiselName, dump}
import chisel3._
import chisel3.aop.{Aspect, Select}
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselMain, ChiselStage, NoRunFirrtlCompilerAnnotation}
import chiselTests.{ChiselPropSpec, NamedModuleTester}
import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}

import scala.annotation.StaticAnnotation
import scala.reflect.ClassTag

trait ChiselNameCheckAnnotation extends NoTargetAnnotation {
  def check(): Boolean
}
case class ExpectedNameAnnotation(actualName: String, desiredName: String) extends ChiselNameCheckAnnotation {
  def check(): Boolean = actualName == desiredName
}
case class ExpectedTempAnnotation(actualName: String) extends ChiselNameCheckAnnotation {
  def check(): Boolean = actualName(0) == '_'
}

object expectName {
  def apply(m: Data, name: String) = ExpectedNameAnnotation(m.toTarget.ref, name)
  def apply(m: BaseModule, name: String) = ExpectedNameAnnotation(m.instanceName, name)
  def temp(m: Data) = ExpectedTempAnnotation(m.toTarget.ref)
}

object ChiselNameSpec {
  def run[T <: RawModule](gen: () => T, annotations: AnnotationSeq): Seq[ChiselNameCheckAnnotation] = {
    new ChiselStage().run(Seq(ChiselGeneratorAnnotation(gen), NoRunFirrtlCompilerAnnotation) ++ annotations).collect {
      case c: ChiselNameCheckAnnotation => c
    }
  }
  def aspectTest[T <: RawModule](f: T => Unit)(implicit ctag: ClassTag[T]): Unit = {
    case object BuiltAspect extends Aspect[T] {
      override def toAnnotation(top: T): AnnotationSeq = {f(top); Nil}
    }
    BuiltAspect
    run(() => ctag.runtimeClass.newInstance().asInstanceOf[T], Seq(BuiltAspect))
  }
}

@chiselName
class MCN extends RawModule
class MC0 extends RawModule
@chiselName
class MBN extends RawModule {
  { val iC0 = Module(new MC0()) }
  { val iCN = Module(new MCN()) }
}
class MB0 extends RawModule {
  { val iC0 = Module(new MC0()) }
  { val iCN = Module(new MCN()) }
}
@chiselName
class MAN extends MultiIOModule {
  { val iCN = Module(new MCN()) }
  { val iC0 = Module(new MC0()) }
  { val iBN = Module(new MBN()) }
  { val iB0 = Module(new MB0()) }
}
class MA0 extends MultiIOModule {
  { val iCN = Module(new MCN()) }
  { val iC0 = Module(new MC0()) }
  { val iMN = Module(new MBN()) }
  { val iM0 = Module(new MB0()) }
}

class NamePluginSpec extends ChiselPropSpec {

  import ChiselNameSpec._

  property("@chiselName should enable capturing instance names in scope") {


    aspectTest {
      top: MAN =>
        Select.instances(top)(0).instanceName should be("iCN")
        Select.instances(top)(1).instanceName should be("iC0")

        val bN = Select.instances(top)(2)
        bN.instanceName should be("iBN")
        Select.instances(bN)(0).instanceName should be("iC0")
        Select.instances(bN)(1).instanceName should be("iCN")

        // Because B isn't marked as @chiselName, its children instances
        // aren't given instance names
        val b0 = Select.instances(top)(3)
        b0.instanceName should be("iB0")
        Select.instances(b0)(0).instanceName should be("MC0")
        Select.instances(b0)(1).instanceName should be("MCN")
    }

    aspectTest {
      top: MA0 =>
        Select.instances(top)(0).instanceName should be("MCN")
        Select.instances(top)(1).instanceName should be("MC0")

        val bN = Select.instances(top)(2)
        bN.instanceName should be("MBN")
        Select.instances(bN)(0).instanceName should be("iC0")
        Select.instances(bN)(1).instanceName should be("iCN")

        // Because B isn't marked as @chiselName, its children instances
        // aren't given instance names
        val b0 = Select.instances(top)(3)
        b0.instanceName should be("MB0")
        Select.instances(b0)(0).instanceName should be("MC0")
        Select.instances(b0)(1).instanceName should be("MCN")
    }
  }
}

