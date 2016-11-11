// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.Module
import chisel3.internal.InstanceId
import chisel3.testers.BasicTester
import org.scalatest._

import scala.util.DynamicVariable

//scalastyle:off magic.number

/**
  * This Spec file illustrates use of Donggyu's component name API, it currently only
  * uses three methods .signalName, .parentModName and .pathName
  *
  * This is also an illustration of how to implement an annotation system in chisel3
  * A local (my) Driver and Builder are created to provide thread-local access to
  * an annotation map, and then a post elaboration annotation processor can resolve
  * the keys and could serialize the annotations to a file for use by firrtl passes
  */

class SomeSubMod(param1: Int, param2: Int) extends Module {
  val io = new Bundle {
    val in = UInt(INPUT, 16)
    val out = SInt(OUTPUT, 32)
  }
  val annotate = MyBuilder.myDynamicContext.annotationMap

  annotate(AnnotationKey(this, JustThisRef))   = s"SomeSubMod($param1, $param2)"
  annotate(AnnotationKey(io.in, AllRefs))      = "sub mod io.in"
  annotate(AnnotationKey(io.out, JustThisRef)) = "sub mod io.out"
}

class AnnotatingExample extends Module {
  val io = new Bundle {
    val a  = UInt(INPUT, 32)
    val b  = UInt(INPUT, 32)
    val e  = Bool(INPUT)
    val z  = UInt(OUTPUT, 32)
    val v  = Bool(OUTPUT)
    val bun = new Bundle {
      val nested_1 = UInt(INPUT, 12)
      val nested_2 = Bool(OUTPUT)
    }
  }
  val x = Reg(UInt(width = 32))
  val y = Reg(UInt(width = 32))

  val subModule1 = Module(new SomeSubMod(1, 2))
  val subModule2 = Module(new SomeSubMod(3, 4))


  val annotate = MyBuilder.myDynamicContext.annotationMap

  annotate(AnnotationKey(subModule2, AllRefs))     = s"SomeSubMod was used"

  annotate(AnnotationKey(x, JustThisRef)) = "I am register X"
  annotate(AnnotationKey(y, AllRefs)) = "I am register Y"
  annotate(AnnotationKey(io.a, JustThisRef)) = "I am io.a"
  annotate(AnnotationKey(io.bun.nested_1, AllRefs)) = "I am io.bun.nested_1"
  annotate(AnnotationKey(io.bun.nested_2, JustThisRef)) = "I am io.bun.nested_2"
}

class AnnotatingExampleTester extends BasicTester {
  val dut = Module(new AnnotatingExample)

  stop()
}

class AnnotatingExampleSpec extends FlatSpec with Matchers {
  behavior of "Annotating components of a circuit"

  it should "contain the following relative keys" in {
    val annotationMap = MyDriver.buildAnnotatedCircuit { () => new AnnotatingExampleTester }

    annotationMap.contains("SomeSubMod.io.in") should be(true)
    annotationMap.contains("AnnotatingExample.y") should be(true)

    annotationMap("SomeSubMod.io.in") should be("sub mod io.in")
  }
  it should "contain the following absolute keys" in {
    val annotationMap = MyDriver.buildAnnotatedCircuit { () => new AnnotatingExampleTester }

    annotationMap.contains("AnnotatingExampleTester.dut.subModule2.io.out") should be (true)
    annotationMap.contains("AnnotatingExampleTester.dut.x") should be (true)

    annotationMap("AnnotatingExampleTester.dut.subModule2.io.out") should be ("sub mod io.out")
  }
}

trait AnnotationScope
case object AllRefs     extends AnnotationScope
case object JustThisRef extends AnnotationScope

object AnnotationKey {
  def apply(component: InstanceId): AnnotationKey = {
    AnnotationKey(component, AllRefs)
  }
}
case class AnnotationKey(val component: InstanceId, scope: AnnotationScope) {
  override def toString: String = {
    scope match {
      case JustThisRef =>
        s"${component.pathName}"
      case AllRefs =>
        s"${component.parentModName}.${component.instanceName}"
      case  _ =>
        s"${component.toString}_unknown_scope"
    }
  }
}

class AnnotationMap extends scala.collection.mutable.HashMap[AnnotationKey, String]

class MyDynamicContext {
  val annotationMap = new AnnotationMap
}

object MyBuilder {
  private val myDynamicContextVar = new DynamicVariable[Option[MyDynamicContext]](None)

  def myDynamicContext: MyDynamicContext =
    myDynamicContextVar.value getOrElse new MyDynamicContext

  def processAnnotations(annotationMap: AnnotationMap): Map[String, String] = {
    annotationMap.map { case (k,v) => k.toString -> v}.toMap
  }

  def build[T <: Module](f: => T): Map[String, String] = {
    myDynamicContextVar.withValue(Some(new MyDynamicContext)) {
      Driver.emit(() => f)
      processAnnotations(myDynamicContextVar.value.get.annotationMap)
    }
  }
}

object MyDriver extends BackendCompilationUtilities {
  /**
    * illustrates a chisel3 style driver that, annotations can only processed within this structure
    */
  def buildAnnotatedCircuit[T <: Module](gen: () => T): Map[String, String] = MyBuilder.build(gen())
}
