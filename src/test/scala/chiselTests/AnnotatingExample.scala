// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.core.Module
import chisel3.internal.Builder
import chisel3.internal.firrtl.{Emitter, Circuit}
import chisel3.testers.BasicTester
import org.scalatest._
import org.scalatest.prop._

import scala.util.DynamicVariable

class SomeSubMod extends Module {
  val io = new Bundle {
    val in = UInt(INPUT, 16)
    val out = SInt(OUTPUT, 32)
  }
  MyBuilder.myDynamicContext.annotationMap(AnnotationKey(io.in, AllRefs))  = "sub mod io.in"
  MyBuilder.myDynamicContext.annotationMap(AnnotationKey(io.out, JustThisRef)) = "sub mod io.out"
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

  val subModule1 = Module(new SomeSubMod)
  val subModule2 = Module(new SomeSubMod)


  val annotate = MyBuilder.myDynamicContext.annotationMap

  annotate(AnnotationKey(x, JustThisRef)) = "I am register X"
  annotate(AnnotationKey(io.a, JustThisRef)) = "I am io.a"
  annotate(AnnotationKey(io.bun.nested_1, JustThisRef)) = "I am io.bun.nested_1"
  annotate(AnnotationKey(io.bun.nested_2, JustThisRef)) = "I am io.bun.nested_2"

  when (x > y)   { x := x -% y }
  .otherwise     { y := y -% x }
  when (io.e) { x := io.a; y := io.b }
  io.z := x
  io.v := y === UInt(0)
}

class AnnotatingExampleTester(a: Int, b: Int, z: Int) extends BasicTester {
  val dut = Module(new AnnotatingExample)
  val first = Reg(init=Bool(true))
  dut.io.a := UInt(a)
  dut.io.b := UInt(b)
  dut.io.e := first
  when(first) { first := Bool(false) }
  when(!first && dut.io.v) {
    assert(dut.io.z === UInt(z))
    stop()
  }
}

class AnnotatingExampleSpec extends ChiselPropSpec {

  property("show node info") {
    MyDriver.doStuff { () => new AnnotatingExampleTester(1, 2, 3) }
  }

}

trait AnnotationScope
case object Default extends AnnotationScope
case object AllRefs extends AnnotationScope
case object JustThisRef extends AnnotationScope

case class AnnotationKey(val component: Data, scope: AnnotationScope)

class MyDynamicContext {
  val annotationMap = new scala.collection.mutable.HashMap[AnnotationKey, String]
}

object MyBuilder {
  private val myDynamicContextVar = new DynamicVariable[Option[MyDynamicContext]](None)

  def myDynamicContext: MyDynamicContext =
    myDynamicContextVar.value getOrElse (new MyDynamicContext)

  def build[T <: Module](f: => T): Unit = {
    myDynamicContextVar.withValue(Some(new MyDynamicContext)) {
      Driver.emit(() => f)
      val list = myDynamicContextVar.value.get.annotationMap.map { case (k,v) =>
        k match {
          case (AnnotationKey(signal, JustThisRef)) =>
            f"Just this ref ${signal.pathName + signal.signalName}%60s -> $v%30s  component $signal"
          case (AnnotationKey(signal, AllRefs)) =>
            f"All refs      ${signal.signalName}%60s -> $v%30s  component $signal"
          case  _ =>
            s"Unknown annotation key $k"
        }
      }.toList.sorted
      println(list.mkString("\n"))
    }
  }
}

object MyDriver extends BackendCompilationUtilities {
  def doStuff[T <: Module](gen: () => T): Unit = MyBuilder.build(Module(gen()))
}