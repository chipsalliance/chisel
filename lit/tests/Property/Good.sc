// RUN: scala-cli --server=false --java-home=%JAVAHOME --extra-jars=%RUNCLASSPATH --scala-version=%SCALAVERSION --scala-option="-Xplugin:%SCALAPLUGINJARS" %s | FileCheck %s
import chisel3._
import chisel3.properties._

// CHECK-LABEL: circuit IntPropTest :
// CHECK: public module IntPropTest :
// CHECK-NEXT: output intProp : Integer
class IntPropTest extends RawModule {
  val intProp = IO(Output(Property[Int]()))
  intProp := Property(1)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new IntPropTest))

// CHECK-LABEL: circuit PropertyTest :
class PropertyTest extends Module {
  val i = IO(Input(UInt(8.W)))
  val o = IO(Output(UInt(8.W)))

  val m = Module(new Module {
    val i = IO(Input(UInt(8.W)))
    val r = RegNext(i)
    val o = IO(Output(UInt(8.W)))
    val p = IO(Output(Property[Int]()))
    p := Property(789)
    o := r
    val nested = Module(new Module {
      val i = IO(Input(UInt(8.W)))
      val r = RegNext(i)
      val o = IO(Output(UInt(8.W)))
      val p = IO(Output(Property[Int]()))
      p := Property(789)
      o := r
    })
    nested.i := i
    o := nested.o
  })
  m.i := i
  o := m.o

  // CHECK: output f : Double
  // CHECK: output bool : Bool
  // CHECK: output p : Path
  val f = IO(Output(Property[Double]()))
  val bool = IO(Output(Property[Boolean]()))
  val p = IO(Output(Property[Path]()))
  p := Property(Path(i))

  // CHECK-NEXT: output a : List<List<Integer>>
  // CHECK-NEXT: output b : List<List<Integer>>
  val a = IO(Output(Property[Seq[Seq[Property[Int]]]]()))
  val b = IO(Output(Property[Seq[Property[Seq[Int]]]]()))
  // CHECK:      propassign p, path("OMReferenceTarget:~|PropertyTest>i")
  // CHECK-NEXT: propassign a, List<List<Integer>>(List<Integer>(Integer(123)))
  // CHECK-NEXT: propassign b, List<List<Integer>>(List<Integer>(Integer(456)))
  a := Property(Seq[Seq[Int]](Seq(123)))
  b := Property(Seq[Seq[Int]](Seq(456)))

  // CHECK: propassign f, Double(1.23)
  // CHECK: propassign bool, Bool(true)
  f := Property(1.23)
  bool := Property(true)
}

println(circt.stage.ChiselStage.emitCHIRRTL(new PropertyTest))
