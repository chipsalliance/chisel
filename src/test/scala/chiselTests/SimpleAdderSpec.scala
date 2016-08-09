//// See LICENSE for license details.
//
//package chiselTests
//
//import chisel3.core._
//import chisel3.{Bundle, Module}
//import org.scalatest.{Matchers, FlatSpec}
//
//class SimpleAdder extends Module {
//  val io = new Bundle {
//    val a1 = FixedPoint(INPUT, 6, 4)
//    val a2 = FixedPoint(INPUT, 8, 1)
//    val c  = FixedPoint(INPUT, 12, 5)
//  }
//
//  val register1 = Reg(FixedPoint())
//
//  register1 := io.a1 + io.a2
//
//  io.c := register1
//}
//class SimpleAdderTester(c: SimpleAdder, backend: Option[Backend] = None) extends PeekPokeTester(c, _backend = backend) {
//  poke(c.io.a1.underlying, 4)
//  poke(c.io.a2.underlying, 0x30)
//  step(1)
//  println(s"peek ${peek(c.io.c.underlying)}")
//}
//class SimpleAdderSpec extends FlatSpec with Matchers {
//  //class SimpleAdderSpec extends FlatSpec with Matchers {
//  //  behavior of "Width inference"
//  //
//  //  def portAnnotator(ports: Seq[Port]): Unit = {
//  //    ports.foreach { port => port match {
//  //      case Port(b:Bundle, _) =>
//  //        println(s"Got bundle $b")
//  //        b.namedElts.foreach { case (name, e) =>
//  //          println(s"got element ${port.id}.$e")
//  //        }
//  //
//  //      case _ =>
//  //    }
//  //    }
//  //  }
//  //
//
//}
