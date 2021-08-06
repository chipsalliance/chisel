package chiselTests.hierarchy

import chisel3._
import chisel3.stage.{ChiselStage, DesignAnnotation, ChiselGeneratorAnnotation}
import _root_.firrtl.annotations._

trait Utils extends chiselTests.Utils {
  import Annotations._
  def elaborate[T <: RawModule](bc: => T): T = {
    ChiselGeneratorAnnotation(() => bc).elaborate.collectFirst {
      case d: DesignAnnotation[T] => d.design
    }.get.asInstanceOf[T]
  }
  def check(bc: => RawModule, target: IsMember, tag: String, printOutput: Boolean = false): Unit = {
    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = bc, args = Array("--full-stacktrace"))
    if(printOutput) println(output)
    val anno = MarkAnnotation(target, tag)
    assert(annotations.toSeq.contains(anno), s"${annotations.toSeq} does not contain $anno!")
  }
  def check(bc: => RawModule, instance: String, of: String): Unit = {
    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = bc, args = Array("--full-stacktrace"))
    val insts = output.split("\n").map(_.split("\\s+").toList).collect {
      case _ :: "inst" :: x :: "of" :: y :: rest => (x, y)
    }
    assert(insts.contains((instance, of)), s"""Output does not contain $instance of $of! Contains the following:${"\n"}${insts.mkString("\n")}""")
  }
  def check(bc: => RawModule, targets: Seq[(IsMember, String)]): Unit = {
    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = bc, args = Array("--full-stacktrace"))
    targets.foreach { case (target, tag) =>
      val anno = MarkAnnotation(target, tag)
      assert(annotations.toSeq.contains(anno), s"${annotations.toSeq} does not contain $anno!")
    }
  }
  implicit class Str2RefTarget(str: String) {
    def rt = Target.deserialize(str).asInstanceOf[ReferenceTarget]
    def it = Target.deserialize(str).asInstanceOf[InstanceTarget]
    def mt = Target.deserialize(str).asInstanceOf[ModuleTarget]
    def ct = Target.deserialize(str).asInstanceOf[CircuitTarget]
  }
}