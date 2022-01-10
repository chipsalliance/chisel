package chiselTests

import java.io.{BufferedReader, File, FileReader, InputStream}

import chisel3._
import chisel3.util.experimental.{Tracker, TrackerAnnotation}

class TrackerSpec extends ChiselFlatSpec with ChiselRunners {

  behavior of "Tracker.track"

  class DCEInverter extends Module {
    val io = IO(new Bundle{
      val in = Input(UInt(1.W))
      val out = Output(UInt(1.W))
    })
    io.out := io.in
    val notA = Wire(UInt(1.W))
    notA := DontCare
    Tracker.track(notA)
    Tracker.track(io)
  }

  it should "track DCE'd wires and lowering ios" in {
    val annos = runChiselStage(new DCEInverter)
    val trackers = annos.collect { case t: TrackerAnnotation => t }.map(_.serialize)
    trackers should contain ("<DELETED> <-- ~DCEInverter|DCEInverter>notA")
    trackers should contain ("~DCEInverter|DCEInverter>io_in <-- ~DCEInverter|DCEInverter>io")
    trackers should contain ("~DCEInverter|DCEInverter>io_out <-- ~DCEInverter|DCEInverter>io")
  }

  class TwoInverters extends Module {
    val io = IO(new Bundle{
      val in = Input(UInt(1.W))
      val out = Output(UInt(1.W))
    })
    val i0 = Module(new DCEInverter)
    val i1 = Module(new DCEInverter)
    i0.io.in := io.in
    i1.io.in := i0.io.out
    io.out := i1.io.out
    Tracker.track(i0)
    Tracker.track(i1)
  }

  it should "track deduped modules" in {
    val annos = runChiselStage(new TwoInverters)
    val trackers = annos.collect { case t: TrackerAnnotation => t }.map(_.serialize)
    trackers should contain ("<DELETED> <-- ~TwoInverters|DCEInverter>notA <-- ~TwoInverters|DCEInverter_1>notA")
    trackers should contain ("<DELETED> <-- ~TwoInverters|DCEInverter>notA")
    trackers should contain ("~TwoInverters|DCEInverter>io_in <-- ~TwoInverters|DCEInverter>io <-- ~TwoInverters|DCEInverter_1>io")
    trackers should contain ("~TwoInverters|DCEInverter>io_out <-- ~TwoInverters|DCEInverter>io <-- ~TwoInverters|DCEInverter_1>io")
    trackers should contain ("~TwoInverters|DCEInverter>io_in <-- ~TwoInverters|DCEInverter>io")
    trackers should contain ("~TwoInverters|DCEInverter>io_out <-- ~TwoInverters|DCEInverter>io")
    trackers should contain ("~TwoInverters|DCEInverter <-- ~TwoInverters|DCEInverter_1")
    trackers should contain ("~TwoInverters|DCEInverter")
  }

}
