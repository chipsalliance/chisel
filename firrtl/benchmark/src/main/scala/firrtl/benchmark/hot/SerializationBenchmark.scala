// SPDX-License-Identifier: Apache-2.0
package firrtl.benchmark.hot

import firrtl.benchmark.util._
import firrtl.ir.Serializer

object SerializationBenchmark extends App {
  val inputFile = args(0)
  val warmup = args(1).toInt
  val runs = args(2).toInt
  val select = if(args.length > 3) args(3) else "o"

  val input = filenameToCircuit(inputFile)

  if(select == "n") {
    println("Benchmarking new Serializer.serialize")
    firrtl.benchmark.hot.util.benchmark(warmup, runs)(Serializer.serialize(input))
  } else if(select == "o") {
    println("Benchmarking legacy serialization")
    firrtl.benchmark.hot.util.benchmark(warmup, runs)(input.serialize)
  } else if(select.startsWith("test")) {
    println("Testing the new serialization against the old one")
    val o = input.serialize.split('\n').filterNot(_.trim.isEmpty)
    val n = Serializer.serialize(input).split('\n').filterNot(_.trim.isEmpty)
    val silent = select.endsWith("silent")

    println(s"Old lines: ${o.length}")
    println(s"New lines: ${n.length}")
    o.zip(n).zipWithIndex.foreach { case ((ol, nl), ii) =>
      if(ol != nl) {
        println(s"❌@$ii OLD: |$ol|")
        println(s"❌@$ii NEW: |$nl|")
        throw new RuntimeException()
      } else if(!silent) {
        println(s"✅        |$ol")
      }
    }

  }
}
