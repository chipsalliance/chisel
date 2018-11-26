package chiselTests.benchmarks

import chisel3._
import org.scalameter.api._

import scala.collection.immutable.{HashMap, ListMap}
import scala.util.Random


class MyRecord(size: Int) extends Record {
  def key(i: Int): String = "elt_" + i.toString

  val builder = ListMap.newBuilder[String, Data]
  for (i <- 0 until size) {
    builder += key(i) -> 0.U(32.W)
  }
  val elements = builder.result

  override def cloneType: MyRecord.this.type = new MyRecord(size).asInstanceOf[this.type]

  def randomElement: Data = elements(key(Random.nextInt(size)))
}

class MyFastRecord(size: Int) extends FastRecord {
  def key(i: Int): String = "elt_" + i.toString

  val orderBuilder = List.newBuilder[String]
  val hashBuilder = HashMap.newBuilder[String, Data]
  for (i <- 0 until size) {
    orderBuilder += key(i)
    hashBuilder += key(i) -> 0.U(32.W)
  }

  val elementsOrder = orderBuilder.result
  val elementsHash = hashBuilder.result

  override def cloneType: MyFastRecord.this.type = new MyFastRecord(size).asInstanceOf[this.type]

  def randomElement: Data = elementsHash(key(Random.nextInt(size)))
}


object RecordBench extends Bench.LocalTime {
  val sizes = Gen.range("size")(100, 2000, 100)
  val records = for {size <- sizes} yield new MyRecord(size)
  val fastRecords = for {size <- sizes} yield new MyFastRecord(size)

  performance of "Record" in {
    measure method "instantiate" in {
      using(sizes) in {
        s => new MyRecord(s)
      }
    }
    measure method "getElements" in {
      using(records) in {
        r => r.getElements
      }
    }
    measure method "random access" in {
      using(records) in {
        r => for (i <- 0 until 100) {
          r.randomElement
        }
      }
    }
  }

  performance of "FastRecord" in {
    measure method "instantiate" in {
      using(sizes) in {
        s => new MyFastRecord(s)
      }
    }
    measure method "getElements" in {
      using(fastRecords) in {
        r => r.getElements
      }
    }
    measure method "random access" in {
      using(fastRecords) in {
        r => for (i <- 0 until 100) {
          r.randomElement
        }
      }
    }
  }
}

