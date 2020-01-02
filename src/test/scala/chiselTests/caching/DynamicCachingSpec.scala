package chiselTests.caching

import java.io.{ByteArrayOutputStream, FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation}
import com.twitter.chill.MeatLocker



/*
case class CacheableRecursiveModule(depth: Int, maxDepth: Int, nchild: Int) extends Cacheable[CacheableRecursiveModuleImpl] {
  override def buildImpl: CacheableRecursiveModuleImpl = new CacheableRecursiveModuleImpl(depth, maxDepth, nchild)
}

class CacheableRecursiveModuleImpl(depth: Int, maxDepth: Int, nchild: Int) extends MultiIOModule {

  //override def cacheTag = createTag(List(classOf[RecursiveModule], depth, maxDepth, nchild))

  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  val x = (1 until 10000).foldLeft(0)((a, b) => a + b)

  val reg = RegNext(in)

  val ret = if(depth < maxDepth) {
    (0 until nchild).foldLeft(reg) { (signal, i) =>
      val inst = Module(new CacheableRecursiveModuleImpl(depth + 1, maxDepth, nchild)).suggestName(s"inst$i")
      inst.in := signal
      inst.out
    }
  } else {
    reg
  }

  out := ret
}

class RecursiveModule(depth: Int, maxDepth: Int, nchild: Int) extends MultiIOModule {

  //override def cacheTag = createTag(List(classOf[RecursiveModule], depth, maxDepth, nchild))

  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  val x = (1 until 10000).foldLeft(0)((a, b) => a + b)

  val reg = RegNext(in)

  val ret = if(depth < maxDepth) {
    (0 until nchild).foldLeft(reg) { (signal, i) =>
      val inst = Module(new RecursiveModule(depth + 1, maxDepth, nchild)).suggestName(s"inst$i")
      inst.in := signal
      inst.out
    }
  } else {
    reg
  }

  out := ret
}

class DynamicCachingSpec extends ChiselFlatSpec {
  "Ser/des with kryo MeatLocker with Chisel" should "work" in {
    val (elaborateTime, elaboratedDut) = firrtl.Utils.time(
      ChiselGeneratorAnnotation(() => Module(new RecursiveModule(0, 13, 2))).reload
    )

    val (writingTime, _) = firrtl.Utils.time{
      val oos = new ObjectOutputStream(new FileOutputStream("/tmp/cacheMeatlocker"))
      oos.writeObject(MeatLocker(elaboratedDut))
      oos.close
    }

    val (reloadTime, reloadedDut) =  firrtl.Utils.time(
      ChiselGeneratorAnnotation(() => {
        val ois = new ObjectInputStream(new FileInputStream("/tmp/cacheMeatlocker"))
        val obj = ois.readObject.asInstanceOf[MeatLocker[RecursiveModule]]
        ois.close
        obj.get
      }).reload.asInstanceOf[RecursiveModule]
    )

    println(s"Elaborating: $elaborateTime")
    println(s"Writing: $writingTime")
    println(s"Reloading: $reloadTime")
    println("Done")

    assert(reloadTime < elaborateTime)
  }

  "Determining cache tag prior to elaboration" should "be possible" in {

    /*
    def elaborate(): RecursiveModule = {
      val gen = () => new RecursiveModule(0, 13, 2)
      val tag = getTag(gen)
      println(s"Elaboration tag: $tag")
      val oos = new ObjectOutputStream(new FileOutputStream(s"/tmp/$tag.cache"))
      val elaboratedDut = ChiselGeneratorAnnotation(() => {
        Module(gen())
      }).reload
      oos.writeObject(MeatLocker(elaboratedDut))
      oos.close
      elaboratedDut.asInstanceOf[RecursiveModule]
    }

    def reload(): RecursiveModule = {
      ChiselGeneratorAnnotation(() => {
        val gen = () => new RecursiveModule(0, 13, 2)
        val tag = getTag(gen)
        println(s"Reloading tag: $tag")
        val ois = new ObjectInputStream(new FileInputStream(s"/tmp/$tag.cache"))
        val obj = ois.readObject.asInstanceOf[MeatLocker[RecursiveModule]]
        ois.close
        obj.get
      }).reload.asInstanceOf[RecursiveModule]
    }

    elaborate()
    reload()
    println("Done")
     */
  }

  /*
  "Explicit cache gets/sets" should "work" in {
    //val gen = () => new RecursiveModule(0, 13, 2)
    println("Starting")
    val elaboratedDut = ChiselGeneratorAnnotation(() => {
      CacheCheck(new CacheableRecursiveModule(0, 4, 2))
    }).elaborateWithCache(ChiselCacheAnnotation(Some("/tmp/"), Some("/tmp/")))
    elaboratedDut
  }
   */

}

 */
