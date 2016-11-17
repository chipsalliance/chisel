// See LICENSE for license details.

package chisel3.iotesters

import java.io.File

import chisel3._

import scala.collection.immutable.ListMap
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

// Provides a template to define tester transactions
trait PeekPokeTests {
  def t: Long
  def rnd: scala.util.Random
  implicit def int(x: Boolean): BigInt
  implicit def int(x: Int):     BigInt
  implicit def int(x: Long):    BigInt
  implicit def int(x: Bits):    BigInt
  def println(msg: String = ""): Unit
  def reset(n: Int): Unit
  def step(n: Int): Unit
  def poke(path: String, x: BigInt): Unit
  def peek(path: String): BigInt
  def poke(signal: Bits, x: BigInt): Unit
  def pokeAt[T <: Bits](signal: Mem[T], x: BigInt, off: Int): Unit
  def peek(signal: Bits): BigInt
  def peekAt[T <: Bits](signal: Mem[T], off: Int): BigInt
  def expect(good: Boolean, msg: => String): Boolean
  def expect(signal: Bits, expected: BigInt, msg: => String = ""): Boolean
  def finish: Boolean
}

abstract class PeekPokeTester[+T <: Module](
    val dut: T,
    base: Int = 16,
    logFile: Option[File] = None) {

  implicit def longToInt(x: Long) = x.toInt
  val optionsManager = Driver.optionsManager

  implicit val logger = (logFile, optionsManager.testerOptions.logFileName) match {
    case (None, "")        => System.out
    case (Some(f), _)      => new java.io.PrintStream(f)
    case (_, logFileName)  => new java.io.PrintStream(new File(logFileName))
  }
  implicit val _verbose = optionsManager.testerOptions.isVerbose
  implicit val _base    = optionsManager.testerOptions.displayBase

  def println(msg: String = "") {
    logger.println(msg)
  }

  /****************************/
  /*** Simulation Interface ***/
  /****************************/
  val backend = Driver.backend.get

  /********************************/
  /*** Classic Tester Interface ***/
  /********************************/
  /* Simulation Time */
  private var simTime = 0L
  protected[iotesters] def incTime(n: Int) { simTime += n }
  def t = simTime

  /** Indicate a failure has occurred.  */
  private var failureTime = -1L
  private var ok = true
  def fail = if (ok) {
    failureTime = simTime
    ok = false
  }

  val rnd = backend.rnd
  rnd.setSeed(optionsManager.testerOptions.testerSeed)
  logger.println(s"SEED ${optionsManager.testerOptions.testerSeed}")

  /** Convert a Boolean to BigInt */
  implicit def int(x: Boolean): BigInt = if (x) 1 else 0
  /** Convert an Int to BigInt */
  implicit def int(x: Int):     BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  /** Convert a Long to BigInt */
  implicit def int(x: Long):    BigInt = (BigInt(x >>> 1) << 1) | BigInt(x & 1)
  /** Convert Bits to BigInt */
  implicit def int(x: Bits):    BigInt = x.litValue()

  def reset(n: Int = 1) {
    backend.reset(n)
  }

  def step(n: Int) {
    if (_verbose) logger println s"STEP $simTime -> ${simTime+n}"
    backend.step(n)
    incTime(n)
  }

  def poke(path: String, value: BigInt) = backend.poke(path, value)

  def peek(path: String) = backend.peek(path)

  def poke(signal: Bits, value: BigInt) {
    if (!signal.isLit) backend.poke(signal, value, None)
    // TODO: Warn if signal.isLit
  }

  /** Locate a specific bundle element, given a name path.
    * TODO: Handle Vecs
    *
    * @param path - list of element names (presumably bundles) terminating in a non-bundle (i.e., Bits) element.
    * @param bundle - bundle containing the element
    * @return the element (as Bits)
    */
  private def getBundleElement(path: List[String], bundle: ListMap[String, Data]): Bits = {
    (path, bundle(path.head)) match {
      case (head :: Nil, element: Bits) => element
      case (head :: tail, b: Bundle) => getBundleElement(tail, b.elements)
      case _ => throw new Exception(s"peek/poke bundle element mismatch $path")
    }
  }

  /** Poke a Bundle given a map of elements and values.
    *
    * @param signal the bundle to be poked
    * @param map a map from names (using '.' to delimit bundle elements), to BigInt values
    */
  def poke(signal: Bundle, map: Map[String, BigInt]): Unit =  {
    val circuitElements = signal.elements
    for ( (key, value) <- map) {
      val subKeys = key.split('.').toList
      val element = getBundleElement(subKeys, circuitElements)
      poke(element, value)
    }
  }

  def poke(signal: Aggregate, value: IndexedSeq[BigInt]): Unit =  {
    (signal.flatten zip value.reverse).foreach(x => poke(x._1, x._2))
  }

  def pokeAt[TT <: Bits](data: Mem[TT], value: BigInt, off: Int): Unit = {
    backend.poke(data, value, Some(off))
  }

  def peek(signal: Bits):BigInt = {
    if (!signal.isLit) backend.peek(signal, None) else signal.litValue()
  }

  def peek(signal: Aggregate): IndexedSeq[BigInt] =  {
    signal.flatten map (x => backend.peek(x, None))
  }

  /** Populate a map of names ("dotted Bundles) to Bits.
    * TODO: Deal with Vecs
    *
    * @param map the map to be constructed
    * @param indexPrefix an array of Bundle name prefixes
    * @param signalName the signal to be added to the map
    * @param signalData the signal object to be added to the map
    */
  private def setBundleElement(map: mutable.LinkedHashMap[String, Bits], indexPrefix: ArrayBuffer[String], signalName: String, signalData: Data): Unit = {
    indexPrefix += signalName
    signalData match {
      case bundle: Bundle =>
        for ((name, value) <- bundle.elements) {
          setBundleElement(map, indexPrefix, name, value)
        }
      case bits: Bits =>
        val index = indexPrefix.mkString(".")
        map(index) = bits
    }
    indexPrefix.remove(indexPrefix.size - 1)
  }

  /** Peek an aggregate (Bundle) signal.
    *
    * @param signal the signal to peek
    * @return a map of signal names ("dotted" Bundle) to BigInt values.
    */
  def peek(signal: Bundle): mutable.LinkedHashMap[String, BigInt] = {
    val bitsMap = mutable.LinkedHashMap[String, Bits]()
    val index = ArrayBuffer[String]()
    // Populate the Bits map.
    for ((elementName, elementValue) <- signal.elements) {
      setBundleElement(bitsMap, index, elementName, elementValue)
    }
    val bigIntMap = mutable.LinkedHashMap[String, BigInt]()
    for ((name, bits) <- bitsMap) {
      bigIntMap(name) = peek(bits)
    }
    bigIntMap
  }

  def peekAt[TT <: Bits](data: Mem[TT], off: Int): BigInt = {
    backend.peek(data, Some(off))
  }

  def expect (good: Boolean, msg: => String): Boolean = {
    if (_verbose || ! good) logger println s"""EXPECT AT $simTime $msg ${if (good) "PASS" else "FAIL"}"""
    if (!good) fail
    good
  }

  def expect(signal: Bits, expected: BigInt, msg: => String = ""): Boolean = {
    if (!signal.isLit) {
      val good = backend.expect(signal, expected, msg)
      if (!good) fail
      good
    } else expect(signal.litValue() == expected, s"${signal.litValue()} == $expected")
  }

  def expect (signal: Aggregate, expected: IndexedSeq[BigInt]): Boolean = {
    (signal.flatten, expected.reverse).zipped.foldLeft(true) { (result, x) => result && expect(x._1, x._2)}
  }

  /** Return true or false if an aggregate signal (Bundle) matches the expected map of values.
    * TODO: deal with Vecs
    *
    * @param signal the Bundle to "expect"
    * @param expected a map of signal names ("dotted" Bundle notation) to BigInt values
    * @return true if the specified values match, false otherwise.
    */
  def expect (signal: Bundle, expected: Map[String, BigInt]): Boolean = {
    val bitsMap = mutable.LinkedHashMap[String, Bits]()
    val index = ArrayBuffer[String]()
    for ((elementName, elementValue) <- signal.elements) {
      setBundleElement(bitsMap, index, elementName, elementValue)
    }
    expected.forall{ case ((name, value)) => expect(bitsMap(name), value) }
  }

  def finish: Boolean = {
    try {
      backend.finish
    } catch {
      // Depending on load and timing, we may get a TestApplicationException
      //  when the test application exits.
      //  Check the exit value.
      //  Anything other than 0 is an error.
      case e: TestApplicationException => if (e.exitVal != 0) fail
    }
    logger println s"""RAN $simTime CYCLES ${if (ok) "PASSED" else s"FAILED FIRST AT CYCLE $failureTime"}"""
    ok
  }
}
