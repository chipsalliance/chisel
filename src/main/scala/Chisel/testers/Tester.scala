/*
 Copyright (c) 2011, 2012, 2013, 2014 The Regents of the University of
 California (Regents). All Rights Reserved.  Redistribution and use in
 source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:

    * Redistributions of source code must retain the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      two paragraphs of disclaimer in the documentation and/or other materials
      provided with the distribution.
    * Neither the name of the Regents nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

 IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
 TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
*/

package Chisel.testers
import Chisel._
import scala.collection.mutable.{ArrayBuffer, HashMap, Queue => ScalaQueue}
import scala.collection.immutable.ListSet
import scala.util.Random
import java.io._
import java.lang.Double.{longBitsToDouble, doubleToLongBits}
import java.lang.Float.{intBitsToFloat, floatToIntBits}
import scala.sys.process.{Process, ProcessIO}

abstract class Tester[+T <: Module](c: T, isTrace: Boolean = true) extends FileSystemUtilities {
  private var _testIn: Option[InputStream] = None
  private var _testErr: Option[InputStream] = None
  private var _testOut: Option[OutputStream] = None
  private lazy val _reader: BufferedReader = new BufferedReader(new InputStreamReader(_testIn.get))
  private lazy val _writer: BufferedWriter = new BufferedWriter(new OutputStreamWriter(_testOut.get))
  private lazy val _logger: BufferedReader = new BufferedReader(new InputStreamReader(_testErr.get))
  var t = 0 // simulation time
  var delta = 0
  private val _pokeMap = HashMap[Bits, BigInt]()
  private val _peekMap = HashMap[Bits, BigInt]()
  private val _signalMap = HashMap[String, Int]()
   val _clocks:  List[(Clock, Int)]// = TesterDriver.clocks map (clk => clk -> clk.period.round.toInt)
  private val _clockLens = HashMap(_clocks:_*)
  private val _clockCnts = HashMap(_clocks:_*)
  val _inputs: ListSet[Bits]
  val _outputs: ListSet[Bits]
  //ListSet(c.wires.unzip._2: _*) partition (_.dir == INPUT)
  private var isStale = false
  val _logs = ScalaQueue[String]()
  def testOutputString = {
    if(_logs.isEmpty) "" else _logs.dequeue
  }

  object SIM_CMD extends Enumeration { val RESET, STEP, UPDATE, POKE, PEEK, GETID, SETCLK, FIN = Value }
  /**
   * Waits until the emulator streams are ready. This is a dirty hack related
   * to the way Process works. TODO: FIXME.
   */
  def waitForStreams() = {
    var waited = 0
    while (_testIn == None || _testOut == None || _testErr == None) {
      Thread.sleep(100)
      if (waited % 10 == 0 && waited > 30) {
        ChiselError.info("waiting for emulator process streams to be valid ...")
      }
    }
  }

  private def writeln(str: String) {
    _writer write str
    _writer.newLine
    _writer.flush
  }

  private def dumpLogs = {
    while (_logger.ready) {
      _logs enqueue _logger.readLine
    }
  }

  private def readln: String = {
    Option(_reader.readLine) match {
      case None =>
        dumpLogs
        while (!_logs.isEmpty)
          println(testOutputString)
        throw new Exception("Errors occurred in simulation")
      case Some(ln) => ln
    }
  }

  private def sendCmd(cmd: SIM_CMD.Value) {
    writeln(cmd.id.toString)
  }

  private val writeMask = int(-1L) 
  private def writeValue(v: BigInt, w: Int = 1) {
    for (i <- ((w - 1) >> 6) to 0 by -1) {
      writeln(((v >> (64 * i)) & writeMask).toString(16))
    }
  }

  def dumpName(data: Data): String = TesterDriver.backend match {
    case _: FloBackend => data.name
    case _: VerilogBackend => data.name
  }

  def setClock(clk: Clock, len: Int) {
    _clockLens(clk) = len
    _clockCnts(clk) = len
    sendCmd(SIM_CMD.SETCLK)
    writeln(clk.name)
    writeValue(len)
  }

  def setClocks(clocks: Iterable[(Clock, Int)]) {
    clocks foreach { case (clk, len) => setClock(clk, len) }
  }

  def signed_fix(dtype: Data, rv: BigInt): BigInt = {
    val w = dtype.getWidth
    dtype match {
      /* Any "signed" node */
      case _: SInt | _ : Flo | _: Dbl => (if(rv >= (BigInt(1) << w - 1)) (rv - (BigInt(1) << w)) else rv)
      /* anything else (i.e., UInt) */
      case _ => (rv)
    }
  }

  def peek(id: Int) = {
    sendCmd(SIM_CMD.PEEK)
    writeln(id.toString)
    try { BigInt(readln, 16) } catch { case e: Throwable => BigInt(0) }
  }
  def peekPath(path: String) = { 
    peek(_signalMap getOrElseUpdate (path, getId(path)))
  }
  def peekNode(node: Data, off: Option[Int] = None) = {
    peekPath(dumpName(node) + ((off map ("[" + _ + "]")) getOrElse ""))
  }
  //def peekAt[T <: Bits](data: Mem[T], off: Int): BigInt = {
  //  val value = peekNode(data, Some(off))
  //  if (isTrace) println("  PEEK %s[%d] -> %s".format(dumpName(data), off, value.toString(16)))
  //  value
  //}
  def peek(data: Bits): BigInt = {
    if (isStale) update
    val value = if (/*data.isTopLevelIO &&*/ data.dir == INPUT) _pokeMap(data)
                else signed_fix(data, _peekMap getOrElse (data, peekNode(data)))
    if (isTrace) println("  PEEK " + dumpName(data) + " -> " + value.toString(16))
    value
  }
  def peek(data: Aggregate): Array[BigInt] = {
    data.flatten.map (peek(_)).toArray
  }
  def peek(data: Flo): Float = {
    intBitsToFloat(peek(data.asInstanceOf[Bits]).toInt)
  }
  def peek(data: Dbl): Double = {
    longBitsToDouble(peek(data.asInstanceOf[Bits]).toLong)
  }


  def poke(id: Int, v: BigInt, w: Int = 1) {
    sendCmd(SIM_CMD.POKE)
    writeln(id.toString)
    writeValue(v, w)
  }
  def pokePath(path: String, v: BigInt, w: Int = 1) {
    poke(_signalMap getOrElseUpdate (path, getId(path)), v, w)
  }
  //def pokeNode(node: Node, v: BigInt, off: Option[Int] = None) {
  //  pokePath(dumpName(node) + ((off map ("[" + _ + "]")) getOrElse ""), v, node.needWidth)
  //}
  //def pokeAt[T <: Bits](data: Mem[T], value: BigInt, off: Int): Unit = {
  //  if (isTrace) println("  POKE %s[%d] <- %s".format(dumpName(data), off, value.toString(16)))
  //  pokeNode(data, value, Some(off))
  //}
  def poke(data: Bits, x: Boolean) { this.poke(data, int(x)) }
  def poke(data: Bits, x: Int)     { this.poke(data, int(x)) }
  def poke(data: Bits, x: Long)    { this.poke(data, int(x)) }
  def poke(data: Bits, x: BigInt)  {
    val value = if (x >= 0) x else {
      val cnt = (data.getWidth - 1) >> 6
      ((0 to cnt) foldLeft BigInt(0))((res, i) => res | (int((x >> (64 * i)).toLong) << (64 * i)))
    }
    if (isTrace) println("  POKE " + dumpName(data) + " <- " + value.toString(16))
    if (/*data.isTopLevelIO &&*/ data.dir == INPUT)
      _pokeMap(data) = value
    else if (/*data.isTopLevelIO &&*/ data.dir == OUTPUT)
      println("  NOT ALLOWED TO POKE OUTPUT " + dumpName(data))
    //else 
    //  pokeNode(data, value)
    isStale = true
  }
  def poke(data: Aggregate, x: Array[BigInt]): Unit = {
    val kv = (data.flatten, x.reverse).zipped
    for ((x, y) <- kv) poke(x, y)
  }
  def poke(data: Flo, x: Float): Unit = {
    poke(data.asInstanceOf[Bits], BigInt(floatToIntBits(x)))
  }
  def poke(data: Dbl, x: Double): Unit = {
    poke(data.asInstanceOf[Bits], BigInt(doubleToLongBits(x)))
  }

  def readOutputs {
    _peekMap.clear
    _outputs foreach (x => _peekMap(x) = try { BigInt(readln, 16) } catch { case e: Throwable => BigInt(0) })
  }

  def writeInputs {
    _inputs foreach (x => writeValue(_pokeMap getOrElse (x, BigInt(0)), x.getWidth))
  }

  def reset(n: Int = 1) {
    if (isTrace) println("RESET " + n)
    for (i <- 0 until n) {
      sendCmd(SIM_CMD.RESET)
      readOutputs
    }
  }

  protected def update {
    sendCmd(SIM_CMD.UPDATE)
    writeInputs
    readOutputs
    isStale = false
  }

  private def calcDelta = {
    val min = (_clockCnts.values foldLeft Int.MaxValue)(math.min(_, _))
    _clockCnts.keys foreach (_clockCnts(_) -= min)
    (_clockCnts filter (_._2 == 0)).keys foreach (k => _clockCnts(k) = _clockLens(k)) 
    min
  }

  protected def takeStep {
    sendCmd(SIM_CMD.STEP)
    writeInputs
    delta += calcDelta
    readOutputs
    dumpLogs
    isStale = false
  }

  protected def getId(path: String) = {
    sendCmd(SIM_CMD.GETID)
    writeln(path)
    readln.toInt
  }

  def step(n: Int) {
    if (isTrace) println("STEP " + n + " -> " + (t + n))
    (0 until n) foreach (_ => takeStep)
    t += n
  }

  def int(x: Boolean): BigInt = if (x) 1 else 0
  def int(x: Int):     BigInt = (BigInt(x >>> 1) << 1) | x & 1
  def int(x: Long):    BigInt = (BigInt(x >>> 1) << 1) | x & 1
  def int(x: Bits):    BigInt = x.litValue()

  var ok = true
  var failureTime = -1

  def expect (good: Boolean, msg: String): Boolean = {
    if (isTrace)
      println(msg + " " + (if (good) "PASS" else "FAIL"))
    if (!good) { ok = false; if (failureTime == -1) failureTime = t; }
    good
  }

  def expect (data: Bits, expected: BigInt): Boolean = {
    val mask = (BigInt(1) << data.getWidth) - 1
    val got = peek(data) & mask
    val exp = expected & mask
    expect(got == exp, "EXPECT " + dumpName(data) + " <- " + got.toString(16) + " == " + exp.toString(16))
  }

  def expect (data: Aggregate, expected: Array[BigInt]): Boolean = {
    val kv = (data.flatten, expected.reverse).zipped;
    var allGood = true
    for ((d, e) <- kv)
      allGood = expect(d, e) && allGood
    allGood
  }

  /* We need the following so scala doesn't use our "tolerant" Float version of expect.
   */
  def expect (data: Bits, expected: Int): Boolean = {
    expect(data, int(expected))
  }
  def expect (data: Bits, expected: Long): Boolean = {
    expect(data, int(expected))
  }
  def expect (data: Flo, expected: Float): Boolean = {
    val got = peek(data)
    expect(got == expected, "EXPECT " + dumpName(data) + " <- " + got + " == " + expected)
  }
  def expect (data: Dbl, expected: Double): Boolean = {
    val got = peek(data)
    expect(got == expected, "EXPECT " + dumpName(data) + " <- " + got + " == " + expected)
  }

  /* Compare the floating point value of a node with an expected floating point value.
   * We will tolerate differences in the bottom bit.
   */
  def expect (data: Bits, expected: Float): Boolean = {
    val gotBits = peek(data).toInt
    val expectedBits = java.lang.Float.floatToIntBits(expected)
    var gotFLoat = java.lang.Float.intBitsToFloat(gotBits)
    var expectedFloat = expected
    if (gotFLoat != expectedFloat) {
      val gotDiff = gotBits - expectedBits
      // Do we have a single bit difference?
      if (scala.math.abs(gotDiff) <= 1) {
        expectedFloat = gotFLoat
      }
    }
    expect(gotFLoat == expectedFloat,
       "EXPECT " + dumpName(data) + " <- " + gotFLoat + " == " + expectedFloat)
  }

  val rnd = if (TesterDriver.testerSeedValid) new Random(TesterDriver.testerSeed) else new Random()
  val process: Process = {
    val n = TesterDriver.name
    val target = TesterDriver.targetDir + "/" + n
    // If the caller has provided a specific command to execute, use it.
    val cmd = TesterDriver.testCommand match {
      case Some(cmd) => TesterDriver.targetDir + "/" + cmd
      case None => TesterDriver.backend match {
        case b: FloBackend => target
          /*val command = ArrayBuffer(b.floDir + "fix-console", ":is-debug", "true", ":filename", target + ".hex", ":flo-filename", target + ".mwe.flo")
          if (TesterDriver.isVCD) { command ++= ArrayBuffer(":is-vcd-dump", "true") }
          if (TesterDriver.emitTempNodes) { command ++= ArrayBuffer(":emit-temp-nodes", "true") }
          command ++= ArrayBuffer(":target-dir", TesterDriver.targetDir)
          command.mkString(" ")*/
        case b: VerilogBackend => target + " -q +vcs+initreg+0 "
        case _ => target
      }
    }
    println("SEED " + TesterDriver.testerSeed)
    println("STARTING " + cmd)
    val processBuilder = Process(cmd)
    val pio = new ProcessIO(
      in => _testOut = Option(in), out => _testErr = Option(out), err => _testIn = Option(err))
    val process = processBuilder.run(pio)
    waitForStreams()
    t = 0
    readOutputs
    reset(5)
    while (_logger.ready) println(_logger.readLine)
    process
  }

  def finish {
    sendCmd(SIM_CMD.FIN)
    _testIn match { case Some(in) => in.close case None => }
    _testErr match { case Some(err) => err.close case None => }
    _testOut match { case Some(out) => { out.flush ; out.close } case None => }
    process.destroy()
    println("RAN " + t + " CYCLES " + (if (ok) "PASSED" else "FAILED FIRST AT CYCLE " + failureTime))
    if(!ok) throwException("Module under test FAILED at least one test vector.")
  }

  //_signalMap ++= TesterDriver.signalMap flatMap {
  //  case (m: Mem[_], id) => 
  //    (0 until m.n) map (idx => "%s[%d]".format(dumpName(m), idx) -> (id + idx))
  //  case (node, id) => Seq(dumpName(node) -> id)
  //}
}
