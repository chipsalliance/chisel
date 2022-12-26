package chisel3.std.tilelink

import chisel3._
import chisel3.internal.firrtl.Width

/**
  * opcode field of TileLink messages
  *
  *  cf. TileLink Spec 1.8.1 Table 13
  */
object OpCode {
  protected[tilelink] val width: Width = 3.W

  val PutFullData:    UInt = 0.U(width)
  val PutPartialData: UInt = 1.U(width)
  val ArithmeticData: UInt = 2.U(width)
  val LogicalData:    UInt = 3.U(width)
  val Get:            UInt = 4.U(width)
  val Intent:         UInt = 5.U(width)
  val AcquireBlock:   UInt = 6.U(width)
  val AcquirePerm:    UInt = 7.U(width)
  val ProbePerm:      UInt = 7.U(width)
  val AccessAck:      UInt = 0.U(width)
  val AccessAckData:  UInt = 1.U(width)
  val HintAck:        UInt = 2.U(width)
  val ProbeAck:       UInt = 4.U(width)
  val ProbeAckData:   UInt = 5.U(width)
  val Release:        UInt = 6.U(width)
  val ReleaseData:    UInt = 7.U(width)
  val Grant:          UInt = 4.U(width)
  val GrantData:      UInt = 5.U(width)
  val ProbeBlock:     UInt = 6.U(width)
  val ReleaseAck:     UInt = 6.U(width)
  // GrantAck has no Opcode
}

/**
  * param field of TileLink messages
  *
  * Allowed values are dependent on opcode of the message.
  */
object Param {
  protected[tilelink] val width: Width = 3.W

  /**
    * Reserved param fields (e.g. a_param in Get message) should be tied to zero
    */
  def tieZero: UInt = 0.U(width)

  /** TileLink Spec 1.8.1 Table 23
    */
  object Arithmetic {
    val MIN:  UInt = 0.U(width)
    val MAX:  UInt = 1.U(width)
    val MINU: UInt = 2.U(width)
    val MAXU: UInt = 3.U(width)
    val ADD:  UInt = 4.U(width)
  }

  /** TileLink Spec 1.8.1 Table 25
    */
  object Logical {
    val XOR:  UInt = 0.U(width)
    val OR:   UInt = 1.U(width)
    val AND:  UInt = 2.U(width)
    val SWAP: UInt = 3.U(width)
  }

  /** TileLink Spec 1.8.1 Table 27 Intent
    */
  object Intent {
    val PrefetchRead:  UInt = 0.U(width)
    val PrefetchWrite: UInt = 1.U(width)
  }

  /** TileLink Spec 1.8.1 Table 31 Cap
    */
  object Cap {
    val toT: UInt = 0.U(width)
    val toB: UInt = 1.U(width)
    val toN: UInt = 2.U(width)
  }

  /** TileLink Spec 1.8.1 Table 31 Grow
    */
  object Grow {
    val NtoB: UInt = 0.U(width)
    val NtoT: UInt = 1.U(width)
    val BtoT: UInt = 2.U(width)
  }

  /** TileLink Spec 1.8.1 Table 31 Prune
    */
  object Prune {
    val TtoB: UInt = 0.U(width)
    val TtoN: UInt = 1.U(width)
    val BtoN: UInt = 2.U(width)
  }

  /** TileLink Spec 1.8.1 Table 31 Report
    */
  object Report {
    val TtoT: UInt = 3.U(width)
    val BtoB: UInt = 4.U(width)
    val NtoN: UInt = 5.U(width)
  }
}
