// SPDX-License-Identifier: Apache-2.0

package chiselTests.util.experimental.minimizer

import chisel3._
import chisel3.util._
import chisel3.util.experimental.decode._
import chisel3.util.pla
import chiseltest._
import chiseltest.formal._
import org.scalatest.flatspec.AnyFlatSpec

class DecodeTestModule(minimizer: Minimizer, table: TruthTable) extends Module {
  val i = IO(Input(UInt(table.table.head._1.getWidth.W)))
  val (unminimizedI, unminimizedO) = pla(table.table.toSeq)
  unminimizedI := i
  val minimizedO: UInt = decoder(minimizer, i, table)

  chisel3.experimental.verification.assert(
    // for each instruction, if input matches, output should match, not no matched, fallback to default
    (table.table.map { case (key, value) => (i === key) && (minimizedO === value) } ++
      Seq(table.table.map(_._1).map(i =/= _).reduce(_ && _) && minimizedO === table.default)).reduce(_ || _)
  )
}

trait MinimizerSpec extends AnyFlatSpec with ChiselScalatestTester with Formal {
  def minimizer: Minimizer

  def minimizerTest(testcase: TruthTable) = {
    verify(new DecodeTestModule(minimizer, table = testcase), Seq(BoundedCheck(1)))
  }

  // Term that being commented out is the result of which is same as default,
  // making optimization opportunities to decoder algorithms

  "case0" should "pass" in {
    minimizerTest(TruthTable(
      Map(
        // BitPat("b000") -> BitPat("b0"),
        BitPat("b001") -> BitPat("b?"),
        BitPat("b010") -> BitPat("b?"),
        // BitPat("b011") -> BitPat("b0"),
        BitPat("b100") -> BitPat("b1"),
        BitPat("b101") -> BitPat("b1"),
        // BitPat("b110") -> BitPat("b0"),
        BitPat("b111") -> BitPat("b1")
      ),
      BitPat("b0")
    ))
  }

  "case1" should "pass" in {
    minimizerTest(TruthTable(
      Map(
        BitPat("b000") -> BitPat("b0"),
        BitPat("b001") -> BitPat("b?"),
        BitPat("b010") -> BitPat("b?"),
        BitPat("b011") -> BitPat("b0"),
        // BitPat("b100") -> BitPat("b1"),
        // BitPat("b101") -> BitPat("b1"),
        BitPat("b110") -> BitPat("b0"),
        // BitPat("b111") -> BitPat("b1")
      ),
      BitPat("b1")
    ))
  }

  "caseX" should "pass" in {
    minimizerTest(TruthTable(
      Map(
        BitPat("b000") -> BitPat("b0"),
        // BitPat("b001") -> BitPat("b?"),
        // BitPat("b010") -> BitPat("b?"),
        BitPat("b011") -> BitPat("b0"),
        BitPat("b100") -> BitPat("b1"),
        BitPat("b101") -> BitPat("b1"),
        BitPat("b110") -> BitPat("b0"),
        BitPat("b111") -> BitPat("b1")
      ),
      BitPat("b?")
    ))
  }

  "caseMultiDefault" should "pass" in {
    minimizerTest(TruthTable(
      Map(
        BitPat("b000") -> BitPat("b0100"),
        BitPat("b001") -> BitPat("b?111"),
        BitPat("b010") -> BitPat("b?000"),
        BitPat("b011") -> BitPat("b0101"),
        BitPat("b111") -> BitPat("b1101")
      ),
      BitPat("b?100")
    ))
  }

  "case7SegDecoder" should "pass" in {
    minimizerTest(TruthTable(
      Map(
        BitPat("b0000") -> BitPat("b111111001"),
        BitPat("b0001") -> BitPat("b011000001"),
        BitPat("b0010") -> BitPat("b110110101"),
        BitPat("b0011") -> BitPat("b111100101"),
        BitPat("b0100") -> BitPat("b011001101"),
        BitPat("b0101") -> BitPat("b101101101"),
        BitPat("b0110") -> BitPat("b101111101"),
        BitPat("b0111") -> BitPat("b111000001"),
        BitPat("b1000") -> BitPat("b111111101"),
        BitPat("b1001") -> BitPat("b111101101"),
      ),
      BitPat("b???????10")
    ))
  }

  // A simple RV32I decode table example
  "caseRV32I" should "pass" in {
    val BEQ = "?????????????????000?????1100011"
    val BNE = "?????????????????001?????1100011"
    val BLT = "?????????????????100?????1100011"
    val BGE = "?????????????????101?????1100011"
    val BLTU = "?????????????????110?????1100011"
    val BGEU = "?????????????????111?????1100011"
    val JALR = "?????????????????000?????1100111"
    val JAL = "?????????????????????????1101111"
    val LUI = "?????????????????????????0110111"
    val AUIPC = "?????????????????????????0010111"
    val ADDI = "?????????????????000?????0010011"
    val SLTI = "?????????????????010?????0010011"
    val SLTIU = "?????????????????011?????0010011"
    val XORI = "?????????????????100?????0010011"
    val ORI = "?????????????????110?????0010011"
    val ANDI = "?????????????????111?????0010011"
    val ADD = "0000000??????????000?????0110011"
    val SUB = "0100000??????????000?????0110011"
    val SLL = "0000000??????????001?????0110011"
    val SLT = "0000000??????????010?????0110011"
    val SLTU = "0000000??????????011?????0110011"
    val XOR = "0000000??????????100?????0110011"
    val SRL = "0000000??????????101?????0110011"
    val SRA = "0100000??????????101?????0110011"
    val OR = "0000000??????????110?????0110011"
    val AND = "0000000??????????111?????0110011"
    val LB = "?????????????????000?????0000011"
    val LH = "?????????????????001?????0000011"
    val LW = "?????????????????010?????0000011"
    val LBU = "?????????????????100?????0000011"
    val LHU = "?????????????????101?????0000011"
    val SB = "?????????????????000?????0100011"
    val SH = "?????????????????001?????0100011"
    val SW = "?????????????????010?????0100011"
    val FENCE = "?????????????????000?????0001111"
    val MRET = "00110000001000000000000001110011"
    val WFI = "00010000010100000000000001110011"
    val CEASE = "00110000010100000000000001110011"
    val CSRRW = "?????????????????001?????1110011"
    val CSRRS = "?????????????????010?????1110011"
    val CSRRC = "?????????????????011?????1110011"
    val CSRRWI = "?????????????????101?????1110011"
    val CSRRSI = "?????????????????110?????1110011"
    val CSRRCI = "?????????????????111?????1110011"
    val SCALL = "00000000000000000000000001110011"
    val SBREAK = "00000000000100000000000001110011"
    val SLLI_RV32 = "0000000??????????001?????0010011"
    val SRLI_RV32 = "0000000??????????101?????0010011"
    val SRAI_RV32 = "0100000??????????101?????0010011"

    val A1_X = "??"
    val A1_ZERO = "00"
    val A1_RS1 = "01"
    val A1_PC = "10"

    val IMM_X = "???"
    val IMM_S = "000"
    val IMM_SB = "001"
    val IMM_U = "010"
    val IMM_UJ = "011"
    val IMM_I = "100"
    val IMM_Z = "101"

    val A2_X = "??"
    val A2_ZERO = "00"
    val A2_SIZE = "01"
    val A2_RS2 = "10"
    val A2_IMM = "11"

    val X = "?"
    val N = "0"
    val Y = "1"

    val DW_X = X
    val DW_XPR = Y

    val M_X = "?????"
    val M_XRD = "00000"
    val M_XWR = "00001"

    val CSR_X = "???"
    val CSR_N = "000"
    val CSR_I = "100"
    val CSR_W = "101"
    val CSR_S = "110"
    val CSR_C = "111"

    val FN_X = "????"
    val FN_ADD = "0000"
    val FN_SL = "0001"
    val FN_SEQ = "0010"
    val FN_SNE = "0011"
    val FN_XOR = "0100"
    val FN_SR = "0101"
    val FN_OR = "0110"
    val FN_AND = "0111"
    val FN_SUB = "1010"
    val FN_SRA = "1011"
    val FN_SLT = "1100"
    val FN_SGE = "1101"
    val FN_SLTU = "1110"
    val FN_SGEU = "1111"

    minimizerTest(TruthTable(
      Map(
        BNE -> Seq(Y, N, N, Y, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_SB, DW_X, FN_SNE, N, M_X, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        BEQ -> Seq(Y, N, N, Y, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_SB, DW_X, FN_SEQ, N, M_X, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        BLT -> Seq(Y, N, N, Y, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_SB, DW_X, FN_SLT, N, M_X, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        BLTU -> Seq(Y, N, N, Y, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_SB, DW_X, FN_SLTU, N, M_X, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        BGE -> Seq(Y, N, N, Y, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_SB, DW_X, FN_SGE, N, M_X, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        BGEU -> Seq(Y, N, N, Y, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_SB, DW_X, FN_SGEU, N, M_X, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        JAL -> Seq(Y, N, N, N, Y, N, N, N, N, A2_SIZE, A1_PC, IMM_UJ, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        JALR -> Seq(Y, N, N, N, N, Y, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        AUIPC -> Seq(Y, N, N, N, N, N, N, N, N, A2_IMM, A1_PC, IMM_U, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        LB -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, Y, M_XRD, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        LH -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, Y, M_XRD, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        LW -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, Y, M_XRD, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        LBU -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, Y, M_XRD, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        LHU -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, Y, M_XRD, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SB -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_IMM, A1_RS1, IMM_S, DW_XPR, FN_ADD, Y, M_XWR, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        SH -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_IMM, A1_RS1, IMM_S, DW_XPR, FN_ADD, Y, M_XWR, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        SW -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_IMM, A1_RS1, IMM_S, DW_XPR, FN_ADD, Y, M_XWR, N, N, N, N, N, N, N, CSR_N, N, N, N, N),
        LUI -> Seq(Y, N, N, N, N, N, N, N, N, A2_IMM, A1_ZERO, IMM_U, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        ADDI -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SLTI -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_SLT, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SLTIU -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_SLTU, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        ANDI -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_AND, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        ORI -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_OR, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        XORI -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_XOR, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        ADD -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SUB -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_SUB, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SLT -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_SLT, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SLTU -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_SLTU, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        AND -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_AND, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        OR -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_OR, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        XOR -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_XOR, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SLL -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_SL, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SRL -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_SR, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SRA -> Seq(Y, N, N, N, N, N, Y, Y, N, A2_RS2, A1_RS1, IMM_X, DW_XPR, FN_SRA, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        FENCE -> Seq(Y, N, N, N, N, N, N, N, N, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, N, N, N, N, N, N, N, CSR_N, N, Y, N, N),
        SCALL -> Seq(Y, N, N, N, N, N, N, X, N, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, N, N, N, N, N, N, N, CSR_I, N, N, N, N),
        SBREAK -> Seq(Y, N, N, N, N, N, N, X, N, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, N, N, N, N, N, N, N, CSR_I, N, N, N, N),
        MRET -> Seq(Y, N, N, N, N, N, N, X, N, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, N, N, N, N, N, N, N, CSR_I, N, N, N, N),
        WFI -> Seq(Y, N, N, N, N, N, N, X, N, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, N, N, N, N, N, N, N, CSR_I, N, N, N, N),
        CEASE -> Seq(Y, N, N, N, N, N, N, X, N, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, N, N, N, N, N, N, N, CSR_I, N, N, N, N),
        CSRRW -> Seq(Y, N, N, N, N, N, N, Y, N, A2_ZERO, A1_RS1, IMM_X, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_W, N, N, N, N),
        CSRRS -> Seq(Y, N, N, N, N, N, N, Y, N, A2_ZERO, A1_RS1, IMM_X, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_S, N, N, N, N),
        CSRRC -> Seq(Y, N, N, N, N, N, N, Y, N, A2_ZERO, A1_RS1, IMM_X, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_C, N, N, N, N),
        CSRRWI -> Seq(Y, N, N, N, N, N, N, N, N, A2_IMM, A1_ZERO, IMM_Z, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_W, N, N, N, N),
        CSRRSI -> Seq(Y, N, N, N, N, N, N, N, N, A2_IMM, A1_ZERO, IMM_Z, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_S, N, N, N, N),
        CSRRCI -> Seq(Y, N, N, N, N, N, N, N, N, A2_IMM, A1_ZERO, IMM_Z, DW_XPR, FN_ADD, N, M_X, N, N, N, N, N, N, Y, CSR_C, N, N, N, N),
        SLLI_RV32 -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_SL, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SRLI_RV32 -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_SR, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
        SRAI_RV32 -> Seq(Y, N, N, N, N, N, N, Y, N, A2_IMM, A1_RS1, IMM_I, DW_XPR, FN_SRA, N, M_X, N, N, N, N, N, N, Y, CSR_N, N, N, N, N),
      ).map { case (k, v) => BitPat(s"b$k") -> BitPat(s"b${v.reduce(_ + _)}") },
      BitPat(s"b${Seq(N, X, X, X, X, X, X, X, X, A2_X, A1_X, IMM_X, DW_X, FN_X, N, M_X, X, X, X, X, X, X, X, CSR_X, X, X, X, X).reduce(_ + _)}")
    ))
  }
}
