// See LICENSE for license details.

package firrtl

import org.antlr.v4.runtime.{CommonToken, Token}

import scala.annotation.tailrec
import scala.collection.mutable
import firrtl.antlr.FIRRTLParser

/*
 *    ANTLR4 helper to handle indentation tokens in Lexer
 *    code adapted from: "https://github.com/yshavit/antlr-denter" (Yuval Shavit, MIT license)
 */

abstract class LexerHelper {

  import FIRRTLParser.{NEWLINE, INDENT, DEDENT}

  private val tokenBuffer = mutable.Queue.empty[Token]
  private val indentations = mutable.Stack[Int]()
  private var reachedEof = false

  private def eofHandler(t: Token): Token = {
    // when we reach EOF, unwind all indentations. If there aren't any, insert a NEWLINE. This lets the grammar treat
    // un-indented expressions as just being NEWLINE-terminated, rather than NEWLINE|EOF.
    val ret =
      if (indentations.isEmpty)
        createToken(NEWLINE, t)
      else
        unwindTo(0, t)

    tokenBuffer.enqueue(t)
    reachedEof = true

    ret
  }

  def nextToken(): Token = {
    // first run
    if (indentations.isEmpty) {
      indentations.push(0)

      @tailrec
      def findFirstRead(): Token = {
        val t = pullToken()
        if (t.getType != NEWLINE) t else findFirstRead()
      }

      val firstRealToken = findFirstRead()

      if (firstRealToken.getCharPositionInLine > 0) {
        indentations.push(firstRealToken.getCharPositionInLine)
        tokenBuffer.enqueue(createToken(INDENT, firstRealToken))
      }
      tokenBuffer.enqueue(firstRealToken)
    }

    def handleNewlineToken(token: Token): Token = {
      @tailrec
      def nonNewline(token: Token) : (Token, Token) = {
        val nextNext = pullToken()
        if(nextNext.getType == NEWLINE)
          nonNewline(nextNext)
        else
          (token, nextNext)
      }
      val (nxtToken, nextNext) = nonNewline(token)

      if (nextNext.getType == Token.EOF)
        eofHandler(nextNext)
      else {
        val nlText = nxtToken.getText
        val indent =
          if (nlText.length > 0 && nlText.charAt(0) == '\r')
            nlText.length - 2
          else
            nlText.length - 1

        val prevIndent = indentations.head

        val retToken =
          if (indent == prevIndent)
            nxtToken
          else if (indent > prevIndent) {
            indentations.push(indent)
            createToken(INDENT, nxtToken)
          } else {
            unwindTo(indent, nxtToken)
          }

        tokenBuffer.enqueue(nextNext)
        retToken
      }
    }

    val t = if (tokenBuffer.isEmpty)
      pullToken()
    else
      tokenBuffer.dequeue

    if (reachedEof)
      t
    else if (t.getType == NEWLINE)
      handleNewlineToken(t)
    else if (t.getType == Token.EOF)
      eofHandler(t)
    else
      t
  }

  // will be overridden to FIRRTLLexer.super.nextToken() in the g4 file
  protected def pullToken(): Token

  private def createToken(tokenType: Int, copyFrom: Token): Token =
    new CommonToken(copyFrom) {
      setType(tokenType)
      tokenType match {
        case `NEWLINE` => setText("<NEWLINE>")
        case `INDENT` => setText("<INDENT>")
        case `DEDENT` => setText("<DEDENT>")
      }
    }

  /**
    * Returns a DEDENT token, and also queues up additional DEDENTs as necessary.
    *
    * @param targetIndent the "size" of the indentation (number of spaces) by the end
    * @param copyFrom     the triggering token
    * @return a DEDENT token
    */
  private def unwindTo(targetIndent: Int, copyFrom: Token): Token = {
    assert(tokenBuffer.isEmpty, tokenBuffer)
    tokenBuffer.enqueue(createToken(NEWLINE, copyFrom))
    // To make things easier, we'll queue up ALL of the dedents, and then pop off the first one.
    // For example, here's how some text is analyzed:
    //
    //  Text          :  Indentation  :  Action         : Indents Deque
    //  [ baseline ]  :  0            :  nothing        : [0]
    //  [   foo    ]  :  2            :  INDENT         : [0, 2]
    //  [    bar   ]  :  3            :  INDENT         : [0, 2, 3]
    //  [ baz      ]  :  0            :  DEDENT x2      : [0]

    @tailrec
    def doPop(): Unit = {
      val prevIndent = indentations.pop()
      if (prevIndent < targetIndent) {
        indentations.push(prevIndent)
        tokenBuffer.enqueue(createToken(INDENT, copyFrom))
      } else if (prevIndent > targetIndent) {
        tokenBuffer.enqueue(createToken(DEDENT, copyFrom))
        doPop()
      }
    }

    doPop()

    indentations.push(targetIndent)
    tokenBuffer.dequeue
  }
}
