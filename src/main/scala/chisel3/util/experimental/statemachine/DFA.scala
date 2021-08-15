// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.statemachine
import logger.LazyLogging

/**
  * @tparam State a finite set of states
  * @tparam Symbol a finite set of input symbols,
  * @param transition transition functions, (state, symbol) -> state
  * @param initialState the initial state of NFA
  * @param finalStates functions to judge if a state is final state
  */
private case class DFA[State, Symbol](
  initialState: State,
  transition:   Map[(State, Symbol), State],
  finalStates:  Set[State])
    extends LazyLogging {
  def move(state: State, word: Symbol): State = {
    val nextState: State = transition(state, word)
    logger.trace(s"$state -$word-> $nextState")
    nextState
  }

  /**
    * transform state by input words with [[transition]]
    * @param state current input state of DFA
    * @param words the remain word to be processed, the first word in words will be consumed,
    *              use generated state and remain words for iteration
    */
  def run(state: State, words: Seq[Symbol]): State = {
    words.toList match {
      case word :: wordTail => run(move(state, word), wordTail)
      case Nil              => state
    }
  }

  /**
    * @param words is a list of Symbol, send to DFA in sequence,
    *              find if it can be accepted by this [[DFA]]
    */
  def accepts(words: Seq[Symbol]): Boolean = {
    finalStates.contains(run(initialState, words))
  }
}
