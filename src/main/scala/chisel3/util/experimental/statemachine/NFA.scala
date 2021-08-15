// SPDX-License-Identifier: Apache-2.0

package chisel3.util.experimental.statemachine

import scala.annotation.tailrec

/**
  * @tparam State a finite set of states
  * @tparam Symbol a finite set of input symbols
  * @param transition transition functions
  *                   state, Some(symbol) -> state
  *                   state, None -> state means epsilon transition
  * @param initialState the initial state of NFA
  * @param finalStates functions to judge if a state is final state
  *
  * @example
  */
private case class eNFA[State, Symbol](
  initialState: State,
  transition:   Map[(State, Option[Symbol]), Set[State]],
  finalStates:  Set[State]) {
  lazy val alphabet: Set[Symbol] = transition.keys.map(_._2).filter(_.isDefined).map(_.get).toSet
  lazy val states: Set[State] =
    Set(initialState) ++ transition.keys.map(_._1).toSet ++ transition.values.flatten.toSet ++ finalStates
  lazy val nfa: NFA[State, Symbol] = new NFA[State, Symbol](
    initialState = epsilonClosure(initialState),
    transition = states
      .flatMap(state =>
        alphabet.map(word => {
          val stateEpsilonClosure: Set[State] = epsilonClosure(state)
          ((stateEpsilonClosure, word), move(stateEpsilonClosure, Some(word)).map(epsilonClosure))
        })
      )
      .filter(_._2.nonEmpty)
      .toMap,
    finalStates = finalStates.map(epsilonClosure)
  )

  /**
    * @param states is a set of state, represent a epsilon closure.
    * @param word is a word in alphabet.
    * @return a set of states which can be transited by word.
    */
  def move(states: Set[State], word: Option[Symbol]): Set[State] = {
    states.flatMap(s => transition.getOrElse((s, word), Set[State]()))
  }

  /**
    * transform state by input words with [[transition]]
    * @param states current input states of NFA
    * @param words the remain word to be processed, the first word in words will be consumed,
    *              use generated state and remain words for iteration
    */
  def run(states: Set[State], words: Seq[Symbol]): Set[State] = {
    words.toList match {
      case word :: wordTail => run(move(states, Some(word)), wordTail)
      case Nil              => states
    }
  }

  /**
    * @param words is a list of Symbol, send to NFA in sequence,
    *              find if it can be accepted by this [[eNFA]]
    */
  def accepts(words: Seq[Symbol]): Boolean = run(Set(initialState), words).subsetOf(finalStates)

  /**
    * return a epsilon closure of a input state.
    */
  def epsilonClosure(state: State): Set[State] = epsilonClosure(Set(state))

  /**
    * helper function to generate all the epsilonClosure
    * states is a set of state in a same epsilon closure
    * each return will generate the next sets of epsilon closure
    * util generated states is same as input states
    */
  @tailrec
  private def epsilonClosure(states: Set[State]): Set[State] = {
    val newStates: Set[State] = move(states, None) ++ states
    if (newStates == states) {
      states
    } else {
      epsilonClosure(newStates)
    }
  }

}

private case class NFA[State, Symbol](
  initialState: Set[State],
  transition:   Map[(Set[State], Symbol), Set[Set[State]]],
  finalStates:  Set[Set[State]]) {
  lazy val (
    singleEdge: Map[(Set[State], Symbol), Set[Set[State]]],
    multiEdge:  Map[(Set[State], Symbol), Set[Set[State]]]
  ) = transition.partition(_._2.size == 1)

  private lazy val cleanMultiEdge: NFA[State, Symbol] = {
    if (multiEdge.isEmpty) this
    else
      new NFA[State, Symbol](
        initialState = initialState,
        transition = {
          multiEdge.foldLeft(Set[((Set[State], Symbol), Set[Set[State]])]()) { (l, r) =>
            {
              l ++ (r match {
                case ((state, symbol), nextStates) =>
                  val mergedState:      Set[State] = nextStates.flatten
                  val mergedTransition: ((Set[State], Symbol), Set[Set[State]]) = ((state, symbol), Set(mergedState))
                  transition.filterKeys(s => nextStates.contains(s._1)).map {
                    case ((_, symbol), nextStates) => ((mergedState, symbol), nextStates)
                  } + mergedTransition
              })
            }
          } ++ singleEdge.toSet
        }.toMap,
        finalStates = finalStates
      ).cleanMultiEdge
  }

  def dfa: DFA[Set[State], Symbol] = DFA(
    initialState = cleanMultiEdge.initialState,
    transition = cleanMultiEdge.transition.map {
      case ((state, symbol), nextState) => ((state, symbol), nextState.flatten)
    },
    finalStates = cleanMultiEdge.finalStates
  )

  /**
    * @param states is a set of state, represent a epsilon closure.
    * @param word is a word in alphabet.
    * @return a set of states which can be transited by word.
    */
  def move(states: Set[Set[State]], word: Symbol): Set[Set[State]] = {
    states.flatMap(s => transition.getOrElse((s, word), Set[Set[State]]()))
  }

  /**
    * transform state by input words with [[transition]]
    * @param states current input states of NFA
    * @param words the remain word to be processed, the first word in words will be consumed,
    *              use generated state and remain words for iteration
    */
  def run(states: Set[Set[State]], words: Seq[Symbol]): Set[Set[State]] = {
    words.toList match {
      case word :: wordTail => run(move(states, word), wordTail)
      case Nil              => states
    }
  }

  /**
    * @param words is a list of Symbol, send to NFA in sequence,
    *              find if it can be accepted by this [[eNFA]]
    */
  def accepts(words: Seq[Symbol]): Boolean = run(Set(initialState), words).subsetOf(finalStates)

}
