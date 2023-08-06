//===- FSMGraph.cpp - FSM Graph ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMGraph.h"

using namespace circt;
using namespace fsm;

void FSMTransitionEdge::erase() {
  // Update the prev node to point to the next node.
  if (prevUse)
    prevUse->nextUse = nextUse;
  else
    nextState->firstUse = nextUse;
  // Update the next node to point to the prev node.
  if (nextUse)
    nextUse->prevUse = prevUse;
  currentState->eraseTransitionEdge(this);
}

void FSMStateNode::eraseTransitionEdge(FSMTransitionEdge *edge) {
  edge->getTransition().erase();
  transitions.erase(edge);
}

FSMTransitionEdge *FSMStateNode::addTransitionEdge(FSMStateNode *nextState,
                                                   TransitionOp transition) {
  auto *transitionEdge = new FSMTransitionEdge(this, transition, nextState);
  nextState->recordUse(transitionEdge);
  transitions.push_back(transitionEdge);
  return transitionEdge;
}

void FSMStateNode::recordUse(FSMTransitionEdge *transition) {
  transition->nextUse = firstUse;
  if (firstUse)
    firstUse->prevUse = transition;
  firstUse = transition;
}

FSMGraph::FSMGraph(Operation *op) {
  machine = dyn_cast<MachineOp>(op);
  assert(machine && "Expected a fsm::MachineOp");

  // Find all states in the machine.
  for (auto stateOp : machine.getOps<StateOp>()) {
    // Add an edge to indicate that this state transitions to some other state.
    auto *currentStateNode = getOrAddState(stateOp);

    for (auto transitionOp : stateOp.getTransitions().getOps<TransitionOp>()) {
      auto *nextStateNode = getOrAddState(transitionOp.getNextStateOp());
      currentStateNode->addTransitionEdge(nextStateNode, transitionOp);
    }
  }
}

FSMStateNode *FSMGraph::lookup(StringAttr name) {
  auto it = nodeMap.find(name);
  if (it != nodeMap.end())
    return it->second;
  return nullptr;
}

FSMStateNode *FSMGraph::lookup(StateOp state) {
  return lookup(state.getNameAttr());
}

FSMStateNode *FSMGraph::getOrAddState(StateOp state) {
  // Try to insert an FSMStateNode. If its not inserted, it returns
  // an iterator pointing to the node.
  auto *&node = nodeMap[state.getNameAttr()];
  if (!node) {
    node = new FSMStateNode(state);
    nodes.push_back(node);
  }
  return node;
}

FSMStateNode *FSMGraph::createState(OpBuilder &builder, Location loc,
                                    StringRef name) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToEnd(&getMachine().getBody().front());
  auto stateOp = builder.create<StateOp>(loc, name);
  return getOrAddState(stateOp);
}

FSMTransitionEdge *FSMGraph::createTransition(OpBuilder &builder, Location loc,
                                              StateOp from, StateOp to) {
  auto *currentStateNode = getOrAddState(from);
  auto *nextStateNode = getOrAddState(to);
  OpBuilder::InsertionGuard g(builder);
  // Set the insertion point to the end of the transitions.
  builder.setInsertionPointToEnd(&from.getTransitions().getBlocks().front());
  auto transition = builder.create<TransitionOp>(loc, to);
  return currentStateNode->addTransitionEdge(nextStateNode, transition);
}

void FSMGraph::eraseState(StateOp state) {
  auto *stateNode = getOrAddState(state);

  for (auto *incomingTransition : llvm::make_early_inc_range(stateNode->uses()))
    incomingTransition->erase();

  for (auto *outgoingTransitions : llvm::make_early_inc_range(*stateNode))
    outgoingTransitions->erase();
  nodeMap.erase(state.getNameAttr());
  nodes.erase(stateNode);
}

void FSMGraph::renameState(StateOp state, StringRef name) {
  auto *stateNode = getOrAddState(state);
  auto nameStrAttr = StringAttr::get(state.getContext(), name);

  state.setName(nameStrAttr);

  // Update in- and outgoing transitions to the state.
  auto updateTransitions = [&](auto &&transitionRange) {
    for (auto *transition : transitionRange) {
      auto transitionOp = transition->getTransition();
      transitionOp->setAttr(transitionOp.getNextStateAttrName(), nameStrAttr);
    }
  };

  updateTransitions(stateNode->uses());
  updateTransitions(*stateNode);

  // Update nodemap
  nodeMap.erase(state.getNameAttr());
  nodeMap[nameStrAttr] = stateNode;
}
