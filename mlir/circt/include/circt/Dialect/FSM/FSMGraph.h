//===- FSMGraph.h - FSM graph -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FSMGraph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMGRAPH_H
#define CIRCT_DIALECT_FSM_FSMGRAPH_H

#include "circt/Dialect/FSM/FSMOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"

#include <regex>

namespace circt {
namespace fsm {

namespace detail {
/// This just maps a iterator of references to an iterator of addresses.
template <typename It>
struct AddressIterator
    : public llvm::mapped_iterator<It, typename It::pointer (*)(
                                           typename It::reference)> {
  // This using statement is to get around a bug in MSVC.  Without it, it
  // tries to look up "It" as a member type of the parent class.
  using Iterator = It;
  static typename Iterator::value_type *
  addrOf(typename Iterator::value_type &v) noexcept {
    return std::addressof(v);
  }
  /* implicit */ AddressIterator(Iterator iterator)
      : llvm::mapped_iterator<It, typename Iterator::pointer (*)(
                                      typename Iterator::reference)>(iterator,
                                                                     addrOf) {}
};

// Escapes any occurance of (regex) characters 'c' in 'str'. If 'noEscape' is
// set, does not prepend a '\' character to the replaced value.
static void escape(std::string &str, const char *c, bool noEscape = false) {
  std::string replacement = std::string(c);
  if (!noEscape)
    replacement = R"(\)" + replacement;
  str = std::regex_replace(str, std::regex(c), replacement);
}

// Dumps a range of operations to a string in a format suitable for embedding
// inside a .dot edge/node label.
template <typename TOpRange>
static std::string dotSafeDumpOps(TOpRange ops) {
  std::string dump;
  llvm::raw_string_ostream ss(dump);
  llvm::interleave(
      ops, ss, [&](mlir::Operation &op) { op.print(ss); }, "\\n");

  // Ensure that special characters which may be present in the Op dumps are
  // properly escaped.
  escape(dump, R"(")");
  escape(dump, R"(\{)", /*noEscape=*/true);
  escape(dump, R"(\})", /*noEscape=*/true);
  return dump;
}

// Dumps a range of operations to a string.
template <typename TOpRange>
static std::string dumpOps(TOpRange ops) {
  std::string dump;
  llvm::raw_string_ostream ss(dump);
  llvm::interleave(
      ops, ss, [&](mlir::Operation &op) { op.print(ss); }, "\n");

  return dump;
}

} // namespace detail

class FSMStateNode;

/// This is an edge in the FSMGraph. This tracks a transition between two
/// states.
class FSMTransitionEdge
    : public llvm::ilist_node_with_parent<FSMTransitionEdge, FSMStateNode> {
public:
  /// Get the state where the transition originates from.
  FSMStateNode *getCurrentState() const { return currentState; }

  /// Get the module which the FSM-like is instantiating.
  FSMStateNode *getNextState() const { return nextState; }

  TransitionOp getTransition() const { return transition; }

  /// Erase this transition, removing it from the source state and the target
  /// state's use-list.
  void erase();

private:
  friend class FSMGraphBase;
  friend class FSMStateNode;

  FSMTransitionEdge(FSMStateNode *currentState, TransitionOp transition,
                    FSMStateNode *nextState)
      : currentState(currentState), transition(transition),
        nextState(nextState) {}
  FSMTransitionEdge(const FSMTransitionEdge &) = delete;

  /// The state where this transition originates from.
  FSMStateNode *currentState;

  /// The transition that this is tracking.
  TransitionOp transition;

  /// The next state of this transition.
  FSMStateNode *nextState;

  /// Intrusive linked list for other uses.
  FSMTransitionEdge *nextUse = nullptr;
  FSMTransitionEdge *prevUse = nullptr;
};

/// This is a Node in the FSMGraph.  Each node represents a state in the FSM.
class FSMStateNode : public llvm::ilist_node<FSMStateNode> {
  using FSMTransitionList = llvm::iplist<FSMTransitionEdge>;

public:
  FSMStateNode() : state(nullptr) {}

  /// Get the state operation that this node is tracking.
  StateOp getState() const { return state; }

  /// Adds a new transition edge from this state to 'nextState'.
  FSMTransitionEdge *addTransitionEdge(FSMStateNode *nextState,
                                       TransitionOp transition);

  /// Erases a transition edge from this state. This also removes the underlying
  /// TransitionOp.
  void eraseTransitionEdge(FSMTransitionEdge *edge);

  /// Iterate outgoing FSM transitions of this state.
  using iterator = detail::AddressIterator<FSMTransitionList::iterator>;
  iterator begin() { return transitions.begin(); }
  iterator end() { return transitions.end(); }

  /// Iterator for state uses.
  struct UseIterator
      : public llvm::iterator_facade_base<
            UseIterator, std::forward_iterator_tag, FSMTransitionEdge *> {
    UseIterator() : current(nullptr) {}
    UseIterator(FSMStateNode *node) : current(node->firstUse) {}
    FSMTransitionEdge *operator*() const { return current; }
    using llvm::iterator_facade_base<UseIterator, std::forward_iterator_tag,
                                     FSMTransitionEdge *>::operator++;
    UseIterator &operator++() {
      assert(current && "incrementing past end");
      current = current->nextUse;
      return *this;
    }
    bool operator==(const UseIterator &other) const {
      return current == other.current;
    }

  private:
    FSMTransitionEdge *current;
  };

  /// Iterate the instance records which instantiate this module.
  UseIterator usesBegin() { return {this}; }
  UseIterator usesEnd() { return {}; }
  llvm::iterator_range<UseIterator> uses() {
    return llvm::make_range(usesBegin(), usesEnd());
  }

private:
  friend class FSMTransitionEdge;

  FSMStateNode(StateOp state) : state(state) {}
  FSMStateNode(const FSMStateNode &) = delete;

  /// Record that a tramsition referenced this state.
  void recordUse(FSMTransitionEdge *transition);

  /// The state.
  StateOp state;

  /// List of outgoing transitions from this state.
  FSMTransitionList transitions;

  /// List of transitions which reference this state.
  FSMTransitionEdge *firstUse = nullptr;

  /// Provide access to the constructor.
  friend class FSMGraph;
};

/// Graph representing FSM machines, their states and transitions between these.
class FSMGraph {
  /// This is the list of FSMStateNodes in the graph.
  using NodeList = llvm::iplist<FSMStateNode>;

public:
  /// Create a new graph of an FSM machine operation.
  explicit FSMGraph(Operation *operation);

  /// Look up a FSMStateNode for a state.
  FSMStateNode *lookup(StateOp op);

  /// Lookup a FSMStateNode by name.
  FSMStateNode *lookup(StringAttr name);

  /// Lookup a FSMStateNode for a state.
  FSMStateNode *operator[](StateOp op) { return lookup(op); }

  /// Get the node corresponding to the entry state of the FSM.
  FSMStateNode *getEntryNode();

  /// Return the FSM machine operation which this graph tracks.
  MachineOp getMachine() const { return machine; }

  /// Retrieves the state node for a 'state'. If the node does not yet exists, a
  /// new state node is created.
  FSMStateNode *getOrAddState(StateOp state);

  /// Creates a new StateOp operation in this machine and updates the graph.
  FSMStateNode *createState(OpBuilder &builder, Location loc, StringRef name);

  /// Creates a new transition operation between two states and updates the
  /// graph.
  FSMTransitionEdge *createTransition(OpBuilder &builder, Location loc,
                                      StateOp from, StateOp to);

  /// Removes this state from the graph. This will also remove the state in the
  /// underlying machine and all transitions to the state.
  void eraseState(StateOp state);

  /// Renames the 'state' to the provided 'name', and updates all referencing
  /// transitions.
  void renameState(StateOp state, StringRef name);

  /// Iterate through all states.
  using iterator = detail::AddressIterator<NodeList::iterator>;
  iterator begin() { return nodes.begin(); }
  iterator end() { return nodes.end(); }

private:
  FSMGraph(const FSMGraph &) = delete;

  /// The node under which all states are nested.
  MachineOp machine;

  /// The storage for graph nodes, with deterministic iteration.
  NodeList nodes;

  /// This maps each StateOp to its graph node.
  llvm::DenseMap<Attribute, FSMStateNode *> nodeMap;
};

} // namespace fsm
} // namespace circt

// Provide graph traits for iterating the states.
template <>
struct llvm::GraphTraits<circt::fsm::FSMStateNode *> {
  using NodeType = circt::fsm::FSMStateNode;
  using NodeRef = NodeType *;

  // Helper for getting the next state of a transition edge.
  static NodeRef getNextState(const circt::fsm::FSMTransitionEdge *transition) {
    return transition->getNextState();
  }

  using ChildIteratorType =
      llvm::mapped_iterator<circt::fsm::FSMStateNode::iterator,
                            decltype(&getNextState)>;

  static NodeRef getEntryNode(NodeRef node) { return node; }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static ChildIteratorType child_begin(NodeRef node) {
    return {node->begin(), &getNextState};
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static ChildIteratorType child_end(NodeRef node) {
    return {node->end(), &getNextState};
  }
};

// Graph traits for the FSM graph.
template <>
struct llvm::GraphTraits<circt::fsm::FSMGraph *>
    : public llvm::GraphTraits<circt::fsm::FSMStateNode *> {
  using nodes_iterator = circt::fsm::FSMGraph::iterator;

  static NodeRef getEntryNode(circt::fsm::FSMGraph *graph) {
    return graph->getEntryNode();
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static nodes_iterator nodes_begin(circt::fsm::FSMGraph *graph) {
    return graph->begin();
  }
  // NOLINTNEXTLINE(readability-identifier-naming)
  static nodes_iterator nodes_end(circt::fsm::FSMGraph *graph) {
    return graph->end();
  }
};

// Graph traits for DOT labelling.
template <>
struct llvm::DOTGraphTraits<circt::fsm::FSMGraph *>
    : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(circt::fsm::FSMStateNode *node,
                                  circt::fsm::FSMGraph *) {
    // The name of the graph node is the state name.
    return node->getState().getSymName().str();
  }

  static std::string getNodeDescription(circt::fsm::FSMStateNode *node,
                                        circt::fsm::FSMGraph *) {
    // The description of the node is the dump of its Output region.
    return circt::fsm::detail::dumpOps(node->getState().getOutput().getOps());
  }

  template <typename Iterator>
  static std::string getEdgeAttributes(const circt::fsm::FSMStateNode *node,
                                       Iterator it, circt::fsm::FSMGraph *) {
    // Set an edge label that is the dump of the inner contents of the guard of
    // the transition.
    circt::fsm::FSMTransitionEdge *edge = *it.getCurrent();
    circt::fsm::TransitionOp transition = edge->getTransition();
    if (!transition.hasGuard())
      return "";

    std::string attrs = "label=\"";
    attrs += circt::fsm::detail::dotSafeDumpOps(llvm::make_filter_range(
        transition.getGuard().getOps(), [](mlir::Operation &op) {
          // Ignore implicit fsm.return/fsm.output operations with no operands.
          if (isa<circt::fsm::ReturnOp, circt::fsm::OutputOp>(op))
            return op.getNumOperands() != 0;

          return true;
        }));
    attrs += "\"";
    return attrs;
  }

  static void addCustomGraphFeatures(const circt::fsm::FSMGraph *graph,
                                     GraphWriter<circt::fsm::FSMGraph *> &gw) {
    // Print a separate node for global variables in the FSM.
    llvm::raw_ostream &os = gw.getOStream();

    os << "variables [shape=record,label=\"Variables|";
    os << circt::fsm::detail::dotSafeDumpOps(llvm::make_filter_range(
        graph->getMachine().getOps(), [](mlir::Operation &op) {
          // Filter state ops; these are printed as separate nodes in the graph.
          return !isa<circt::fsm::StateOp>(&op);
        }));
    os << "\"]";
  }
};

#endif // CIRCT_DIALECT_FSM_FSMGRAPH_H
