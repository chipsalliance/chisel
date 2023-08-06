//===- Problems.h - Modeling of scheduling problems -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static scheduling algorithms for use in HLS flows all solve similar problems.
// The classes in this file serve as an interface between clients and algorithm
// implementations, and model a basic scheduling problem and some commonly used
// extensions (e.g. modulo scheduling). This includes problem-specific
// verification methods.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_PROBLEMS_H
#define CIRCT_SCHEDULING_PROBLEMS_H

#include "circt/Scheduling/DependenceIterator.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"

#include <optional>

#define DEFINE_COMMON_MEMBERS(ProblemClass)                                    \
protected:                                                                     \
  ProblemClass() {}                                                            \
                                                                               \
public:                                                                        \
  static constexpr auto PROBLEM_NAME = #ProblemClass;                          \
  static ProblemClass get(Operation *containingOp) {                           \
    ProblemClass prob;                                                         \
    prob.setContainingOp(containingOp);                                        \
    return prob;                                                               \
  }

namespace circt {
namespace scheduling {

/// This class models the most basic scheduling problem.
///
/// A problem instance is comprised of:
///
///  - *Operations*: The vertices in the data-flow graph to be scheduled.
///  - *Dependences*: The edges in the data-flow graph to be scheduled, modeling
///    a precedence relation between the involved operations.
///  - *Operator types*: An abstraction of the characteristics of the target
///    representation, e.g. modules in the  HLS component library, available
///    functional units, etc. -- operations are executed on instances/units of
///    an operator type.
///
/// Operations and operator types are stored explicitly. The registered
/// operations induce a subgraph of the SSA graph. We implicitly include the
/// dependences corresponding to its *def-use* relationships in the problem,
/// e.g. if operation *y*'s second operand uses the first result produced by
/// *x*, we'd have a dependence `x:0 --> y:1`. Clients can additionally register
/// explicit, *auxiliary* dependence between operations, e.g. to encode memory
/// dependencies or other ordering constraints. Auxiliary dependences do not
/// distinguish specific operands/results. The differences between the flavors
/// are transparent to concrete algorithms.
///
/// All components of the problem (operations, dependences, operator types, as
/// well as the instance itself) can be annotated with *properties*. In this
/// basic problem, we model
///
/// - `linkedOperatorType` maps operations to their operator types.
/// - `latency`, an operator type-property denoting the number of time steps
///   after which the operator's result is available.
/// - `startTime`, an operation-property for the time step in which an operation
///   is started. Together, the start times for all operations represent the
///   problem's solution, i.e. the schedule.
///
/// Subclasses, i.e. corresponding to more complex scheduling problems, can
/// declare additional properties as needed.
//
/// The `check...` methods perform validity checks before scheduling, e.g. that
/// all operations have an associated operator type, etc.
///
/// The `verify...` methods check the correctness of the solution determined by
/// a concrete scheduling algorithm, e.g. that there are start times available
/// for each registered operation, and the precedence constraints as modeled by
/// the dependences are satisfied.
class Problem {
  DEFINE_COMMON_MEMBERS(Problem)

public:
  virtual ~Problem() = default;

  friend detail::DependenceIterator;

  //===--------------------------------------------------------------------===//
  // Aliases for the problem components
  //===--------------------------------------------------------------------===//
public:
  /// A thin wrapper to allow a uniform handling of def-use and auxiliary
  /// dependences.
  using Dependence = detail::Dependence;

  /// Operator types are distinguished by name (chosen by the client).
  using OperatorType = mlir::StringAttr;

  //===--------------------------------------------------------------------===//
  // Aliases for containers storing the problem components and properties
  //===--------------------------------------------------------------------===//
public:
  using OperationSet = llvm::SetVector<Operation *>;
  using DependenceRange = llvm::iterator_range<detail::DependenceIterator>;
  using OperatorTypeSet = llvm::SetVector<OperatorType>;

protected:
  using AuxDependenceMap =
      llvm::DenseMap<Operation *, llvm::SmallSetVector<Operation *, 4>>;

  template <typename T>
  using OperationProperty = llvm::DenseMap<Operation *, std::optional<T>>;
  template <typename T>
  using DependenceProperty = llvm::DenseMap<Dependence, std::optional<T>>;
  template <typename T>
  using OperatorTypeProperty = llvm::DenseMap<OperatorType, std::optional<T>>;
  template <typename T>
  using InstanceProperty = std::optional<T>;

  //===--------------------------------------------------------------------===//
  // Containers for problem components and properties
  //===--------------------------------------------------------------------===//
private:
  // Operation containing the ops for this scheduling problem. Used for its
  // MLIRContext and to emit diagnostics.
  Operation *containingOp;

  // Problem components
  OperationSet operations;
  AuxDependenceMap auxDependences;
  OperatorTypeSet operatorTypes;

  // Operation properties
  OperationProperty<OperatorType> linkedOperatorType;
  OperationProperty<unsigned> startTime;

  // Operator type properties
  OperatorTypeProperty<unsigned> latency;

  //===--------------------------------------------------------------------===//
  // Problem construction
  //===--------------------------------------------------------------------===//
public:
  /// Include \p op in this scheduling problem.
  void insertOperation(Operation *op) { operations.insert(op); }

  /// Include \p dep in the scheduling problem. Return failure if \p dep does
  /// not represent a valid def-use or auxiliary dependence between operations.
  /// The endpoints become registered operations w.r.t. the problem.
  LogicalResult insertDependence(Dependence dep);

  /// Include \p opr in this scheduling problem.
  void insertOperatorType(OperatorType opr) { operatorTypes.insert(opr); }

  /// Retrieves the operator type identified by the client-specific \p name. The
  /// operator type is automatically registered in the scheduling problem.
  OperatorType getOrInsertOperatorType(StringRef name);

  //===--------------------------------------------------------------------===//
  // Access to problem components
  //===--------------------------------------------------------------------===//
public:
  /// Return the operation containing this problem, e.g. to emit diagnostics.
  Operation *getContainingOp() { return containingOp; }
  /// Set the operation containing this problem, e.g. to emit diagnostics.
  void setContainingOp(Operation *op) { containingOp = op; }

  /// Return true if \p op is part of this problem.
  bool hasOperation(Operation *op) { return operations.contains(op); }
  /// Return the set of operations.
  const OperationSet &getOperations() { return operations; }

  /// Return a range object to transparently iterate over \p op's *incoming*
  ///  1) implicit def-use dependences (backed by the SSA graph), and then
  ///  2) explictly added auxiliary dependences.
  ///
  /// In other words, this yields dependences whose destination operation is
  /// \p op, and whose source operations are \p op's predecessors in the problem
  /// graph.
  ///
  /// To iterate over all of the scheduling problem's dependences, simply
  /// process the ranges for all registered operations.
  DependenceRange getDependences(Operation *op);

  /// Return true if \p opr is part of this problem.
  bool hasOperatorType(OperatorType opr) { return operatorTypes.contains(opr); }
  /// Return the set of operator types.
  const OperatorTypeSet &getOperatorTypes() { return operatorTypes; }

  //===--------------------------------------------------------------------===//
  // Access to properties
  //===--------------------------------------------------------------------===//
public:
  /// The linked operator type provides the runtime characteristics for \p op.
  std::optional<OperatorType> getLinkedOperatorType(Operation *op) {
    return linkedOperatorType.lookup(op);
  }
  void setLinkedOperatorType(Operation *op, OperatorType opr) {
    linkedOperatorType[op] = opr;
  }

  /// The latency is the number of cycles \p opr needs to compute its result.
  std::optional<unsigned> getLatency(OperatorType opr) {
    return latency.lookup(opr);
  }
  void setLatency(OperatorType opr, unsigned val) { latency[opr] = val; }

  /// Return the start time for \p op, as computed by the scheduler.
  /// These start times comprise the basic problem's solution, i.e. the
  /// *schedule*.
  std::optional<unsigned> getStartTime(Operation *op) {
    return startTime.lookup(op);
  }
  void setStartTime(Operation *op, unsigned val) { startTime[op] = val; }

  //===--------------------------------------------------------------------===//
  // Access to derived properties
  //===--------------------------------------------------------------------===//
public:
  /// Returns the end time for \p op, as computed by the scheduler.
  /// This end time is derived from the start time and the operator type's
  /// latency.
  std::optional<unsigned> getEndTime(Operation *op);

  //===--------------------------------------------------------------------===//
  // Optional names (for exporting and debugging instances)
  //===--------------------------------------------------------------------===//
private:
  StringAttr instanceName, libraryName;
  SmallDenseMap<Operation *, StringAttr> operationNames;

public:
  StringAttr getInstanceName() { return instanceName; }
  void setInstanceName(StringAttr name) { instanceName = name; }

  StringAttr getLibraryName() { return libraryName; }
  void setLibraryName(StringAttr name) { libraryName = name; }

  StringAttr getOperationName(Operation *op) {
    return operationNames.lookup(op);
  }
  void setOperationName(Operation *op, StringAttr name) {
    operationNames[op] = name;
  }

  //===--------------------------------------------------------------------===//
  // Properties as string key-value pairs (e.g. for DOT graphs)
  //===--------------------------------------------------------------------===//
public:
  using PropertyStringVector =
      llvm::SmallVector<std::pair<std::string, std::string>, 2>;

  virtual PropertyStringVector getProperties(Operation *op);
  virtual PropertyStringVector getProperties(Dependence dep);
  virtual PropertyStringVector getProperties(OperatorType opr);
  virtual PropertyStringVector getProperties();

  //===--------------------------------------------------------------------===//
  // Property-specific validators
  //===--------------------------------------------------------------------===//
protected:
  /// \p op is linked to a registered operator type.
  virtual LogicalResult checkLinkedOperatorType(Operation *op);
  /// \p opr has a latency.
  virtual LogicalResult checkLatency(OperatorType opr);
  /// \p op has a start time.
  virtual LogicalResult verifyStartTime(Operation *op);
  /// \p dep's source operation is available before \p dep's destination
  /// operation starts.
  virtual LogicalResult verifyPrecedence(Dependence dep);

  //===--------------------------------------------------------------------===//
  // Client API for problem validation
  //===--------------------------------------------------------------------===//
public:
  /// Return success if the constructed scheduling problem is valid.
  virtual LogicalResult check();
  /// Return success if the computed solution is valid.
  virtual LogicalResult verify();
};

/// This class models a cyclic scheduling problem. Its solution can be used to
/// construct a pipelined datapath with a fixed, integer initiation interval,
/// in which the execution of multiple iterations/samples/etc. may overlap.
class CyclicProblem : public virtual Problem {
  DEFINE_COMMON_MEMBERS(CyclicProblem)

private:
  DependenceProperty<unsigned> distance;
  InstanceProperty<unsigned> initiationInterval;

public:
  /// The distance determines whether a dependence has to be satisfied in the
  /// same iteration (distance=0 or not set), or distance-many iterations later.
  std::optional<unsigned> getDistance(Dependence dep) {
    return distance.lookup(dep);
  }
  void setDistance(Dependence dep, unsigned val) { distance[dep] = val; }

  /// The initiation interval (II) is the number of time steps between
  /// subsequent iterations, i.e. a new iteration is started every II time
  /// steps. The best possible value is 1, meaning that a corresponding pipeline
  /// accepts new data every cycle. This property is part of the cyclic
  /// problem's solution.
  std::optional<unsigned> getInitiationInterval() { return initiationInterval; }
  void setInitiationInterval(unsigned val) { initiationInterval = val; }

  virtual PropertyStringVector getProperties(Dependence dep) override;
  virtual PropertyStringVector getProperties() override;

protected:
  /// \p dep's source operation is available before \p dep's destination
  /// operation starts (\p dep's distance iterations later).
  virtual LogicalResult verifyPrecedence(Dependence dep) override;
  /// This problem has a non-zero II.
  virtual LogicalResult verifyInitiationInterval();

public:
  virtual LogicalResult verify() override;
};

/// This class models the accumulation of physical propagation delays on
/// combinational paths along SSA dependences.
///
/// Each operator type is annotated with estimated values for incoming and
/// outgoing delays. Combinational operators (zero-latency, no internal
/// registers) have only a single delay; this important special case is modeled
/// by setting the incoming and outgoing delays to the same value.
///
/// A solution to this problem comprises per-operation start times in a
/// continuous unit, e.g. in nanoseconds, inside the discrete time steps/cycles
/// determined by the underlying scheduling problem.
class ChainingProblem : public virtual Problem {
  DEFINE_COMMON_MEMBERS(ChainingProblem)

private:
  OperatorTypeProperty<float> incomingDelay, outgoingDelay;
  OperationProperty<float> startTimeInCycle;

public:
  /// The incoming delay denotes the propagation time from the operand inputs to
  /// either the result outputs (combinational operators) or the first internal
  /// register stage.
  std::optional<float> getIncomingDelay(OperatorType opr) {
    return incomingDelay.lookup(opr);
  }
  void setIncomingDelay(OperatorType opr, float delay) {
    incomingDelay[opr] = delay;
  }

  /// The outgoing delay denotes the propagation time from either the operand
  /// inputs (combinational operators) or the last internal register stage to
  /// the result outputs.
  std::optional<float> getOutgoingDelay(OperatorType opr) {
    return outgoingDelay.lookup(opr);
  }
  void setOutgoingDelay(OperatorType opr, float delay) {
    outgoingDelay[opr] = delay;
  }

  /// Computed by the scheduler, this start time is relative to the beginning of
  /// the cycle that \p op starts in.
  std::optional<float> getStartTimeInCycle(Operation *op) {
    return startTimeInCycle.lookup(op);
  }
  void setStartTimeInCycle(Operation *op, float time) {
    startTimeInCycle[op] = time;
  }

  virtual PropertyStringVector getProperties(Operation *op) override;
  virtual PropertyStringVector getProperties(OperatorType opr) override;

protected:
  /// Incoming/outgoing delays are set for \p opr and non-negative. The delays
  /// are equal if \p opr is a zero-latency operator type.
  virtual LogicalResult checkDelays(OperatorType opr);

  /// \p op has a non-negative start time in its cycle.
  virtual LogicalResult verifyStartTimeInCycle(Operation *op);
  /// If \p dep is an SSA edge and its source operation finishes in the same
  /// time step as the destination operation, the source's result is available
  /// before the destination starts in that cycle.
  virtual LogicalResult verifyPrecedenceInCycle(Dependence dep);

public:
  virtual LogicalResult check() override;
  virtual LogicalResult verify() override;
};

/// This class models a resource-constrained scheduling problem. An optional,
/// non-zero *limit* marks operator types to be *shared* by the operations using
/// them. In an HLS setting, this corresponds to multiplexing multiple
/// operations onto a pre-allocated number of operator instances. These
/// instances are assumed to be *fully pipelined*, meaning each instance can
/// accept new operands (coming from a distinct operation) in each time step.
///
/// A solution to this problem is feasible iff the number of operations that use
/// a certain limited operator type, and start in the same time step, does not
/// exceed the operator type's limit. These constraints do not apply to operator
/// types without a limit (not set, or 0).
class SharedOperatorsProblem : public virtual Problem {
  DEFINE_COMMON_MEMBERS(SharedOperatorsProblem)

private:
  OperatorTypeProperty<unsigned> limit;

public:
  /// The limit is the maximum number of operations using \p opr that are
  /// allowed to start in the same time step.
  std::optional<unsigned> getLimit(OperatorType opr) {
    return limit.lookup(opr);
  }
  void setLimit(OperatorType opr, unsigned val) { limit[opr] = val; }

  virtual PropertyStringVector getProperties(OperatorType opr) override;

protected:
  /// If \p opr is limited, it has a non-zero latency.
  virtual LogicalResult checkLatency(OperatorType opr) override;
  /// \p opr is not oversubscribed in any time step.
  virtual LogicalResult verifyUtilization(OperatorType opr);

public:
  virtual LogicalResult verify() override;
};

/// This class models the modulo scheduling problem as the composition of the
/// cyclic problem and the resource-constrained problem with fully-pipelined
/// shared operators.
///
/// A solution to this problem comprises an integer II and integer start times
/// for all registered operations, and is feasible iff:
///  (1) The precedence constraints implied by the `CyclicProblem`'s dependence
///      edges are satisfied, and
///  (2) The number of operations that use a certain limited operator type,
///      and start in the same congruence class (= start time *mod* II), does
///      not exceed the operator type's limit.
class ModuloProblem : public virtual CyclicProblem,
                      public virtual SharedOperatorsProblem {
  DEFINE_COMMON_MEMBERS(ModuloProblem)

protected:
  /// \p opr is not oversubscribed in any congruence class modulo II.
  virtual LogicalResult verifyUtilization(OperatorType opr) override;

public:
  virtual LogicalResult verify() override;
};

} // namespace scheduling
} // namespace circt

#undef DEFINE_COMMON_MEMBERS

#endif // CIRCT_SCHEDULING_PROBLEMS_H
