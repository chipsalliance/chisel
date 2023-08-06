//===- InferWidths.cpp - Infer width of types -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InferWidths pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "infer-widths"

using mlir::InferTypeOpInterface;
using mlir::WalkOrder;

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static void diagnoseUninferredType(InFlightDiagnostic &diag, Type t,
                                   Twine str) {
  auto basetype = type_dyn_cast<FIRRTLBaseType>(t);
  if (!basetype)
    return;
  if (!basetype.hasUninferredWidth())
    return;

  if (basetype.isGround())
    diag.attachNote() << "Field: \"" << str << "\"";
  else if (auto vecType = type_dyn_cast<FVectorType>(basetype))
    diagnoseUninferredType(diag, vecType.getElementType(), str + "[]");
  else if (auto bundleType = type_dyn_cast<BundleType>(basetype))
    for (auto &elem : bundleType.getElements())
      diagnoseUninferredType(diag, elem.type, str + "." + elem.name.getValue());
}

//===----------------------------------------------------------------------===//
// Constraint Expressions
//===----------------------------------------------------------------------===//

namespace {
struct Expr;
} // namespace

/// Allow rvalue refs to `Expr` and subclasses to be printed to streams.
template <typename T, typename std::enable_if<std::is_base_of<Expr, T>::value,
                                              int>::type = 0>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const T &e) {
  e.print(os);
  return os;
}

// Allow expression subclasses to be hashed.
namespace mlir {
template <typename T, typename std::enable_if<std::is_base_of<Expr, T>::value,
                                              int>::type = 0>
inline llvm::hash_code hash_value(const T &e) {
  return e.hash_value();
}
} // namespace mlir

namespace {
#define EXPR_NAMES(x)                                                          \
  Root##x, Var##x, Derived##x, Id##x, Known##x, Add##x, Pow##x, Max##x, Min##x
#define EXPR_KINDS EXPR_NAMES()
#define EXPR_CLASSES EXPR_NAMES(Expr)

/// An expression on the right-hand side of a constraint.
struct Expr {
  enum class Kind { EXPR_KINDS };
  std::optional<int32_t> solution;
  Kind kind;

  /// Print a human-readable representation of this expr.
  void print(llvm::raw_ostream &os) const;

protected:
  Expr(Kind kind) : kind(kind) {}
  llvm::hash_code hash_value() const { return llvm::hash_value(kind); }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct ExprBase : public Expr {
  ExprBase() : Expr(DerivedKind) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
  bool operator==(const Expr &other) const {
    if (auto otherSame = dyn_cast<DerivedT>(other))
      return *static_cast<DerivedT *>(this) == otherSame;
    return false;
  }
};

/// The root containing all expressions.
struct RootExpr : public ExprBase<RootExpr, Expr::Kind::Root> {
  RootExpr(std::vector<Expr *> &exprs) : exprs(exprs) {}
  void print(llvm::raw_ostream &os) const { os << "root"; }
  std::vector<Expr *> &exprs;
};

/// A free variable.
struct VarExpr : public ExprBase<VarExpr, Expr::Kind::Var> {
  void print(llvm::raw_ostream &os) const {
    // Hash the `this` pointer into something somewhat human readable. Since
    // this is just for debug dumping, we wrap around at 65536 variables.
    os << "var" << ((size_t)this / llvm::PowerOf2Ceil(sizeof(*this)) & 0xFFFF);
  }

  /// The constraint expression this variable is supposed to be greater than or
  /// equal to. This is not part of the variable's hash and equality property.
  Expr *constraint = nullptr;

  /// The upper bound this variable is supposed to be smaller than or equal to.
  Expr *upperBound = nullptr;
  std::optional<int32_t> upperBoundSolution;
};

/// A derived width.
///
/// These are generated for `InvalidValueOp`s which want to derived their width
/// from connect operations that they are on the right hand side of.
struct DerivedExpr : public ExprBase<DerivedExpr, Expr::Kind::Derived> {
  void print(llvm::raw_ostream &os) const {
    // Hash the `this` pointer into something somewhat human readable.
    os << "derive"
       << ((size_t)this / llvm::PowerOf2Ceil(sizeof(*this)) & 0xFFF);
  }

  /// The expression this derived width is equivalent to.
  Expr *assigned = nullptr;
};

/// An identity expression.
///
/// This expression evaluates to its inner expression. It is used in a very
/// specific case of constraints on variables, in order to be able to track
/// where the constraint was imposed. Constraints on variables are represented
/// as `var >= <expr>`. When the first constraint `a` is imposed, it is stored
/// as the constraint expression (`var >= a`). When the second constraint `b` is
/// imposed, a *new* max expression is allocated (`var >= max(a, b)`).
/// Expressions are annotated with a location when they are created, which in
/// this case are connect ops. Since imposing the first constraint does not
/// create any new expression, the location information of that connect would be
/// lost. With an identity expression, imposing the first constraint becomes
/// `var >= identity(a)`, which is a *new* expression and properly tracks the
/// location info.
struct IdExpr : public ExprBase<IdExpr, Expr::Kind::Id> {
  IdExpr(Expr *arg) : arg(arg) { assert(arg); }
  void print(llvm::raw_ostream &os) const { os << "*" << *arg; }
  bool operator==(const IdExpr &other) const {
    return kind == other.kind && arg == other.arg;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), arg);
  }

  /// The inner expression.
  Expr *const arg;
};

/// A known constant value.
struct KnownExpr : public ExprBase<KnownExpr, Expr::Kind::Known> {
  KnownExpr(int32_t value) : ExprBase() { solution = value; }
  void print(llvm::raw_ostream &os) const { os << *solution; }
  bool operator==(const KnownExpr &other) const {
    return *solution == *other.solution;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), *solution);
  }
};

/// A unary expression. Contains the actual data. Concrete subclasses are merely
/// there for show and ease of use.
struct UnaryExpr : public Expr {
  bool operator==(const UnaryExpr &other) const {
    return kind == other.kind && arg == other.arg;
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), arg);
  }

  /// The child expression.
  Expr *const arg;

protected:
  UnaryExpr(Kind kind, Expr *arg) : Expr(kind), arg(arg) { assert(arg); }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct UnaryExprBase : public UnaryExpr {
  template <typename... Args>
  UnaryExprBase(Args &&...args)
      : UnaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
};

/// A power of two.
struct PowExpr : public UnaryExprBase<PowExpr, Expr::Kind::Pow> {
  using UnaryExprBase::UnaryExprBase;
  void print(llvm::raw_ostream &os) const { os << "2^" << arg; }
};

/// A binary expression. Contains the actual data. Concrete subclasses are
/// merely there for show and ease of use.
struct BinaryExpr : public Expr {
  bool operator==(const BinaryExpr &other) const {
    return kind == other.kind && lhs() == other.lhs() && rhs() == other.rhs();
  }
  llvm::hash_code hash_value() const {
    return llvm::hash_combine(Expr::hash_value(), *args);
  }
  Expr *lhs() const { return args[0]; }
  Expr *rhs() const { return args[1]; }

  /// The child expressions.
  Expr *const args[2];

protected:
  BinaryExpr(Kind kind, Expr *lhs, Expr *rhs) : Expr(kind), args{lhs, rhs} {
    assert(lhs);
    assert(rhs);
  }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Expr::Kind DerivedKind>
struct BinaryExprBase : public BinaryExpr {
  template <typename... Args>
  BinaryExprBase(Args &&...args)
      : BinaryExpr(DerivedKind, std::forward<Args>(args)...) {}
  static bool classof(const Expr *e) { return e->kind == DerivedKind; }
};

/// An addition.
struct AddExpr : public BinaryExprBase<AddExpr, Expr::Kind::Add> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "(" << *lhs() << " + " << *rhs() << ")";
  }
};

/// The maximum of two expressions.
struct MaxExpr : public BinaryExprBase<MaxExpr, Expr::Kind::Max> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "max(" << *lhs() << ", " << *rhs() << ")";
  }
};

/// The minimum of two expressions.
struct MinExpr : public BinaryExprBase<MinExpr, Expr::Kind::Min> {
  using BinaryExprBase::BinaryExprBase;
  void print(llvm::raw_ostream &os) const {
    os << "min(" << *lhs() << ", " << *rhs() << ")";
  }
};

void Expr::print(llvm::raw_ostream &os) const {
  TypeSwitch<const Expr *>(this).Case<EXPR_CLASSES>(
      [&](auto *e) { e->print(os); });
}

} // namespace

//===----------------------------------------------------------------------===//
// Fast bump allocator with optional interning
//===----------------------------------------------------------------------===//

namespace {

/// An allocation slot in the `InternedAllocator`.
template <typename T>
struct InternedSlot {
  T *ptr;
  InternedSlot(T *ptr) : ptr(ptr) {}
};

/// A simple bump allocator that ensures only ever one copy per object exists.
/// The allocated objects must not have a destructor.
template <typename T, typename std::enable_if_t<
                          std::is_trivially_destructible<T>::value, int> = 0>
class InternedAllocator {
  using Slot = InternedSlot<T>;
  llvm::DenseSet<Slot> interned;
  llvm::BumpPtrAllocator &allocator;

public:
  InternedAllocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object if it does not yet exist, or return a pointer to the
  /// existing one. `R` is the type of the object to be allocated. `R` must be
  /// derived from or be the type `T`.
  template <typename R = T, typename... Args>
  std::pair<R *, bool> alloc(Args &&...args) {
    auto stack_value = R(std::forward<Args>(args)...);
    auto stack_slot = Slot(&stack_value);
    auto it = interned.find(stack_slot);
    if (it != interned.end())
      return std::make_pair(static_cast<R *>(it->ptr), false);
    auto heap_value = new (allocator) R(std::move(stack_value));
    interned.insert(Slot(heap_value));
    return std::make_pair(heap_value, true);
  }
};

/// A simple bump allocator. The allocated objects must not have a destructor.
/// This allocator is mainly there for symmetry with the `InternedAllocator`.
template <typename T, typename std::enable_if_t<
                          std::is_trivially_destructible<T>::value, int> = 0>
class Allocator {
  llvm::BumpPtrAllocator &allocator;

public:
  Allocator(llvm::BumpPtrAllocator &allocator) : allocator(allocator) {}

  /// Allocate a new object. `R` is the type of the object to be allocated. `R`
  /// must be derived from or be the type `T`.
  template <typename R = T, typename... Args>
  R *alloc(Args &&...args) {
    return new (allocator) R(std::forward<Args>(args)...);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Constraint Solver
//===----------------------------------------------------------------------===//

namespace {
/// A canonicalized linear inequality that maps a constraint on var `x` to the
/// linear inequality `x >= max(a*x+b, c) + (failed ? ∞ : 0)`.
///
/// The inequality separately tracks recursive (a, b) and non-recursive (c)
/// constraints on `x`. This allows it to properly identify the combination of
/// the two constraints `x >= x-1` and `x >= 4` to be satisfiable as
/// `x >= max(x-1, 4)`. If it only tracked inequality as `x >= a*x+b`, the
/// combination of these two constraints would be `x >= x+4` (due to max(-1,4) =
/// 4), which would be unsatisfiable.
///
/// The `failed` flag acts as an additional `∞` term that renders the inequality
/// unsatisfiable. It is used as a tombstone value in case an operation renders
/// the equality unsatisfiable (e.g. `x >= 2**x` would be represented as the
/// inequality `x >= ∞`).
///
/// Inequalities represented in this form can easily be checked for
/// unsatisfiability in the presence of recursion by inspecting the coefficients
/// a and b. The `sat` function performs this action.
struct LinIneq {
  // x >= max(a*x+b, c) + (failed ? ∞ : 0)
  int32_t rec_scale = 0;   // a
  int32_t rec_bias = 0;    // b
  int32_t nonrec_bias = 0; // c
  bool failed = false;

  /// Create a new unsatisfiable inequality `x >= ∞`.
  static LinIneq unsat() { return LinIneq(true); }

  /// Create a new inequality `x >= (failed ? ∞ : 0)`.
  explicit LinIneq(bool failed = false) : failed(failed) {}

  /// Create a new inequality `x >= bias`.
  explicit LinIneq(int32_t bias) : nonrec_bias(bias) {}

  /// Create a new inequality `x >= scale*x+bias`.
  explicit LinIneq(int32_t scale, int32_t bias) {
    if (scale != 0) {
      rec_scale = scale;
      rec_bias = bias;
    } else {
      nonrec_bias = bias;
    }
  }

  /// Create a new inequality `x >= max(rec_scale*x+rec_bias, nonrec_bias) +
  /// (failed ? ∞ : 0)`.
  explicit LinIneq(int32_t rec_scale, int32_t rec_bias, int32_t nonrec_bias,
                   bool failed = false)
      : failed(failed) {
    if (rec_scale != 0) {
      this->rec_scale = rec_scale;
      this->rec_bias = rec_bias;
      this->nonrec_bias = nonrec_bias;
    } else {
      this->nonrec_bias = std::max(rec_bias, nonrec_bias);
    }
  }

  /// Combine two inequalities by taking the maxima of corresponding
  /// coefficients.
  ///
  /// This essentially combines `x >= max(a1*x+b1, c1)` and `x >= max(a2*x+b2,
  /// c2)` into a new `x >= max(max(a1,a2)*x+max(b1,b2), max(c1,c2))`. This is
  /// a pessimistic upper bound, since e.g. `x >= 2x-10` and `x >= x-5` may both
  /// hold, but the resulting `x >= 2x-5` may pessimistically not hold.
  static LinIneq max(const LinIneq &lhs, const LinIneq &rhs) {
    return LinIneq(std::max(lhs.rec_scale, rhs.rec_scale),
                   std::max(lhs.rec_bias, rhs.rec_bias),
                   std::max(lhs.nonrec_bias, rhs.nonrec_bias),
                   lhs.failed || rhs.failed);
  }

  /// Combine two inequalities by summing up the two right hand sides.
  ///
  /// This is a tricky one, since the addition of the two max terms will lead to
  /// a maximum over four possible terms (similar to a binomial expansion). In
  /// order to shoehorn this back into a two-term maximum, we have to pick the
  /// recursive term that will grow the fastest.
  ///
  /// As an example for this problem, consider the following addition:
  ///
  ///   x >= max(a1*x+b1, c1) + max(a2*x+b2, c2)
  ///
  /// We would like to expand and rearrange this again into a maximum:
  ///
  ///   x >= max(a1*x+b1 + max(a2*x+b2, c2), c1 + max(a2*x+b2, c2))
  ///   x >= max(max(a1*x+b1 + a2*x+b2, a1*x+b1 + c2),
  ///            max(c1 + a2*x+b2, c1 + c2))
  ///   x >= max((a1+a2)*x+(b1+b2), a1*x+(b1+c2), a2*x+(b2+c1), c1+c2)
  ///
  /// Since we are combining two two-term maxima, there are four possible ways
  /// how the terms can combine, leading to the above four-term maximum. An easy
  /// upper bound of the form we want would be the following:
  ///
  ///   x >= max(max(a1+a2, a1, a2)*x + max(b1+b2, b1+c2, b2+c1), c1+c2)
  ///
  /// However, this is a very pessimistic upper-bound that will declare very
  /// common patterns in the IR as unbreakable cycles, despite them being very
  /// much breakable. For example:
  ///
  ///   x >= max(x, 42) + max(0, -3)  <-- breakable recursion
  ///   x >= max(max(1+0, 1, 0)*x + max(42+0, -3, 42), 42-2)
  ///   x >= max(x + 42, 39)          <-- unbreakable recursion!
  ///
  /// A better approach is to take the expanded four-term maximum, retain the
  /// non-recursive term (c1+c2), and estimate which one of the recursive terms
  /// (first three) will become dominant as we choose greater values for x.
  /// Since x never is inferred to be negative, the recursive term in the
  /// maximum with the highest scaling factor for x will end up dominating as
  /// x tends to ∞:
  ///
  ///   x >= max({
  ///     (a1+a2)*x+(b1+b2) if a1+a2 >= max(a1+a2, a1, a2) and a1>0 and a2>0,
  ///     a1*x+(b1+c2)      if    a1 >= max(a1+a2, a1, a2) and a1>0,
  ///     a2*x+(b2+c1)      if    a2 >= max(a1+a2, a1, a2) and a2>0,
  ///     0                 otherwise
  ///   }, c1+c2)
  ///
  /// In case multiple cases apply, the highest bias of the recursive term is
  /// picked. With this, the above problematic example triggers the second case
  /// and becomes:
  ///
  ///   x >= max(1*x+(0-3), 42-3) = max(x-3, 39)
  ///
  /// Of which the first case is chosen, as it has the lower bias value.
  static LinIneq add(const LinIneq &lhs, const LinIneq &rhs) {
    // Determine the maximum scaling factor among the three possible recursive
    // terms.
    auto enable1 = lhs.rec_scale > 0 && rhs.rec_scale > 0;
    auto enable2 = lhs.rec_scale > 0;
    auto enable3 = rhs.rec_scale > 0;
    auto scale1 = lhs.rec_scale + rhs.rec_scale; // (a1+a2)
    auto scale2 = lhs.rec_scale;                 // a1
    auto scale3 = rhs.rec_scale;                 // a2
    auto bias1 = lhs.rec_bias + rhs.rec_bias;    // (b1+b2)
    auto bias2 = lhs.rec_bias + rhs.nonrec_bias; // (b1+c2)
    auto bias3 = rhs.rec_bias + lhs.nonrec_bias; // (b2+c1)
    auto maxScale = std::max(scale1, std::max(scale2, scale3));

    // Among those terms that have a maximum scaling factor, determine the
    // largest bias value.
    std::optional<int32_t> maxBias;
    if (enable1 && scale1 == maxScale)
      maxBias = bias1;
    if (enable2 && scale2 == maxScale && (!maxBias || bias2 > *maxBias))
      maxBias = bias2;
    if (enable3 && scale3 == maxScale && (!maxBias || bias3 > *maxBias))
      maxBias = bias3;

    // Pick from the recursive terms the one with maximum scaling factor and
    // minimum bias value.
    auto nonrec_bias = lhs.nonrec_bias + rhs.nonrec_bias; // c1+c2
    auto failed = lhs.failed || rhs.failed;
    if (enable1 && scale1 == maxScale && bias1 == *maxBias)
      return LinIneq(scale1, bias1, nonrec_bias, failed);
    if (enable2 && scale2 == maxScale && bias2 == *maxBias)
      return LinIneq(scale2, bias2, nonrec_bias, failed);
    if (enable3 && scale3 == maxScale && bias3 == *maxBias)
      return LinIneq(scale3, bias3, nonrec_bias, failed);
    return LinIneq(0, 0, nonrec_bias, failed);
  }

  /// Check if the inequality is satisfiable.
  ///
  /// The inequality becomes unsatisfiable if the RHS is ∞, or a>1, or a==1 and
  /// b <= 0. Otherwise there exists as solution for `x` that satisfies the
  /// inequality.
  bool sat() const {
    if (failed)
      return false;
    if (rec_scale > 1)
      return false;
    if (rec_scale == 1 && rec_bias > 0)
      return false;
    return true;
  }

  /// Dump the inequality in human-readable form.
  void print(llvm::raw_ostream &os) const {
    bool any = false;
    bool both = (rec_scale != 0 || rec_bias != 0) && nonrec_bias != 0;
    os << "x >= ";
    if (both)
      os << "max(";
    if (rec_scale != 0) {
      any = true;
      if (rec_scale != 1)
        os << rec_scale << "*";
      os << "x";
    }
    if (rec_bias != 0) {
      if (any) {
        if (rec_bias < 0)
          os << " - " << -rec_bias;
        else
          os << " + " << rec_bias;
      } else {
        any = true;
        os << rec_bias;
      }
    }
    if (both)
      os << ", ";
    if (nonrec_bias != 0) {
      any = true;
      os << nonrec_bias;
    }
    if (both)
      os << ")";
    if (failed) {
      if (any)
        os << " + ";
      os << "∞";
    }
    if (!any)
      os << "0";
  }
};

/// A simple solver for width constraints.
class ConstraintSolver {
public:
  ConstraintSolver() = default;

  VarExpr *var() {
    auto v = vars.alloc();
    exprs.push_back(v);
    if (currentInfo)
      info[v].insert(currentInfo);
    if (currentLoc)
      locs[v].insert(*currentLoc);
    return v;
  }
  DerivedExpr *derived() {
    auto *d = derivs.alloc();
    exprs.push_back(d);
    return d;
  }
  KnownExpr *known(int32_t value) { return alloc<KnownExpr>(knowns, value); }
  IdExpr *id(Expr *arg) { return alloc<IdExpr>(ids, arg); }
  PowExpr *pow(Expr *arg) { return alloc<PowExpr>(uns, arg); }
  AddExpr *add(Expr *lhs, Expr *rhs) { return alloc<AddExpr>(bins, lhs, rhs); }
  MaxExpr *max(Expr *lhs, Expr *rhs) { return alloc<MaxExpr>(bins, lhs, rhs); }
  MinExpr *min(Expr *lhs, Expr *rhs) { return alloc<MinExpr>(bins, lhs, rhs); }

  /// Add a constraint `lhs >= rhs`. Multiple constraints on the same variable
  /// are coalesced into a `max(a, b)` expr.
  Expr *addGeqConstraint(VarExpr *lhs, Expr *rhs) {
    if (lhs->constraint)
      lhs->constraint = max(lhs->constraint, rhs);
    else
      lhs->constraint = id(rhs);
    return lhs->constraint;
  }

  /// Add a constraint `lhs <= rhs`. Multiple constraints on the same variable
  /// are coalesced into a `min(a, b)` expr.
  Expr *addLeqConstraint(VarExpr *lhs, Expr *rhs) {
    if (lhs->upperBound)
      lhs->upperBound = min(lhs->upperBound, rhs);
    else
      lhs->upperBound = id(rhs);
    return lhs->upperBound;
  }

  void dumpConstraints(llvm::raw_ostream &os);
  LogicalResult solve();

  using ContextInfo = DenseMap<Expr *, llvm::SmallSetVector<FieldRef, 1>>;
  const ContextInfo &getContextInfo() const { return info; }
  void setCurrentContextInfo(FieldRef fieldRef) { currentInfo = fieldRef; }
  void setCurrentLocation(std::optional<Location> loc) { currentLoc = loc; }

private:
  // Allocator for constraint expressions.
  llvm::BumpPtrAllocator allocator;
  Allocator<VarExpr> vars = {allocator};
  Allocator<DerivedExpr> derivs = {allocator};
  InternedAllocator<KnownExpr> knowns = {allocator};
  InternedAllocator<IdExpr> ids = {allocator};
  InternedAllocator<UnaryExpr> uns = {allocator};
  InternedAllocator<BinaryExpr> bins = {allocator};

  /// A list of expressions in the order they were created.
  std::vector<Expr *> exprs;
  RootExpr root = {exprs};

  /// Add an allocated expression to the list above.
  template <typename R, typename T, typename... Args>
  R *alloc(InternedAllocator<T> &allocator, Args &&...args) {
    auto it = allocator.template alloc<R>(std::forward<Args>(args)...);
    if (it.second)
      exprs.push_back(it.first);
    if (currentInfo)
      info[it.first].insert(currentInfo);
    if (currentLoc)
      locs[it.first].insert(*currentLoc);
    return it.first;
  }

  /// Contextual information for each expression, indicating which values in the
  /// IR lead to this expression.
  ContextInfo info;
  FieldRef currentInfo = {};
  DenseMap<Expr *, llvm::SmallSetVector<Location, 1>> locs;
  std::optional<Location> currentLoc;

  // Forbid copyign or moving the solver, which would invalidate the refs to
  // allocator held by the allocators.
  ConstraintSolver(ConstraintSolver &&) = delete;
  ConstraintSolver(const ConstraintSolver &) = delete;
  ConstraintSolver &operator=(ConstraintSolver &&) = delete;
  ConstraintSolver &operator=(const ConstraintSolver &) = delete;

  void emitUninferredWidthError(VarExpr *var);

  LinIneq checkCycles(VarExpr *var, Expr *expr,
                      SmallPtrSetImpl<Expr *> &seenVars,
                      InFlightDiagnostic *reportInto = nullptr,
                      unsigned indent = 1);
};

} // namespace

/// Print all constraints in the solver to an output stream.
void ConstraintSolver::dumpConstraints(llvm::raw_ostream &os) {
  for (auto *e : exprs) {
    if (auto *v = dyn_cast<VarExpr>(e)) {
      if (v->constraint)
        os << "- " << *v << " >= " << *v->constraint << "\n";
      else
        os << "- " << *v << " unconstrained\n";
    }
  }
}

#ifndef NDEBUG
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const LinIneq &l) {
  l.print(os);
  return os;
}
#endif

/// Compute the canonicalized linear inequality expression starting at `expr`,
/// for the `var` as the left hand side `x` of the inequality. `seenVars` is
/// used as a recursion breaker. Occurrences of `var` itself within the
/// expression are mapped to the `a` coefficient in the inequality. Any other
/// variables are substituted and, in the presence of a recursion in a variable
/// other than `var`, treated as zero. `info` is a mapping from constraint
/// expressions to values and operations that produced the expression, and is
/// used during error reporting. If `reportInto` is present, the function will
/// additionally attach unsatisfiable inequalities as notes to the diagnostic as
/// it encounters them.
LinIneq ConstraintSolver::checkCycles(VarExpr *var, Expr *expr,
                                      SmallPtrSetImpl<Expr *> &seenVars,
                                      InFlightDiagnostic *reportInto,
                                      unsigned indent) {
  auto ineq =
      TypeSwitch<Expr *, LinIneq>(expr)
          .Case<KnownExpr>([&](auto *expr) { return LinIneq(*expr->solution); })
          .Case<VarExpr>([&](auto *expr) {
            if (expr == var)
              return LinIneq(1, 0); // x >= 1*x + 0
            if (!seenVars.insert(expr).second)
              // Count recursions in other variables as 0. This is sane
              // since the cycle is either breakable, in which case the
              // recursion does not modify the resulting value of the
              // variable, or it is not breakable and will be caught by
              // this very function once it is called on that variable.
              return LinIneq(0);
            if (!expr->constraint)
              // Count unconstrained variables as `x >= 0`.
              return LinIneq(0);
            auto l = checkCycles(var, expr->constraint, seenVars, reportInto,
                                 indent + 1);
            seenVars.erase(expr);
            return l;
          })
          .Case<IdExpr>([&](auto *expr) {
            return checkCycles(var, expr->arg, seenVars, reportInto,
                               indent + 1);
          })
          .Case<PowExpr>([&](auto *expr) {
            // If we can evaluate `2**arg` to a sensible constant, do
            // so. This is the case if a == 0 and c < 31 such that 2**c is
            // representable.
            auto arg =
                checkCycles(var, expr->arg, seenVars, reportInto, indent + 1);
            if (arg.rec_scale != 0 || arg.nonrec_bias < 0 ||
                arg.nonrec_bias >= 31)
              return LinIneq::unsat();
            return LinIneq(1 << arg.nonrec_bias); // x >= 2**arg
          })
          .Case<AddExpr>([&](auto *expr) {
            return LinIneq::add(
                checkCycles(var, expr->lhs(), seenVars, reportInto, indent + 1),
                checkCycles(var, expr->rhs(), seenVars, reportInto,
                            indent + 1));
          })
          .Case<MaxExpr, MinExpr>([&](auto *expr) {
            // Combine the inequalities of the LHS and RHS into a single overly
            // pessimistic inequality. We treat `MinExpr` the same as `MaxExpr`,
            // since `max(a,b)` is an upper bound to `min(a,b)`.
            return LinIneq::max(
                checkCycles(var, expr->lhs(), seenVars, reportInto, indent + 1),
                checkCycles(var, expr->rhs(), seenVars, reportInto,
                            indent + 1));
          })
          .Default([](auto) { return LinIneq::unsat(); });

  // If we were passed an in-flight diagnostic and the current inequality is
  // unsatisfiable, attach notes to the diagnostic indicating the values or
  // operations that contributed to this part of the constraint expression.
  if (reportInto && !ineq.sat()) {
    auto report = [&](Location loc) {
      auto &note = reportInto->attachNote(loc);
      note << "constrained width W >= ";
      if (ineq.rec_scale == -1)
        note << "-";
      if (ineq.rec_scale != 1)
        note << ineq.rec_scale;
      note << "W";
      if (ineq.rec_bias < 0)
        note << "-" << -ineq.rec_bias;
      if (ineq.rec_bias > 0)
        note << "+" << ineq.rec_bias;
      note << " here:";
    };
    auto it = locs.find(expr);
    if (it != locs.end())
      for (auto loc : it->second)
        report(loc);
  }
  if (!reportInto)
    LLVM_DEBUG(llvm::dbgs().indent(indent * 2)
               << "- Visited " << *expr << ": " << ineq << "\n");

  return ineq;
}

using ExprSolution = std::pair<std::optional<int32_t>, bool>;

static ExprSolution
computeUnary(ExprSolution arg, llvm::function_ref<int32_t(int32_t)> operation) {
  if (arg.first)
    arg.first = operation(*arg.first);
  return arg;
}

static ExprSolution
computeBinary(ExprSolution lhs, ExprSolution rhs,
              llvm::function_ref<int32_t(int32_t, int32_t)> operation) {
  auto result = ExprSolution{std::nullopt, lhs.second || rhs.second};
  if (lhs.first && rhs.first)
    result.first = operation(*lhs.first, *rhs.first);
  else if (lhs.first)
    result.first = lhs.first;
  else if (rhs.first)
    result.first = rhs.first;
  return result;
}

/// Compute the value of a constraint `expr`. `seenVars` is used as a recursion
/// breaker. Recursive variables are treated as zero. Returns the computed value
/// and a boolean indicating whether a recursion was detected. This may be used
/// to memoize the result of expressions in case they were not involved in a
/// cycle (which may alter their value from the perspective of a variable).
static ExprSolution solveExpr(Expr *expr, SmallPtrSetImpl<Expr *> &seenVars,
                              unsigned defaultWorklistSize) {

  struct Frame {
    Expr *expr;
    unsigned indent;
  };

  // indent only used for debug logs.
  unsigned indent = 1;
  std::vector<Frame> worklist({{expr, indent}});
  llvm::DenseMap<Expr *, ExprSolution> solvedExprs;
  // Reserving the vector size, to avoid frequent reallocs. The worklist can be
  // quite large.
  worklist.reserve(defaultWorklistSize);

  while (!worklist.empty()) {
    auto &frame = worklist.back();
    auto setSolution = [&](ExprSolution solution) {
      // Memoize the result.
      if (solution.first && !solution.second)
        frame.expr->solution = *solution.first;
      solvedExprs[frame.expr] = solution;

      // Produce some useful debug prints.
      LLVM_DEBUG({
        if (!isa<KnownExpr>(frame.expr)) {
          if (solution.first)
            llvm::dbgs().indent(frame.indent * 2)
                << "= Solved " << *frame.expr << " = " << *solution.first;
          else
            llvm::dbgs().indent(frame.indent * 2)
                << "= Skipped " << *frame.expr;
          llvm::dbgs() << " (" << (solution.second ? "cycle broken" : "unique")
                       << ")\n";
        }
      });

      worklist.pop_back();
    };

    // See if we have a memoized result we can return.
    if (frame.expr->solution) {
      LLVM_DEBUG({
        if (!isa<KnownExpr>(frame.expr))
          llvm::dbgs().indent(indent * 2) << "- Cached " << *frame.expr << " = "
                                          << *frame.expr->solution << "\n";
      });
      setSolution(ExprSolution{*frame.expr->solution, false});
      continue;
    }

    // Otherwise compute the value of the expression.
    LLVM_DEBUG({
      if (!isa<KnownExpr>(frame.expr))
        llvm::dbgs().indent(frame.indent * 2)
            << "- Solving " << *frame.expr << "\n";
    });

    TypeSwitch<Expr *>(frame.expr)
        .Case<KnownExpr>([&](auto *expr) {
          setSolution(ExprSolution{*expr->solution, false});
        })
        .Case<VarExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->constraint)) {
            auto solution = solvedExprs[expr->constraint];
            // If we've solved the upper bound already, store the solution.
            // This will be explicitly solved for later if not computed as
            // part of the solving that resolved this constraint.
            // This should only happen if somehow the constraint is
            // solved before visiting this expression, so that our upperBound
            // was not added to the worklist such that it was handled first.
            if (expr->upperBound && solvedExprs.contains(expr->upperBound))
              expr->upperBoundSolution = solvedExprs[expr->upperBound].first;
            seenVars.erase(expr);
            // Constrain variables >= 0.
            if (solution.first && *solution.first < 0)
              solution.first = 0;
            return setSolution(solution);
          }

          // Unconstrained variables produce no solution.
          if (!expr->constraint)
            return setSolution(ExprSolution{std::nullopt, false});
          // Return no solution for recursions in the variables. This is sane
          // and will cause the expression to be ignored when computing the
          // parent, e.g. `a >= max(a, 1)` will become just `a >= 1`.
          if (!seenVars.insert(expr).second)
            return setSolution(ExprSolution{std::nullopt, true});

          worklist.push_back({expr->constraint, indent + 1});
          if (expr->upperBound)
            worklist.push_back({expr->upperBound, indent + 1});
        })
        .Case<IdExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->arg))
            return setSolution(solvedExprs[expr->arg]);
          worklist.push_back({expr->arg, indent + 1});
        })
        .Case<PowExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->arg))
            return setSolution(computeUnary(
                solvedExprs[expr->arg], [](int32_t arg) { return 1 << arg; }));

          worklist.push_back({expr->arg, indent + 1});
        })
        .Case<AddExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->lhs()) &&
              solvedExprs.contains(expr->rhs()))
            return setSolution(computeBinary(
                solvedExprs[expr->lhs()], solvedExprs[expr->rhs()],
                [](int32_t lhs, int32_t rhs) { return lhs + rhs; }));

          worklist.push_back({expr->lhs(), indent + 1});
          worklist.push_back({expr->rhs(), indent + 1});
        })
        .Case<MaxExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->lhs()) &&
              solvedExprs.contains(expr->rhs()))
            return setSolution(computeBinary(
                solvedExprs[expr->lhs()], solvedExprs[expr->rhs()],
                [](int32_t lhs, int32_t rhs) { return std::max(lhs, rhs); }));

          worklist.push_back({expr->lhs(), indent + 1});
          worklist.push_back({expr->rhs(), indent + 1});
        })
        .Case<MinExpr>([&](auto *expr) {
          if (solvedExprs.contains(expr->lhs()) &&
              solvedExprs.contains(expr->rhs()))
            return setSolution(computeBinary(
                solvedExprs[expr->lhs()], solvedExprs[expr->rhs()],
                [](int32_t lhs, int32_t rhs) { return std::min(lhs, rhs); }));

          worklist.push_back({expr->lhs(), indent + 1});
          worklist.push_back({expr->rhs(), indent + 1});
        })
        .Default([&](auto) {
          setSolution(ExprSolution{std::nullopt, false});
        });
  }

  return solvedExprs[expr];
}

/// Solve the constraint problem. This is a very simple implementation that
/// does not fully solve the problem if there are weird dependency cycles
/// present.
LogicalResult ConstraintSolver::solve() {
  LLVM_DEBUG({
    llvm::dbgs() << "\n===----- Constraints -----===\n\n";
    dumpConstraints(llvm::dbgs());
  });

  // Ensure that there are no adverse cycles around.
  LLVM_DEBUG(
      llvm::dbgs() << "\n===----- Checking for unbreakable loops -----===\n\n");
  SmallPtrSet<Expr *, 16> seenVars;
  bool anyFailed = false;

  for (auto *expr : exprs) {
    // Only work on variables.
    auto *var = dyn_cast<VarExpr>(expr);
    if (!var || !var->constraint)
      continue;
    LLVM_DEBUG(llvm::dbgs()
               << "- Checking " << *var << " >= " << *var->constraint << "\n");

    // Canonicalize the variable's constraint expression into a form that allows
    // us to easily determine if any recursion leads to an unsatisfiable
    // constraint. The `seenVars` set acts as a recursion breaker.
    seenVars.insert(var);
    auto ineq = checkCycles(var, var->constraint, seenVars);
    seenVars.clear();

    // If the constraint is satisfiable, we're done.
    // TODO: It's possible that this result is already sufficient to arrive at a
    // solution for the constraint, and the second pass further down is not
    // necessary. This would require more proper handling of `MinExpr` in the
    // cycle checking code.
    if (ineq.sat()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  = Breakable since " << ineq << " satisfiable\n");
      continue;
    }

    // If we arrive here, the constraint is not satisfiable at all. To provide
    // some guidance to the user, we call the cycle checking code again, but
    // this time with an in-flight diagnostic to attach notes indicating
    // unsatisfiable paths in the cycle.
    LLVM_DEBUG(llvm::dbgs()
               << "  = UNBREAKABLE since " << ineq << " unsatisfiable\n");
    anyFailed = true;
    for (auto fieldRef : info.find(var)->second) {
      // Depending on whether this value stems from an operation or not, create
      // an appropriate diagnostic identifying the value.
      auto op = fieldRef.getDefiningOp();
      auto diag = op ? op->emitOpError()
                     : mlir::emitError(fieldRef.getValue().getLoc())
                           << "value ";
      diag << "is constrained to be wider than itself";

      // Re-run the cycle checking, but this time reporting into the diagnostic.
      seenVars.insert(var);
      checkCycles(var, var->constraint, seenVars, &diag);
      seenVars.clear();
    }
  }

  // If there were cycles, return now to avoid complaining to the user about
  // dependent widths not being inferred.
  if (anyFailed)
    return failure();

  // Iterate over the constraint variables and solve each.
  LLVM_DEBUG(llvm::dbgs() << "\n===----- Solving constraints -----===\n\n");
  unsigned defaultWorklistSize = exprs.size() / 2;
  for (auto *expr : exprs) {
    // Only work on variables.
    auto *var = dyn_cast<VarExpr>(expr);
    if (!var)
      continue;

    // Complain about unconstrained variables.
    if (!var->constraint) {
      LLVM_DEBUG(llvm::dbgs() << "- Unconstrained " << *var << "\n");
      emitUninferredWidthError(var);
      anyFailed = true;
      continue;
    }

    // Compute the value for the variable.
    LLVM_DEBUG(llvm::dbgs()
               << "- Solving " << *var << " >= " << *var->constraint << "\n");
    seenVars.insert(var);
    auto solution = solveExpr(var->constraint, seenVars, defaultWorklistSize);
    // Compute the upperBound if there is one and haven't already.
    if (var->upperBound && !var->upperBoundSolution)
      var->upperBoundSolution =
          solveExpr(var->upperBound, seenVars, defaultWorklistSize).first;
    seenVars.clear();

    // Constrain variables >= 0.
    if (solution.first && *solution.first < 0)
      solution.first = 0;
    var->solution = solution.first;

    // In case the width could not be inferred, complain to the user. This might
    // be the case if the width depends on an unconstrained variable.
    if (!solution.first) {
      LLVM_DEBUG(llvm::dbgs() << "  - UNSOLVED " << *var << "\n");
      emitUninferredWidthError(var);
      anyFailed = true;
      continue;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "  = Solved " << *var << " = " << solution.first << " ("
               << (solution.second ? "cycle broken" : "unique") << ")\n");

    // Check if the solution we have found violates an upper bound.
    if (var->upperBoundSolution && var->upperBoundSolution < *solution.first) {
      LLVM_DEBUG(llvm::dbgs() << "  ! Unsatisfiable " << *var
                              << " <= " << var->upperBoundSolution << "\n");
      emitUninferredWidthError(var);
      anyFailed = true;
    }
  }

  // Copy over derived widths.
  for (auto *expr : exprs) {
    // Only work on derived values.
    auto *derived = dyn_cast<DerivedExpr>(expr);
    if (!derived)
      continue;

    auto *assigned = derived->assigned;
    if (!assigned || !assigned->solution) {
      LLVM_DEBUG(llvm::dbgs() << "- Unused " << *derived << " set to 0\n");
      derived->solution = 0;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "- Deriving " << *derived << " = "
                              << assigned->solution << "\n");
      derived->solution = *assigned->solution;
    }
  }

  return failure(anyFailed);
}

// Emits the diagnostic to inform the user about an uninferred width in the
// design. Returns true if an error was reported, false otherwise.
void ConstraintSolver::emitUninferredWidthError(VarExpr *var) {
  FieldRef fieldRef = info.find(var)->second.back();
  Value value = fieldRef.getValue();

  auto diag = mlir::emitError(value.getLoc(), "uninferred width:");

  // Try to hint the user at what kind of node this is.
  if (isa<BlockArgument>(value)) {
    diag << " port";
  } else if (auto op = value.getDefiningOp()) {
    TypeSwitch<Operation *>(op)
        .Case<WireOp>([&](auto) { diag << " wire"; })
        .Case<RegOp, RegResetOp>([&](auto) { diag << " reg"; })
        .Case<NodeOp>([&](auto) { diag << " node"; })
        .Default([&](auto) { diag << " value"; });
  } else {
    diag << " value";
  }

  // Actually print what the user can refer to.
  auto [fieldName, rootKnown] = getFieldName(fieldRef);
  if (!fieldName.empty()) {
    if (!rootKnown)
      diag << " field";
    diag << " \"" << fieldName << "\"";
  }

  if (!var->constraint) {
    diag << " is unconstrained";
  } else if (var->solution && var->upperBoundSolution &&
             var->solution > var->upperBoundSolution) {
    diag << " cannot satisfy all width requirements";
    LLVM_DEBUG(llvm::dbgs() << *var->constraint << "\n");
    LLVM_DEBUG(llvm::dbgs() << *var->upperBound << "\n");
    auto loc = locs.find(var->constraint)->second.back();
    diag.attachNote(loc) << "width is constrained to be at least "
                         << *var->solution << " here:";
    loc = locs.find(var->upperBound)->second.back();
    diag.attachNote(loc) << "width is constrained to be at most "
                         << *var->upperBoundSolution << " here:";
  } else {
    diag << " width cannot be determined";
    LLVM_DEBUG(llvm::dbgs() << *var->constraint << "\n");
    auto loc = locs.find(var->constraint)->second.back();
    diag.attachNote(loc) << "width is constrained by an uninferred width here:";
  }
}

//===----------------------------------------------------------------------===//
// Inference Constraint Problem Mapping
//===----------------------------------------------------------------------===//

namespace {

/// A helper class which maps the types and operations in a design to a set of
/// variables and constraints to be solved later.
class InferenceMapping {
public:
  InferenceMapping(ConstraintSolver &solver, SymbolTable &symtbl)
      : solver(solver), symtbl(symtbl) {}

  LogicalResult map(CircuitOp op);
  LogicalResult mapOperation(Operation *op);

  /// Declare all the variables in the value. If the value is a ground type,
  /// there is a single variable declared.  If the value is an aggregate type,
  /// it sets up variables for each unknown width.
  void declareVars(Value value, Location loc, bool isDerived = false);

  /// Declare a variable associated with a specific field of an aggregate.
  Expr *declareVar(FieldRef fieldRef, Location loc);

  /// Declarate a variable for a type with an unknown width.  The type must be a
  /// non-aggregate.
  Expr *declareVar(FIRRTLType type, Location loc);

  /// Assign the constraint expressions of the fields in the `result` argument
  /// as the max of expressions in the `rhs` and `lhs` arguments. Both fields
  /// must be the same type.
  void maximumOfTypes(Value result, Value rhs, Value lhs);

  /// Constrain the value "larger" to be greater than or equal to "smaller".
  /// These may be aggregate values. This is used for regular connects.
  void constrainTypes(Value larger, Value smaller, bool equal = false);

  /// Constrain the expression "larger" to be greater than or equals to
  /// the expression "smaller".
  void constrainTypes(Expr *larger, Expr *smaller,
                      bool imposeUpperBounds = false, bool equal = false);

  /// Assign the constraint expressions of the fields in the `src` argument as
  /// the expressions for the `dst` argument. Both fields must be of the given
  /// `type`.
  void unifyTypes(FieldRef lhs, FieldRef rhs, FIRRTLType type);

  /// Get the expr associated with the value.  The value must be a non-aggregate
  /// type.
  Expr *getExpr(Value value) const;

  /// Get the expr associated with a specific field in a value.
  Expr *getExpr(FieldRef fieldRef) const;

  /// Get the expr associated with a specific field in a value. If value is
  /// NULL, then this returns NULL.
  Expr *getExprOrNull(FieldRef fieldRef) const;

  /// Set the expr associated with the value. The value must be a non-aggregate
  /// type.
  void setExpr(Value value, Expr *expr);

  /// Set the expr associated with a specific field in a value.
  void setExpr(FieldRef fieldRef, Expr *expr);

  /// Return whether a module was skipped due to being fully inferred already.
  bool isModuleSkipped(FModuleOp module) const {
    return skippedModules.count(module);
  }

  /// Return whether all modules in the mapping were fully inferred.
  bool areAllModulesSkipped() const { return allModulesSkipped; }

private:
  /// The constraint solver into which we emit variables and constraints.
  ConstraintSolver &solver;

  /// The constraint exprs for each result type of an operation.
  DenseMap<FieldRef, Expr *> opExprs;

  /// The fully inferred modules that were skipped entirely.
  SmallPtrSet<Operation *, 16> skippedModules;
  bool allModulesSkipped = true;

  /// Cache of module symbols
  SymbolTable &symtbl;
};

} // namespace

/// Check if a type contains any FIRRTL type with uninferred widths.
static bool hasUninferredWidth(Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<FIRRTLBaseType>([](auto base) { return base.hasUninferredWidth(); })
      .Case<RefType>(
          [](auto ref) { return ref.getType().hasUninferredWidth(); })
      .Default([](auto) { return false; });
}

LogicalResult InferenceMapping::map(CircuitOp op) {
  LLVM_DEBUG(llvm::dbgs()
             << "\n===----- Mapping ops to constraint exprs -----===\n\n");

  // Ensure we have constraint variables established for all module ports.
  for (auto module : op.getOps<FModuleOp>())
    for (auto arg : module.getArguments()) {
      solver.setCurrentContextInfo(FieldRef(arg, 0));
      declareVars(arg, module.getLoc());
    }

  for (auto module : op.getOps<FModuleOp>()) {
    // Check if the module contains *any* uninferred widths. This allows us to
    // do an early skip if the module is already fully inferred.
    bool anyUninferred = false;
    for (auto arg : module.getArguments()) {
      anyUninferred |= hasUninferredWidth(arg.getType());
      if (anyUninferred)
        break;
    }
    module.walk([&](Operation *op) {
      for (auto type : op->getResultTypes())
        anyUninferred |= hasUninferredWidth(type);
      if (anyUninferred)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (!anyUninferred) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping fully-inferred module '"
                              << module.getName() << "'\n");
      skippedModules.insert(module);
      continue;
    }

    allModulesSkipped = false;

    // Go through operations in the module, creating type variables for results,
    // and generating constraints.
    auto result = module.getBodyBlock()->walk(
        [&](Operation *op) { return WalkResult(mapOperation(op)); });
    if (result.wasInterrupted())
      return failure();
  }

  return success();
}

LogicalResult InferenceMapping::mapOperation(Operation *op) {
  // In case the operation result has a type without uninferred widths, don't
  // even bother to populate the constraint problem and treat that as a known
  // size directly. This is done in `declareVars`, which will generate
  // `KnownExpr` nodes for all known widths -- which are the only ones in this
  // case.
  bool allWidthsKnown = true;
  for (auto result : op->getResults()) {
    if (isa<MuxPrimOp, Mux4CellIntrinsicOp, Mux2CellIntrinsicOp>(op))
      if (hasUninferredWidth(op->getOperand(0).getType()))
        allWidthsKnown = false;
    // Only consider FIRRTL types for width constraints. Ignore any foreign
    // types as they don't participate in the width inference process.
    auto resultTy = type_dyn_cast<FIRRTLType>(result.getType());
    if (!resultTy)
      continue;
    if (!hasUninferredWidth(resultTy))
      declareVars(result, op->getLoc());
    else
      allWidthsKnown = false;
  }
  if (allWidthsKnown && !isa<FConnectLike, AttachOp>(op))
    return success();

  // Actually generate the necessary constraint expressions.
  bool mappingFailed = false;
  solver.setCurrentContextInfo(
      op->getNumResults() > 0 ? FieldRef(op->getResults()[0], 0) : FieldRef());
  solver.setCurrentLocation(op->getLoc());
  TypeSwitch<Operation *>(op)
      .Case<ConstantOp>([&](auto op) {
        // If the constant has a known width, use that. Otherwise pick the
        // smallest number of bits necessary to represent the constant.
        Expr *e;
        if (auto width = op.getType().get().getWidth())
          e = solver.known(*width);
        else {
          auto v = op.getValue();
          auto w = v.getBitWidth() - (v.isNegative() ? v.countLeadingOnes()
                                                     : v.countLeadingZeros());
          if (v.isSigned())
            w += 1;
          e = solver.known(std::max(w, 1u));
        }
        setExpr(op.getResult(), e);
      })
      .Case<SpecialConstantOp>([&](auto op) {
        // Nothing required.
      })
      .Case<InvalidValueOp>([&](auto op) {
        // We must duplicate the invalid value for each use, since each use can
        // be inferred to a different width.
        if (!hasUninferredWidth(op.getType()))
          return;
        declareVars(op.getResult(), op.getLoc(), /*isDerived=*/true);
        if (op.use_empty())
          return;

        auto type = op.getType();
        ImplicitLocOpBuilder builder(op->getLoc(), op);
        for (auto &use :
             llvm::make_early_inc_range(llvm::drop_begin(op->getUses()))) {
          // - `make_early_inc_range` since `getUses()` is invalidated upon
          //   `use.set(...)`.
          // - `drop_begin` such that the first use can keep the original op.
          auto clone = builder.create<InvalidValueOp>(type);
          declareVars(clone.getResult(), clone.getLoc(),
                      /*isDerived=*/true);
          use.set(clone);
        }
      })
      .Case<WireOp, RegOp>(
          [&](auto op) { declareVars(op.getResult(), op.getLoc()); })
      .Case<RegResetOp>([&](auto op) {
        // The original Scala code also constrains the reset signal to be at
        // least 1 bit wide. We don't do this here since the MLIR FIRRTL
        // dialect enforces the reset signal to be an async reset or a
        // `uint<1>`.
        declareVars(op.getResult(), op.getLoc());
        // Contrain the register to be greater than or equal to the reset
        // signal.
        constrainTypes(op.getResult(), op.getResetValue());
      })
      .Case<NodeOp>([&](auto op) {
        // Nodes have the same type as their input.
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.getInput(), 0),
                   op.getResult().getType());
      })

      // Aggregate Values
      .Case<SubfieldOp>([&](auto op) {
        BundleType bundleType = op.getInput().getType();
        auto fieldID = bundleType.getFieldID(op.getFieldIndex());
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(op.getInput(), fieldID), op.getType());
      })
      .Case<SubindexOp, SubaccessOp>([&](auto op) {
        // All vec fields unify to the same thing. Always use the first element
        // of the vector, which has a field ID of 1.
        unifyTypes(FieldRef(op.getResult(), 0), FieldRef(op.getInput(), 1),
                   op.getType());
      })
      .Case<SubtagOp>([&](auto op) {
        FEnumType enumType = op.getInput().getType();
        auto fieldID = enumType.getFieldID(op.getFieldIndex());
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(op.getInput(), fieldID), op.getType());
      })

      .Case<RefSubOp>([&](RefSubOp op) {
        uint64_t fieldID = TypeSwitch<FIRRTLBaseType, uint64_t>(
                               op.getInput().getType().getType())
                               .Case<FVectorType>([](auto _) { return 1; })
                               .Case<BundleType>([&](auto type) {
                                 return type.getFieldID(op.getIndex());
                               });
        unifyTypes(FieldRef(op.getResult(), 0),
                   FieldRef(op.getInput(), fieldID), op.getType());
      })

      // Arithmetic and Logical Binary Primitives
      .Case<AddPrimOp, SubPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(solver.max(lhs, rhs), solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<MulPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DivPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        Expr *e;
        if (op.getType().get().isSigned()) {
          e = solver.add(lhs, solver.known(1));
        } else {
          e = lhs;
        }
        setExpr(op.getResult(), e);
      })
      .Case<RemPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.min(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<AndPrimOp, OrPrimOp, XorPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.max(lhs, rhs);
        setExpr(op.getResult(), e);
      })

      // Misc Binary Primitives
      .Case<CatPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(lhs, rhs);
        setExpr(op.getResult(), e);
      })
      .Case<DShlPrimOp>([&](auto op) {
        auto lhs = getExpr(op.getLhs());
        auto rhs = getExpr(op.getRhs());
        auto e = solver.add(lhs, solver.add(solver.pow(rhs), solver.known(-1)));
        setExpr(op.getResult(), e);
      })
      .Case<DShlwPrimOp, DShrPrimOp>([&](auto op) {
        auto e = getExpr(op.getLhs());
        setExpr(op.getResult(), e);
      })

      // Unary operators
      .Case<NegPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })
      .Case<CvtPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = op.getInput().getType().get().isSigned()
                     ? input
                     : solver.add(input, solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Miscellaneous
      .Case<BitsPrimOp>([&](auto op) {
        setExpr(op.getResult(), solver.known(op.getHi() - op.getLo() + 1));
      })
      .Case<HeadPrimOp>([&](auto op) {
        setExpr(op.getResult(), solver.known(op.getAmount()));
      })
      .Case<TailPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.add(input, solver.known(-op.getAmount()));
        setExpr(op.getResult(), e);
      })
      .Case<PadPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.max(input, solver.known(op.getAmount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShlPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.add(input, solver.known(op.getAmount()));
        setExpr(op.getResult(), e);
      })
      .Case<ShrPrimOp>([&](auto op) {
        auto input = getExpr(op.getInput());
        auto e = solver.max(solver.add(input, solver.known(-op.getAmount())),
                            solver.known(1));
        setExpr(op.getResult(), e);
      })

      // Handle operations whose output width matches the input width.
      .Case<NotPrimOp, AsSIntPrimOp, AsUIntPrimOp, ConstCastOp>(
          [&](auto op) { setExpr(op.getResult(), getExpr(op.getInput())); })
      .Case<mlir::UnrealizedConversionCastOp>(
          [&](auto op) { setExpr(op.getResult(0), getExpr(op.getOperand(0))); })

      // Handle operations with a single result type that always has a
      // well-known width.
      .Case<LEQPrimOp, LTPrimOp, GEQPrimOp, GTPrimOp, EQPrimOp, NEQPrimOp,
            AsClockPrimOp, AsAsyncResetPrimOp, AndRPrimOp, OrRPrimOp,
            XorRPrimOp>([&](auto op) {
        auto width = op.getType().getBitWidthOrSentinel();
        assert(width > 0 && "width should have been checked by verifier");
        setExpr(op.getResult(), solver.known(width));
      })
      .Case<MuxPrimOp, Mux2CellIntrinsicOp>([&](auto op) {
        auto *sel = getExpr(op.getSel());
        constrainTypes(sel, solver.known(1));
        maximumOfTypes(op.getResult(), op.getHigh(), op.getLow());
      })
      .Case<Mux4CellIntrinsicOp>([&](Mux4CellIntrinsicOp op) {
        auto *sel = getExpr(op.getSel());
        constrainTypes(sel, solver.known(2));
        maximumOfTypes(op.getResult(), op.getV3(), op.getV2());
        maximumOfTypes(op.getResult(), op.getResult(), op.getV1());
        maximumOfTypes(op.getResult(), op.getResult(), op.getV0());
      })

      .Case<ConnectOp, StrictConnectOp>(
          [&](auto op) { constrainTypes(op.getDest(), op.getSrc()); })
      .Case<RefDefineOp>([&](auto op) {
        // Dest >= Src, but also check Src <= Dest for correctness
        // (but don't solve to make this true, don't back-propagate)
        constrainTypes(op.getDest(), op.getSrc(), true);
      })
      // StrictConnect is an identify constraint
      .Case<StrictConnectOp>([&](auto op) {
        // This back-propagates width from destination to source,
        // causing source to sometimes be inferred wider than
        // it should be (https://github.com/llvm/circt/issues/5391).
        // This is needed to push the width into feeding widthCast?
        constrainTypes(op.getDest(), op.getSrc());
        constrainTypes(op.getSrc(), op.getDest());
      })
      .Case<AttachOp>([&](auto op) {
        // Attach connects multiple analog signals together. All signals must
        // have the same bit width. Signals without bit width inherit from the
        // other signals.
        if (op.getAttached().empty())
          return;
        auto prev = op.getAttached()[0];
        for (auto operand : op.getAttached().drop_front()) {
          auto e1 = getExpr(prev);
          auto e2 = getExpr(operand);
          constrainTypes(e1, e2, /*imposeUpperBounds=*/true);
          constrainTypes(e2, e1, /*imposeUpperBounds=*/true);
          prev = operand;
        }
      })

      // Handle the no-ops that don't interact with width inference.
      .Case<PrintFOp, SkipOp, StopOp, WhenOp, AssertOp, AssumeOp, CoverOp>(
          [&](auto) {})

      // Handle instances of other modules.
      .Case<InstanceOp>([&](auto op) {
        auto refdModule = op.getReferencedModule(symtbl);
        auto module = dyn_cast<FModuleOp>(&*refdModule);
        if (!module) {
          auto diag = mlir::emitError(op.getLoc());
          diag << "extern module `" << op.getModuleName()
               << "` has ports of uninferred width";

          auto fml = cast<FModuleLike>(&*refdModule);
          auto ports = fml.getPorts();
          for (auto &port : ports)
            if (type_cast<FIRRTLBaseType>(port.type).hasUninferredWidth()) {
              diag.attachNote(op.getLoc()) << "Port: " << port.name;
              if (!type_cast<FIRRTLBaseType>(port.type).isGround())
                diagnoseUninferredType(diag, port.type, port.name.getValue());
            }

          diag.attachNote(op.getLoc())
              << "Only non-extern FIRRTL modules may contain unspecified "
                 "widths to be inferred automatically.";
          diag.attachNote(refdModule->getLoc())
              << "Module `" << op.getModuleName() << "` defined here:";
          mappingFailed = true;
          return;
        }
        // Simply look up the free variables created for the instantiated
        // module's ports, and use them for instance port wires. This way,
        // constraints imposed onto the ports of the instance will transparently
        // apply to the ports of the instantiated module.
        for (auto it : llvm::zip(op->getResults(), module.getArguments())) {
          unifyTypes(FieldRef(std::get<0>(it), 0), FieldRef(std::get<1>(it), 0),
                     type_cast<FIRRTLType>(std::get<0>(it).getType()));
        }
      })

      // Handle memories.
      .Case<MemOp>([&](MemOp op) {
        // Create constraint variables for all ports.
        unsigned nonDebugPort = 0;
        for (const auto &result : llvm::enumerate(op.getResults())) {
          declareVars(result.value(), op.getLoc());
          if (!type_isa<RefType>(result.value().getType()))
            nonDebugPort = result.index();
        }

        // A helper function that returns the indeces of the "data", "rdata",
        // and "wdata" fields in the bundle corresponding to a memory port.
        auto dataFieldIndices = [](MemOp::PortKind kind) -> ArrayRef<unsigned> {
          static const unsigned indices[] = {3, 5};
          static const unsigned debug[] = {0};
          switch (kind) {
          case MemOp::PortKind::Read:
          case MemOp::PortKind::Write:
            return ArrayRef<unsigned>(indices, 1); // {3}
          case MemOp::PortKind::ReadWrite:
            return ArrayRef<unsigned>(indices); // {3, 5}
          case MemOp::PortKind::Debug:
            return ArrayRef<unsigned>(debug);
          }
          llvm_unreachable("Imposible PortKind");
        };

        // This creates independent variables for every data port. Yet, what we
        // actually want is for all data ports to share the same variable. To do
        // this, we find the first data port declared, and use that port's vars
        // for all the other ports.
        unsigned firstFieldIndex =
            dataFieldIndices(op.getPortKind(nonDebugPort))[0];
        FieldRef firstData(
            op.getResult(nonDebugPort),
            type_cast<BundleType>(op.getPortType(nonDebugPort).getPassiveType())
                .getFieldID(firstFieldIndex));
        LLVM_DEBUG(llvm::dbgs() << "Adjusting memory port variables:\n");

        // Reuse data port variables.
        auto dataType = op.getDataType();
        for (unsigned i = 0, e = op.getResults().size(); i < e; ++i) {
          auto result = op.getResult(i);
          if (type_isa<RefType>(result.getType())) {
            // Debug ports are firrtl.ref<vector<data-type, depth>>
            // Use FieldRef of 1, to indicate the first vector element must be
            // of the dataType.
            unifyTypes(firstData, FieldRef(result, 1), dataType);
            continue;
          }

          auto portType =
              type_cast<BundleType>(op.getPortType(i).getPassiveType());
          for (auto fieldIndex : dataFieldIndices(op.getPortKind(i)))
            unifyTypes(FieldRef(result, portType.getFieldID(fieldIndex)),
                       firstData, dataType);
        }
      })

      .Case<RefSendOp>([&](auto op) {
        declareVars(op.getResult(), op.getLoc());
        constrainTypes(op.getResult(), op.getBase(), true);
      })
      .Case<RefResolveOp>([&](auto op) {
        declareVars(op.getResult(), op.getLoc());
        constrainTypes(op.getResult(), op.getRef(), true);
      })
      .Case<RefCastOp>([&](auto op) {
        declareVars(op.getResult(), op.getLoc());
        constrainTypes(op.getResult(), op.getInput(), true);
      })
      .Case<mlir::UnrealizedConversionCastOp>([&](auto op) {
        for (Value result : op.getResults()) {
          auto ty = result.getType();
          if (type_isa<FIRRTLType>(ty))
            declareVars(result, op.getLoc());
        }
      })
      .Default([&](auto op) {
        op->emitOpError("not supported in width inference");
        mappingFailed = true;
      });

  // Forceable declarations should have the ref constrained to data result.
  if (auto fop = dyn_cast<Forceable>(op); fop && fop.isForceable())
    unifyTypes(FieldRef(fop.getDataRef(), 0), FieldRef(fop.getDataRaw(), 0),
               fop.getDataType());

  return failure(mappingFailed);
}

/// Declare free variables for the type of a value, and associate the resulting
/// set of variables with that value.
void InferenceMapping::declareVars(Value value, Location loc, bool isDerived) {
  // Declare a variable for every unknown width in the type. If this is a Bundle
  // type or a FVector type, we will have to potentially create many variables.
  unsigned fieldID = 0;
  std::function<void(FIRRTLBaseType)> declare = [&](FIRRTLBaseType type) {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      // Known width integer create a known expression.
      setExpr(FieldRef(value, fieldID), solver.known(width));
      fieldID++;
    } else if (width == -1) {
      // Unknown width integers create a variable.
      FieldRef field(value, fieldID);
      solver.setCurrentContextInfo(field);
      if (isDerived)
        setExpr(field, solver.derived());
      else
        setExpr(field, solver.var());
      fieldID++;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      // Bundle types recursively declare all bundle elements.
      fieldID++;
      for (auto &element : bundleType) {
        declare(element.type);
      }
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      declare(vecType.getElementType());
      // Skip past the rest of the elements
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      for (auto &element : enumType.getElements())
        declare(element.type);
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto type = getBaseType(value.getType()))
    declare(type);
}

/// Assign the constraint expressions of the fields in the `result` argument as
/// the max of expressions in the `rhs` and `lhs` arguments. Both fields must be
/// the same type.
void InferenceMapping::maximumOfTypes(Value result, Value rhs, Value lhs) {
  // Recurse to every leaf element and set larger >= smaller.
  auto fieldID = 0;
  std::function<void(FIRRTLBaseType)> maximize = [&](FIRRTLBaseType type) {
    if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      fieldID++;
      for (auto &element : bundleType.getElements())
        maximize(element.type);
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0)
        maximize(vecType.getElementType());
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      for (auto &element : enumType.getElements())
        maximize(element.type);
    } else if (type.isGround()) {
      auto *e = solver.max(getExpr(FieldRef(rhs, fieldID)),
                           getExpr(FieldRef(lhs, fieldID)));
      setExpr(FieldRef(result, fieldID), e);
      fieldID++;
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto type = getBaseType(result.getType()))
    maximize(type);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type. Types have to be
/// compatible in the sense that they may only differ in the presence or absence
/// of bit widths.
///
/// This function is used to apply regular connects.
/// Set `equal` for constraining larger <= smaller for correctness but not
/// solving.
void InferenceMapping::constrainTypes(Value larger, Value smaller, bool equal) {
  // Recurse to every leaf element and set larger >= smaller. Ignore foreign
  // types as these do not participate in width inference.

  auto fieldID = 0;
  std::function<void(FIRRTLBaseType, Value, Value)> constrain =
      [&](FIRRTLBaseType type, Value larger, Value smaller) {
        if (auto bundleType = type_dyn_cast<BundleType>(type)) {
          fieldID++;
          for (auto &element : bundleType.getElements()) {
            if (element.isFlip)
              constrain(element.type, smaller, larger);
            else
              constrain(element.type, larger, smaller);
          }
        } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
          fieldID++;
          auto save = fieldID;
          // Skip 0 length vectors.
          if (vecType.getNumElements() > 0) {
            constrain(vecType.getElementType(), larger, smaller);
          }
          fieldID = save + vecType.getMaxFieldID();
        } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
          fieldID++;
          for (auto &element : enumType.getElements())
            constrain(element.type, larger, smaller);
        } else if (type.isGround()) {
          // Leaf element, look up their expressions, and create the constraint.
          constrainTypes(getExpr(FieldRef(larger, fieldID)),
                         getExpr(FieldRef(smaller, fieldID)), false, equal);
          fieldID++;
        } else {
          llvm_unreachable("Unknown type inside a bundle!");
        }
      };

  if (auto type = getBaseType(larger.getType()))
    constrain(type, larger, smaller);
}

/// Establishes constraints to ensure the sizes in the `larger` type are greater
/// than or equal to the sizes in the `smaller` type.
void InferenceMapping::constrainTypes(Expr *larger, Expr *smaller,
                                      bool imposeUpperBounds, bool equal) {
  assert(larger && "Larger expression should be specified");
  assert(smaller && "Smaller expression should be specified");

  // If one of the sides is `DerivedExpr`, simply assign the other side as the
  // derived width. This allows `InvalidValueOp`s to properly infer their width
  // from the connects they are used in, but also be inferred to something
  // useful on their own.
  if (auto *largerDerived = dyn_cast<DerivedExpr>(larger)) {
    largerDerived->assigned = smaller;
    LLVM_DEBUG(llvm::dbgs() << "Deriving " << *largerDerived << " from "
                            << *smaller << "\n");
    return;
  }
  if (auto *smallerDerived = dyn_cast<DerivedExpr>(smaller)) {
    smallerDerived->assigned = larger;
    LLVM_DEBUG(llvm::dbgs() << "Deriving " << *smallerDerived << " from "
                            << *larger << "\n");
    return;
  }

  // If the larger expr is a free variable, create a `expr >= x` constraint for
  // it that we can try to satisfy with the smallest width.
  if (auto largerVar = dyn_cast<VarExpr>(larger)) {
    LLVM_ATTRIBUTE_UNUSED auto *c = solver.addGeqConstraint(largerVar, smaller);
    LLVM_DEBUG(llvm::dbgs()
               << "Constrained " << *largerVar << " >= " << *c << "\n");
    // If we're constraining larger == smaller, add the LEQ contraint as well.
    // Solve for GEQ but check that LEQ is true.
    // Used for strictconnect, some reference operations, and anywhere the
    // widths should be inferred strictly in one direction but are required to
    // also be equal for correctness.
    if (equal) {
      LLVM_ATTRIBUTE_UNUSED auto *leq =
          solver.addLeqConstraint(largerVar, smaller);
      LLVM_DEBUG(llvm::dbgs()
                 << "Constrained " << *largerVar << " <= " << *leq << "\n");
    }
    return;
  }

  // If the smaller expr is a free variable but the larger one is not, create a
  // `expr <= k` upper bound that we can verify once all lower bounds have been
  // satisfied. Since we are always picking the smallest width to satisfy all
  // `>=` constraints, any `<=` constraints have no effect on the solution
  // besides indicating that a width is unsatisfiable.
  if (auto *smallerVar = dyn_cast<VarExpr>(smaller)) {
    if (imposeUpperBounds || equal) {
      LLVM_ATTRIBUTE_UNUSED auto *c =
          solver.addLeqConstraint(smallerVar, larger);
      LLVM_DEBUG(llvm::dbgs()
                 << "Constrained " << *smallerVar << " <= " << *c << "\n");
    }
  }
}

/// Assign the constraint expressions of the fields in the `src` argument as the
/// expressions for the `dst` argument. Both fields must be of the given `type`.
void InferenceMapping::unifyTypes(FieldRef lhs, FieldRef rhs, FIRRTLType type) {
  // Fast path for `unifyTypes(x, x, _)`.
  if (lhs == rhs)
    return;

  // Co-iterate the two field refs, recurring into every leaf element and set
  // them equal.
  auto fieldID = 0;
  std::function<void(FIRRTLBaseType)> unify = [&](FIRRTLBaseType type) {
    if (type.isGround()) {
      // Leaf element, unify the fields!
      FieldRef lhsFieldRef(lhs.getValue(), lhs.getFieldID() + fieldID);
      FieldRef rhsFieldRef(rhs.getValue(), rhs.getFieldID() + fieldID);
      LLVM_DEBUG(llvm::dbgs()
                 << "Unify " << getFieldName(lhsFieldRef).first << " = "
                 << getFieldName(rhsFieldRef).first << "\n");
      // Abandon variables becoming unconstrainable by the unification.
      if (auto *var = dyn_cast_or_null<VarExpr>(getExprOrNull(lhsFieldRef)))
        solver.addGeqConstraint(var, solver.known(0));
      setExpr(lhsFieldRef, getExpr(rhsFieldRef));
      fieldID++;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      fieldID++;
      for (auto &element : bundleType) {
        unify(element.type);
      }
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // Skip 0 length vectors.
      if (vecType.getNumElements() > 0) {
        unify(vecType.getElementType());
      }
      fieldID = save + vecType.getMaxFieldID();
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      for (auto &element : enumType.getElements())
        unify(element.type);
    } else {
      llvm_unreachable("Unknown type inside a bundle!");
    }
  };
  if (auto ftype = getBaseType(type))
    unify(ftype);
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(Value value) const {
  assert(type_cast<FIRRTLType>(getBaseType(value.getType())).isGround());
  // A field ID of 0 indicates the entire value.
  return getExpr(FieldRef(value, 0));
}

/// Get the constraint expression for a value.
Expr *InferenceMapping::getExpr(FieldRef fieldRef) const {
  auto expr = getExprOrNull(fieldRef);
  assert(expr && "constraint expr should have been constructed for value");
  return expr;
}

Expr *InferenceMapping::getExprOrNull(FieldRef fieldRef) const {
  auto it = opExprs.find(fieldRef);
  return it != opExprs.end() ? it->second : nullptr;
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(Value value, Expr *expr) {
  assert(type_cast<FIRRTLType>(getBaseType(value.getType())).isGround());
  // A field ID of 0 indicates the entire value.
  setExpr(FieldRef(value, 0), expr);
}

/// Associate a constraint expression with a value.
void InferenceMapping::setExpr(FieldRef fieldRef, Expr *expr) {
  LLVM_DEBUG({
    llvm::dbgs() << "Expr " << *expr << " for " << fieldRef.getValue();
    if (fieldRef.getFieldID())
      llvm::dbgs() << " '" << getFieldName(fieldRef).first << "'";
    auto fieldName = getFieldName(fieldRef);
    if (fieldName.second)
      llvm::dbgs() << " (\"" << fieldName.first << "\")";
    llvm::dbgs() << "\n";
  });
  opExprs[fieldRef] = expr;
}

//===----------------------------------------------------------------------===//
// Inference Result Application
//===----------------------------------------------------------------------===//

namespace {
/// A helper class which maps the types and operations in a design to a set
/// of variables and constraints to be solved later.
class InferenceTypeUpdate {
public:
  InferenceTypeUpdate(InferenceMapping &mapping) : mapping(mapping) {}

  LogicalResult update(CircuitOp op);
  FailureOr<bool> updateOperation(Operation *op);
  FailureOr<bool> updateValue(Value value);
  FIRRTLBaseType updateType(FieldRef fieldRef, FIRRTLBaseType type);

private:
  const InferenceMapping &mapping;
};

} // namespace

/// Update the types throughout a circuit.
LogicalResult InferenceTypeUpdate::update(CircuitOp op) {
  LLVM_DEBUG(llvm::dbgs() << "\n===----- Update types -----===\n\n");
  return mlir::failableParallelForEach(
      op.getContext(), op.getOps<FModuleOp>(), [&](FModuleOp op) {
        // Skip this module if it had no widths to be
        // inferred at all.
        if (mapping.isModuleSkipped(op))
          return success();
        auto isFailed = op.walk<WalkOrder::PreOrder>([&](Operation *op) {
                            if (failed(updateOperation(op)))
                              return WalkResult::interrupt();
                            return WalkResult::advance();
                          }).wasInterrupted();
        return failure(isFailed);
      });
}

/// Update the result types of an operation.
FailureOr<bool> InferenceTypeUpdate::updateOperation(Operation *op) {
  bool anyChanged = false;

  for (Value v : op->getResults()) {
    auto result = updateValue(v);
    if (failed(result))
      return result;
    anyChanged |= *result;
  }

  // If this is a connect operation, width inference might have inferred a RHS
  // that is wider than the LHS, in which case an additional BitsPrimOp is
  // necessary to truncate the value.
  if (auto con = dyn_cast<ConnectOp>(op)) {
    auto lhs = con.getDest();
    auto rhs = con.getSrc();
    auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
    auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());

    // Nothing to do if not base types.
    if (!lhsType || !rhsType)
      return anyChanged;

    auto lhsWidth = lhsType.getBitWidthOrSentinel();
    auto rhsWidth = rhsType.getBitWidthOrSentinel();
    if (lhsWidth >= 0 && rhsWidth >= 0 && lhsWidth < rhsWidth) {
      OpBuilder builder(op);
      auto trunc = builder.createOrFold<TailPrimOp>(con.getLoc(), con.getSrc(),
                                                    rhsWidth - lhsWidth);
      if (type_isa<SIntType>(rhsType))
        trunc =
            builder.createOrFold<AsSIntPrimOp>(con.getLoc(), lhsType, trunc);

      LLVM_DEBUG(llvm::dbgs()
                 << "Truncating RHS to " << lhsType << " in " << con << "\n");
      con->replaceUsesOfWith(con.getSrc(), trunc);
    }
    return anyChanged;
  }

  // If this is a module, update its ports.
  if (auto module = dyn_cast<FModuleOp>(op)) {
    // Update the block argument types.
    bool argsChanged = false;
    SmallVector<Attribute> argTypes;
    argTypes.reserve(module.getNumPorts());
    for (auto arg : module.getArguments()) {
      auto result = updateValue(arg);
      if (failed(result))
        return result;
      argsChanged |= *result;
      argTypes.push_back(TypeAttr::get(arg.getType()));
    }

    // Update the module function type if needed.
    if (argsChanged) {
      module->setAttr(FModuleLike::getPortTypesAttrName(),
                      ArrayAttr::get(module.getContext(), argTypes));
      anyChanged = true;
    }
  }
  return anyChanged;
}

/// Resize a `uint`, `sint`, or `analog` type to a specific width.
static FIRRTLBaseType resizeType(FIRRTLBaseType type, uint32_t newWidth) {
  auto *context = type.getContext();
  return FIRRTLTypeSwitch<FIRRTLBaseType, FIRRTLBaseType>(type)
      .Case<UIntType>([&](auto type) {
        return UIntType::get(context, newWidth, type.isConst());
      })
      .Case<SIntType>([&](auto type) {
        return SIntType::get(context, newWidth, type.isConst());
      })
      .Case<AnalogType>([&](auto type) {
        return AnalogType::get(context, newWidth, type.isConst());
      })
      .Default([&](auto type) { return type; });
}

/// Update the type of a value.
FailureOr<bool> InferenceTypeUpdate::updateValue(Value value) {
  // Check if the value has a type which we can update.
  auto type = type_dyn_cast<FIRRTLType>(value.getType());
  if (!type)
    return false;

  // Fast path for types that have fully inferred widths.
  if (!hasUninferredWidth(type))
    return false;

  // If this is an operation that does not generate any free variables that
  // are determined during width inference, simply update the value type based
  // on the operation arguments.
  if (auto op = dyn_cast_or_null<InferTypeOpInterface>(value.getDefiningOp())) {
    SmallVector<Type, 2> types;
    auto res =
        op.inferReturnTypes(op->getContext(), op->getLoc(), op->getOperands(),
                            op->getAttrDictionary(), op->getPropertiesStorage(),
                            op->getRegions(), types);
    if (failed(res))
      return failure();

    assert(types.size() == op->getNumResults());
    for (auto it : llvm::zip(op->getResults(), types)) {
      LLVM_DEBUG(llvm::dbgs() << "Inferring " << std::get<0>(it) << " as "
                              << std::get<1>(it) << "\n");
      std::get<0>(it).setType(std::get<1>(it));
    }
    return true;
  }

  // Recreate the type, substituting the solved widths.
  auto context = type.getContext();
  unsigned fieldID = 0;
  std::function<FIRRTLBaseType(FIRRTLBaseType)> updateBase =
      [&](FIRRTLBaseType type) -> FIRRTLBaseType {
    auto width = type.getBitWidthOrSentinel();
    if (width >= 0) {
      // Known width integers return themselves.
      fieldID++;
      return type;
    } else if (width == -1) {
      // Unknown width integers return the solved type.
      auto newType = updateType(FieldRef(value, fieldID), type);
      fieldID++;
      return newType;
    } else if (auto bundleType = type_dyn_cast<BundleType>(type)) {
      // Bundle types recursively update all bundle elements.
      fieldID++;
      llvm::SmallVector<BundleType::BundleElement, 3> elements;
      for (auto &element : bundleType) {
        auto updatedBase = updateBase(element.type);
        if (!updateBase)
          return {};
        elements.emplace_back(element.name, element.isFlip, updatedBase);
      }
      return BundleType::get(context, elements, bundleType.isConst());
    } else if (auto vecType = type_dyn_cast<FVectorType>(type)) {
      fieldID++;
      auto save = fieldID;
      // TODO: this should recurse into the element type of 0 length vectors and
      // set any unknown width to 0.
      if (vecType.getNumElements() > 0) {
        auto updatedBase = updateBase(vecType.getElementType());
        if (!updatedBase)
          return {};
        auto newType = FVectorType::get(updatedBase, vecType.getNumElements(),
                                        vecType.isConst());
        fieldID = save + vecType.getMaxFieldID();
        return newType;
      }
      // If this is a 0 length vector return the original type.
      return type;
    } else if (auto enumType = type_dyn_cast<FEnumType>(type)) {
      fieldID++;
      llvm::SmallVector<FEnumType::EnumElement> elements;
      for (auto &element : enumType.getElements()) {
        auto updatedBase = updateBase(element.type);
        if (!updateBase)
          return {};
        elements.emplace_back(element.name, updatedBase);
      }
      return FEnumType::get(context, elements, enumType.isConst());
    }
    llvm_unreachable("Unknown type inside a bundle!");
  };

  // Update the type.
  auto newType = mapBaseTypeNullable(type, updateBase);
  if (!newType)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Update " << value << " to " << newType << "\n");
  value.setType(newType);

  // If this is a ConstantOp, adjust the width of the underlying APInt.
  // Unsized constants have APInts which are *at least* wide enough to hold
  // the value, but may be larger. This can trip up the verifier.
  if (auto op = value.getDefiningOp<ConstantOp>()) {
    auto k = op.getValue();
    auto bitwidth = op.getType().getBitWidthOrSentinel();
    if (k.getBitWidth() > unsigned(bitwidth))
      k = k.trunc(bitwidth);
    op->setAttr("value", IntegerAttr::get(op.getContext(), k));
  }

  return newType != type;
}

/// Update a type.
FIRRTLBaseType InferenceTypeUpdate::updateType(FieldRef fieldRef,
                                               FIRRTLBaseType type) {
  assert(type.isGround() && "Can only pass in ground types.");
  auto value = fieldRef.getValue();
  // Get the inferred width.
  Expr *expr = mapping.getExprOrNull(fieldRef);
  if (!expr || !expr->solution) {
    // It should not be possible to arrive at an uninferred width at this point.
    // In case the constraints are not resolvable, checks before the calls to
    // `updateType` must have already caught the issues and aborted the pass
    // early. Might turn this into an assert later.
    mlir::emitError(value.getLoc(), "width should have been inferred");
    return {};
  }
  int32_t solution = *expr->solution;
  assert(solution >= 0); // The solver infers variables to be 0 or greater.
  return resizeType(type, solution);
}

//===----------------------------------------------------------------------===//
// Hashing
//===----------------------------------------------------------------------===//

// Hash slots in the interned allocator as if they were the pointed-to value
// itself.
namespace llvm {
template <typename T>
struct DenseMapInfo<InternedSlot<T>> {
  using Slot = InternedSlot<T>;
  static Slot getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return Slot(static_cast<T *>(pointer));
  }
  static Slot getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return Slot(static_cast<T *>(pointer));
  }
  static unsigned getHashValue(Slot val) { return mlir::hash_value(*val.ptr); }
  static bool isEqual(Slot LHS, Slot RHS) {
    auto empty = getEmptyKey().ptr;
    auto tombstone = getTombstoneKey().ptr;
    if (LHS.ptr == empty || RHS.ptr == empty || LHS.ptr == tombstone ||
        RHS.ptr == tombstone)
      return LHS.ptr == RHS.ptr;
    return *LHS.ptr == *RHS.ptr;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
class InferWidthsPass : public InferWidthsBase<InferWidthsPass> {
  void runOnOperation() override;
};
} // namespace

void InferWidthsPass::runOnOperation() {
  // Collect variables and constraints
  ConstraintSolver solver;
  InferenceMapping mapping(solver, getAnalysis<SymbolTable>());
  if (failed(mapping.map(getOperation()))) {
    signalPassFailure();
    return;
  }
  if (mapping.areAllModulesSkipped()) {
    markAllAnalysesPreserved();
    return; // fast path if no inferrable widths are around
  }

  // Solve the constraints.
  if (failed(solver.solve())) {
    signalPassFailure();
    return;
  }

  // Update the types with the inferred widths.
  if (failed(InferenceTypeUpdate(mapping).update(getOperation())))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInferWidthsPass() {
  return std::make_unique<InferWidthsPass>();
}
