//===- signals-runtime-wrappers.cpp - Runtime library implementation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the runtime library used in LLHD simulation.
//
//===----------------------------------------------------------------------===//

#include "signals-runtime-wrappers.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace circt::llhd::sim;

//===----------------------------------------------------------------------===//
// Runtime interface
//===----------------------------------------------------------------------===//

int allocSignal(State *state, int index, char *owner, uint8_t *value,
                int64_t size) {
  assert(state && "alloc_signal: state not found");
  std::string sOwner(owner);

  return state->addSignalData(index, sOwner, value, size);
}

void addSigArrayElements(State *state, unsigned index, unsigned size,
                         unsigned numElements) {
  for (size_t i = 0; i < numElements; ++i)
    state->addSignalElement(index, size * i, size);
}

void addSigStructElement(State *state, unsigned index, unsigned offset,
                         unsigned size) {
  state->addSignalElement(index, offset, size);
}

void allocProc(State *state, char *owner, ProcState *procState) {
  assert(state && "alloc_proc: state not found");
  std::string sOwner(owner);
  state->addProcPtr(sOwner, procState);
}

void allocEntity(State *state, char *owner, uint8_t *entityState) {
  assert(state && "alloc_entity: state not found");
  auto it = state->getInstanceIterator(owner);
  (*it).entityState = entityState;
}

void driveSignal(State *state, SignalDetail *detail, uint8_t *value,
                 uint64_t width, int time, int delta, int eps) {
  assert(state && "drive_signal: state not found");

  auto globalIndex = detail->globalIndex;
  auto offset = detail->offset;

  int bitOffset =
      (detail->value - state->signals[globalIndex].getValue()) * 8 + offset;

  // Spawn a new event.
  state->queue.insertOrUpdate(state->time + Time(time, delta, eps), globalIndex,
                              bitOffset, value, width);
}

void llhdSuspend(State *state, ProcState *procState, int time, int delta,
                 int eps) {
  // Add a new scheduled wake up if a time is specified.
  if (time || delta || eps) {
    Time sTime(time, delta, eps);
    state->pushQueue(sTime, procState->inst);
  }
}
