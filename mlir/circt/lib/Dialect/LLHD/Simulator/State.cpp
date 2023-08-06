//===- State.cpp - LLHD simulator state -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the constructs used to keep track of the simulation
// state in the LLHD simulator.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/Simulator/State.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace llvm;
using namespace circt::llhd::sim;

//===----------------------------------------------------------------------===//
// Time
//===----------------------------------------------------------------------===//

std::string Time::toString() const {
  return std::to_string(time) + "ps " + std::to_string(delta) + "d " +
         std::to_string(eps) + "e";
}

//===----------------------------------------------------------------------===//
// Signal
//===----------------------------------------------------------------------===//

std::string Signal::toHexString() const {
  std::string ret;
  raw_string_ostream ss(ret);
  ss << "0x";
  for (int i = size - 1; i >= 0; --i) {
    ss << format_hex_no_prefix(static_cast<int>(value[i]), 2);
  }
  return ret;
}

std::string Signal::toHexString(unsigned elemIndex) const {
  assert(elements.size() > 0 && "the signal type has to be tuple or array!");
  auto elemSize = elements[elemIndex].second;
  auto *ptr = value + elements[elemIndex].first;
  std::string ret;
  raw_string_ostream ss(ret);
  ss << "0x";
  for (int i = elemSize - 1; i >= 0; --i) {
    ss << format_hex_no_prefix(static_cast<int>(ptr[i]), 2);
  }
  return ret;
}

Signal::~Signal() {
  std::free(value);
  value = nullptr;
}

//===----------------------------------------------------------------------===//
// Slot
//===----------------------------------------------------------------------===//

bool Slot::operator<(const Slot &rhs) const { return time < rhs.time; }

bool Slot::operator>(const Slot &rhs) const { return rhs.time < time; }

void Slot::insertChange(int index, int bitOffset, uint8_t *bytes,
                        unsigned width) {
  // Get the amount of 64 bit words required to store the value in an APInt.
  auto size = llvm::divideCeil(width, 8);

  APInt buffer(width, 0);
  llvm::LoadIntFromMemory(buffer, bytes, size);
  auto offsetBufferPair = std::make_pair(bitOffset, buffer);

  if (changesSize >= buffers.size()) {
    // Create a new change buffer if we don't have any unused one available for
    // reuse.
    buffers.push_back(offsetBufferPair);
  } else {
    // Reuse the first available buffer.
    buffers[changesSize] = offsetBufferPair;
  }

  // Map the signal index to the change buffer so we can retrieve
  // it after sorting.
  changes.push_back(std::make_pair(index, changesSize));
  ++changesSize;
}

void Slot::insertChange(unsigned inst) { scheduled.push_back(inst); }

//===----------------------------------------------------------------------===//
// UpdateQueue
//===----------------------------------------------------------------------===//
void UpdateQueue::insertOrUpdate(Time time, int index, int bitOffset,
                                 uint8_t *bytes, unsigned width) {
  auto &slot = getOrCreateSlot(time);
  slot.insertChange(index, bitOffset, bytes, width);
}

void UpdateQueue::insertOrUpdate(Time time, unsigned inst) {
  auto &slot = getOrCreateSlot(time);
  slot.insertChange(inst);
}

Slot &UpdateQueue::getOrCreateSlot(Time time) {
  auto &top = begin()[topSlot];

  // Directly add to top slot.
  if (!top.unused && time == top.time) {
    return top;
  }

  // We need to search through the queue for an existing slot only if we're
  // spawning an event later than the top slot. Adding to an existing slot
  // scheduled earlier than the top slot should never happens, as then it should
  // be the top.
  if (events > 0 && top.time < time) {
    for (size_t i = 0, e = size(); i < e; ++i) {
      if (time == begin()[i].time) {
        return begin()[i];
      }
    }
  }

  // Spawn new event using an existing slot.
  if (!unused.empty()) {
    auto firstUnused = unused.pop_back_val();
    auto &newSlot = begin()[firstUnused];
    newSlot.unused = false;
    newSlot.time = time;

    // Update the top of the queue either if it is currently unused or the new
    // timestamp is earlier than it.
    if (top.unused || time < top.time)
      topSlot = firstUnused;

    ++events;
    return newSlot;
  }

  // We do not have pre-allocated slots available, generate a new one.
  push_back(Slot(time));

  // Update the top of the queue either if it is currently unused or the new
  // timestamp is earlier than it.
  if (top.unused || time < top.time)
    topSlot = size() - 1;

  ++events;
  return back();
}

const Slot &UpdateQueue::top() {
  assert(topSlot < size() && "top is pointing out of bounds!");

  // Sort the changes of the top slot such that all changes to the same signal
  // are in succession.
  auto &top = begin()[topSlot];
  llvm::sort(top.changes.begin(), top.changes.begin() + top.changesSize);
  return top;
}

void UpdateQueue::pop() {
  // Reset internal structures and decrease the event counter.
  auto &curr = begin()[topSlot];
  curr.unused = true;
  curr.changesSize = 0;
  curr.scheduled.clear();
  curr.changes.clear();
  curr.time = Time();
  --events;

  // Add to unused slots list for easy retrieval.
  unused.push_back(topSlot);

  // Update the current top of the queue.
  topSlot = std::distance(
      begin(),
      std::min_element(begin(), end(), [](const auto &a, const auto &b) {
        // a is "smaller" than b if either a's timestamp is earlier than b's, or
        // b is unused (i.e. b has no actual meaning).
        return !a.unused && (a < b || b.unused);
      }));
}

//===----------------------------------------------------------------------===//
// Instance
//===----------------------------------------------------------------------===//

Instance::~Instance() {
  std::free(procState);
  procState = nullptr;
  std::free(entityState);
  entityState = nullptr;
}

//===----------------------------------------------------------------------===//
// State
//===----------------------------------------------------------------------===//

State::~State() {
  for (auto &inst : instances) {
    if (inst.procState) {
      std::free(inst.procState->senses);
    }
  }
}

Slot State::popQueue() {
  assert(!queue.empty() && "the event queue is empty");
  Slot pop = queue.top();
  queue.pop();
  return pop;
}

void State::pushQueue(Time t, unsigned inst) {
  Time newTime = time + t;
  queue.insertOrUpdate(newTime, inst);
  instances[inst].expectedWakeup = newTime;
}

llvm::SmallVectorTemplateCommon<Instance>::iterator
State::getInstanceIterator(std::string instName) {
  auto it =
      std::find_if(instances.begin(), instances.end(),
                   [&](const auto &inst) { return instName == inst.name; });

  assert(it != instances.end() && "instance does not exist!");

  return it;
}

int State::addSignal(std::string name, std::string owner) {
  signals.push_back(Signal(name, owner));
  return signals.size() - 1;
}

void State::addProcPtr(std::string name, ProcState *procStatePtr) {
  auto it = getInstanceIterator(name);

  // Store instance index in process state.
  procStatePtr->inst = it - instances.begin();
  (*it).procState = procStatePtr;
}

int State::addSignalData(int index, std::string owner, uint8_t *value,
                         uint64_t size) {
  auto it = getInstanceIterator(owner);

  uint64_t globalIdx = (*it).sensitivityList[index + (*it).nArgs].globalIndex;
  auto &sig = signals[globalIdx];

  // Add pointer and size to global signal table entry.
  sig.store(value, size);

  // Add the value pointer to the signal detail struct for each instance this
  // signal appears in.
  for (auto inst : signals[globalIdx].getTriggeredInstanceIndices()) {
    for (auto &detail : instances[inst].sensitivityList) {
      if (detail.globalIndex == globalIdx) {
        detail.value = sig.getValue();
      }
    }
  }
  return globalIdx;
}

void State::addSignalElement(unsigned index, unsigned offset, unsigned size) {
  signals[index].pushElement(std::make_pair(offset, size));
}

void State::dumpSignal(llvm::raw_ostream &out, int index) {
  auto &sig = signals[index];
  for (auto inst : sig.getTriggeredInstanceIndices()) {
    out << time.toString() << "  " << instances[inst].path << "/"
        << sig.getName() << "  " << sig.toHexString() << "\n";
  }
}

void State::dumpLayout() {
  llvm::errs() << "::------------------- Layout -------------------::\n";
  for (const auto &inst : instances) {
    llvm::errs() << inst.name << ":\n";
    llvm::errs() << "---path: " << inst.path << "\n";
    llvm::errs() << "---isEntity: " << inst.isEntity << "\n";
    llvm::errs() << "---sensitivity list: ";
    for (auto in : inst.sensitivityList) {
      llvm::errs() << in.globalIndex << " ";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "::----------------------------------------------::\n";
}

void State::dumpSignalTriggers() {
  llvm::errs() << "::------------- Signal information -------------::\n";
  for (size_t i = 0, e = signals.size(); i < e; ++i) {
    llvm::errs() << signals[i].getOwner() << "/" << signals[i].getName()
                 << " triggers: ";
    for (auto trig : signals[i].getTriggeredInstanceIndices()) {
      llvm::errs() << trig << " ";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "::----------------------------------------------::\n";
}
