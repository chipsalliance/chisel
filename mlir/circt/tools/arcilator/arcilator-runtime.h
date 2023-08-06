// NOLINTBEGIN
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ostream>
#include <vector>

struct Signal {
  const char *name;
  unsigned offset;
  unsigned numBits;
  enum Type { Input, Output, Register, Memory, Wire } type;
  // for memories:
  unsigned stride;
  unsigned depth;
};

struct Hierarchy {
  const char *name;
  unsigned numStates;
  unsigned numChildren;
  Signal *states;
  Hierarchy *children;
};

template <unsigned N>
struct Bytes {
  uint8_t byte[N];
};
template <typename T, unsigned Stride, unsigned Depth>
struct Memory {
  union {
    T data;
    uint8_t stride[Stride];
  } words[Depth];
};

template <class ModelLayout>
class ValueChangeDump {
public:
  ValueChangeDump(std::basic_ostream<char> &os, const uint8_t *state)
      : os(os), state(state) {}

  void writeHeader(bool withHierarchy = true) {
    os << "$date\n    October 21, 2015\n$end\n";
    os << "$version\n    Some cryptic MLIR magic\n$end\n";
    os << "$timescale 1ns $end\n";

    os << "$scope module " << ModelLayout::name << " $end\n";

    auto writeSignal = [&](const Signal &state) {
      if (state.type != Signal::Memory) {
        auto &signal =
            allocSignal(state, state.offset, (state.numBits + 7) / 8);
        if (state.type == Signal::Register) {
          os << "$var reg " << state.numBits << " " << signal.abbrev << " "
             << state.name;
        } else {
          os << "$var wire " << state.numBits << " " << signal.abbrev << " "
             << state.name;
        }
        if (state.numBits > 1)
          os << " [" << (state.numBits - 1) << ":0]";
        os << " $end\n";
      } else {
        for (unsigned i = 0; i < state.depth; ++i) {
          auto &signal = allocSignal(state, state.offset + i * state.stride,
                                     (state.numBits + 7) / 8);
          os << "$var reg " << state.numBits << " " << signal.abbrev << " "
             << state.name << "[" << i << "]";
          if (state.numBits > 1)
            os << " [" << (state.numBits - 1) << ":0]";
          os << " $end\n";
        }
      }
    };

    std::function<void(const Hierarchy &)> writeHierarchy =
        [&](const Hierarchy &hierarchy) {
          os << "$scope module " << hierarchy.name << " $end\n";
          for (unsigned i = 0; i < hierarchy.numStates; ++i)
            writeSignal(hierarchy.states[i]);
          for (unsigned i = 0; i < hierarchy.numChildren; ++i)
            writeHierarchy(hierarchy.children[i]);
          os << "$upscope $end\n";
        };

    for (auto &port : ModelLayout::io)
      writeSignal(port);
    if (withHierarchy)
      writeHierarchy(ModelLayout::hierarchy);

    os << "$upscope $end\n";
    os << "$enddefinitions $end\n";
  }

  void writeValues(bool includeUnchanged = false) {
    for (auto &signal : signals) {
      const uint8_t *valNew = state + signal.offset;
      uint8_t *valOld = &previousValues[0] + signal.previousOffset;
      size_t numBytes = (signal.state.numBits + 7) / 8;
      bool unchanged = std::equal(valNew, valNew + numBytes, valOld);
      if (unchanged && !includeUnchanged)
        continue;
      if (signal.state.numBits > 1)
        os << 'b';
      for (unsigned n = signal.state.numBits; n > 0; --n)
        os << (valNew[(n - 1) / 8] & (1 << ((n - 1) % 8)) ? '1' : '0');
      if (signal.state.numBits > 1)
        os << ' ';
      os << signal.abbrev << "\n";
      std::copy(valNew, valNew + numBytes, valOld);
    }
  }

  void writeDumpvars() {
    os << "$dumpvars\n";
    writeValues(true);
  }

  void writeTimestep(size_t timeIncrement) {
    time += timeIncrement;
    os << "#" << time << "\n";
    writeValues();
  }

  size_t time = 0;

private:
  struct VcdSignal {
    std::string abbrev;
    unsigned offset;
    const Signal &state;
    unsigned previousOffset;
  };

  VcdSignal &allocSignal(const Signal &state, unsigned offset,
                         unsigned numBytes) {
    std::string abbrev;
    unsigned rest = signals.size() + 1;
    while (rest != 0) {
      uint8_t c = (rest % 84) + 33;
      if (c >= '0')
        c += 10;
      abbrev += c;
      rest /= 84;
    }
    signals.push_back(
        VcdSignal{abbrev, offset, state, unsigned(previousValues.size())});
    previousValues.resize(previousValues.size() + numBytes);
    return signals.back();
  }

  std::basic_ostream<char> &os;
  const uint8_t *state;
  std::vector<VcdSignal> signals;
  std::vector<uint8_t> previousValues;
};

// NOLINTEND
