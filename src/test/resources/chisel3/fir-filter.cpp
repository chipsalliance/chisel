#include <deque>
#include <stdint.h>
#include <vector>

struct FIRFilter {
  std::vector<uint64_t> coeff;
  std::deque<uint64_t> inputs;

  uint64_t tick(uint64_t input) {
    inputs.push_front(input);
    if (coeff.size() < inputs.size())
      inputs.pop_back();
    uint64_t sum = 0;
    for (int i = 0, e = inputs.size(); i < e; i++)
      sum += coeff[i] * inputs[i];
    return sum;
  }

  FIRFilter(uint64_t *ptr, uint64_t len) {
    coeff.reserve(len);
    for (int i = 0; i < len; i++)
      coeff.push_back(ptr[i]);
  }

  void clear() { inputs.clear(); }
};

extern "C" {
void fir_filter_new(uint64_t* ptr, uint64_t len, FIRFilter **result) {
  *result = new FIRFilter(ptr, len);
}

void fir_filter_reset(FIRFilter *filter) { filter->clear(); }

void fir_filter_tick(FIRFilter *filter, uint64_t input, uint64_t *output) {
  *output = filter->tick(input);
}
}