#include <stdint.h>
#include <iostream>

extern "C" void hello()
{
    std::cout << "hello from c++\n";
}

extern "C" void add(uint32_t* lhs, uint32_t* rhs, uint32_t *result)
{
    *result = *lhs + *rhs;
}

struct CounterState
{
    int counter = 0;
};

extern "C" void create_counter(CounterState** result)
{
    *result = new CounterState;
}

 extern "C" void increment_counter(CounterState** state, uint32_t *result)
{
    *result = (*state)->counter++;
}

