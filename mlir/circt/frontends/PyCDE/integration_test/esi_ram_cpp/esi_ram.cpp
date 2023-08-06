// REQUIRES: esi-cosim
// XFAIL: *

// clang-format off

// Create ESI system
// RUN: rm -rf %t
// RUN: %PYTHON% %S/../esi_ram.py %t 2>&1

// Build the project using the CMakeLists.txt from this directory. Just move
// everything to the output folder in the build directory; this is very convenient
// if we want to run the build manually afterwards.
// RUN: cp %s %t
// RUN: cp %S/CMakeLists.txt %t
// RUN: cmake -S %t \
// RUN:   -B %t/build \
// RUN:   -DCIRCT_DIR=%CIRCT_SOURCE% \
// RUN:   -DPYCDE_OUT_DIR=%t
// RUN: cmake --build %t/build

// Run test
// ... can't glob *.sv because PyCDE always includes driver.sv, but that's not the
// top that we want to use. Just delete it.
// RUN: rm %t/hw/driver.sv
// RUN: esi-cosim-runner.py --tmpdir=%t  \
// RUN:     --no-aux-files               \
// RUN:     --schema %t/hw/schema.capnp  \
// RUN:     --exec %t/build/esi_ram_test \
// RUN:     %t/hw/*.sv

// To run this test manually:
// 1. run `ninja check-pycde-integration` (this will create the output folder, run PyCDE, ...)
// 2. navigate to %t
// 3. In a separate terminal, run esi-cosim-runner.py in server only mode:
//   - cd %t
//   - esi-cosim-runner.py --tmpdir=$(pwd) --schema=$(pwd)/hw/schema.capnp --server-only $(pwd)/hw/top.sv $(pwd)
// 4. In another terminal, run the test executable. When running esi-cosim-runner, it'll print the $port which
//    the test executable should connect to.
//   - cd %t/build
//   - ./esi_ram_test localhost:$port ../hw/schema.capn

// clang-format on
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "esi/backends/capnp.h"

#include ESI_COSIM_CAPNP_H

#include "ESISystem.h"

using namespace esi;
using namespace runtime;

template <typename T>
int logTestFailure(T expected, T actual, int testID) {
  std::cerr << "Test " << testID << " failed: expected " << expected << ", got "
            << actual << std::endl;
  return testID;
}

template <typename TBackend>
int runTest(TBackend &backend) {
  // Connect the ESI system to the provided backend.
  esi::runtime::top top(backend);

  auto write_cmd =
      ESITypes::Struct16871797234873963366{.address = 2, .data = 42};

  auto loopback_result = (*top.bsp->loopback)(write_cmd);
  if (loopback_result != write_cmd)
    return logTestFailure(write_cmd, loopback_result, 1);

  auto read_result = (*top.bsp->read)(2);
  if (read_result != ESITypes::I64(0))
    return logTestFailure(ESITypes::I64(0), read_result, 2);

  read_result = (*top.bsp->read)(3);
  if (read_result != ESITypes::I64(0))
    return logTestFailure(ESITypes::I64(0), read_result, 3);

  (*top.bsp->write)(write_cmd);
  read_result = (*top.bsp->read)(2);
  if (read_result != ESITypes::I64(42))
    return logTestFailure(ESITypes::I64(42), read_result, 4);

  read_result = (*top.bsp->read)(3);
  if (read_result != ESITypes::I64(42))
    return logTestFailure(ESITypes::I64(42), read_result, 5);

  // Re-write a 0 to the memory (mostly for debugging purposes to allow us to
  // keep the server alive and rerun the test).
  write_cmd = ESITypes::Struct16871797234873963366{.address = 2, .data = 0};
  (*top.bsp->write)(write_cmd);
  read_result = (*top.bsp->read)(2);
  if (read_result != ESITypes::I64(0))
    return logTestFailure(ESITypes::I64(0), read_result, 6);

  return 0;
}

int run_cosim_test(const std::string &host, unsigned port) {
  // Run test with cosimulation backend.
  esi::runtime::cosim::CapnpBackend cosim(host, port);
  return runTest(cosim);
}

int main(int argc, char **argv) {
  std::string rpchostport;
  if (argc != 3) {
    // Schema not currently used but required by the ESI cosim tester
    std::cerr
        << "usage: esi_ram_test {rpc hostname}:{rpc port} {path to schema}"
        << std::endl;
    return 1;
  }

  rpchostport = argv[1];

  // Parse the RPC host and port from the command line.
  auto colon = rpchostport.find(':');
  if (colon == std::string::npos) {
    std::cerr << "Invalid RPC host:port string: " << rpchostport << std::endl;
    return 1;
  }
  auto host = rpchostport.substr(0, colon);
  auto port = stoi(rpchostport.substr(colon + 1));

  auto res = run_cosim_test(host, port);
  if (res != 0) {
    std::cerr << "Test failed with error code " << res << std::endl;
    return 1;
  }
  std::cout << "Test passed" << std::endl;
  return 0;
}
