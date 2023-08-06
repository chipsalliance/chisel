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

  auto value = ESITypes::I8(42);
  (*top.hostcomms->Recv)(value);
  auto res = (*top.hostcomms->Send)();
  if (res != value) {
    std::cerr << "Test failed: expected " << value << ", got " << res
              << std::endl;
    return 1;
  }

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
        << "usage: loopback_test {rpc hostname}:{rpc port} {path to schema}"
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
