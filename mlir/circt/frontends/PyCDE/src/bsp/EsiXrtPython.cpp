#include <cstring>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

// pybind11 includes
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_xclbin.h"

// We don't want to clutter up the symbol space any more than necessary, so use
// an anonymous namespace.
namespace {

uint32_t MagicNumOffset = 16;
uint32_t MagicNumberLo = 0xE5100E51;
uint32_t MagicNumberHi = 0x207D98E5;
uint32_t ExpectedVersionNumber = 0;

class Accelerator {
  xrt::device m_device;
  xrt::ip m_ip;

public:
  Accelerator(const std::string &xclbin_path, const std::string kernel_name) {
    m_device = xrt::device(0);
    auto uuid = m_device.load_xclbin(xclbin_path);
    m_ip = xrt::ip(m_device, uuid, kernel_name);

    // Check that this is actually an ESI system.
    uint32_t magicLo = m_ip.read_register(MagicNumOffset);
    uint32_t magicHi = m_ip.read_register(MagicNumOffset + 4);
    if (magicLo != MagicNumberLo || magicHi != MagicNumberHi)
      throw std::runtime_error("Accelerator is not an ESI system");

    // Check version is one we understand.
    if (version() != ExpectedVersionNumber)
      std::cerr
          << "[ESI] Warning: accelerator ESI version may not be compatible\n";
  }

  uint32_t version() { return m_ip.read_register(MagicNumOffset + 8); }
};

} // namespace

PYBIND11_MODULE(esiXrtPython, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init<const std::string &, const std::string &>())
      .def("version", &Accelerator::version);
}
