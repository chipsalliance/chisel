//===- Endpoint.h - Cosim endpoint server -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declare the class which is used to model DPI endpoints.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_COSIM_ENDPOINT_H
#define CIRCT_DIALECT_ESI_COSIM_ENDPOINT_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace circt {
namespace esi {
namespace cosim {

/// Implements a bi-directional, thread-safe bridge between the RPC server and
/// DPI functions.
///
/// Several of the methods below are inline with the declaration to make them
/// candidates for inlining during compilation. This is particularly important
/// on the simulation side since polling happens at each clock and we do not
/// want to slow down the simulation any more than necessary.
class Endpoint {
public:
  /// Representing messages as shared pointers to vectors may be a performance
  /// issue in the future but it is the easiest way to ensure memory
  /// correctness.
  using Blob = std::vector<uint8_t>;
  using BlobPtr = std::shared_ptr<Blob>;

  /// Construct an endpoint which knows and the type IDs in both directions.
  Endpoint(uint64_t sendTypeId, int sendTypeMaxSize, uint64_t recvTypeId,
           int recvTypeMaxSize);
  ~Endpoint();
  /// Disallow copying. There is only ONE endpoint object per logical endpoint
  /// so copying is almost always a bug.
  Endpoint(const Endpoint &) = delete;

  uint64_t getSendTypeId() const { return sendTypeId; }
  uint64_t getRecvTypeId() const { return recvTypeId; }

  /// These two are used to set and unset the inUse flag, to ensure that an open
  /// endpoint is not opened again.
  bool setInUse();
  void returnForUse();

  /// Queue message to the simulation.
  void pushMessageToSim(BlobPtr msg) {
    Lock g(m);
    toCosim.push(msg);
  }

  /// Pop from the to-simulator queue. Return true if there was a message in the
  /// queue.
  bool getMessageToSim(BlobPtr &msg) {
    Lock g(m);
    if (toCosim.empty())
      return false;
    msg = toCosim.front();
    toCosim.pop();
    return true;
  }

  /// Queue message to the RPC client.
  void pushMessageToClient(BlobPtr msg) {
    Lock g(m);
    toClient.push(msg);
  }

  /// Pop from the to-RPC-client queue. Return true if there was a message in
  /// the queue.
  bool getMessageToClient(BlobPtr &msg) {
    Lock g(m);
    if (toClient.empty())
      return false;
    msg = toClient.front();
    toClient.pop();
    return true;
  }

private:
  const uint64_t sendTypeId;
  const uint64_t recvTypeId;
  bool inUse;

  using Lock = std::lock_guard<std::mutex>;

  /// This class needs to be thread-safe. All of the mutable member variables
  /// are protected with this object-wide lock. This may be a performance issue
  /// in the future.
  std::mutex m;
  /// Message queue from RPC client to the simulation.
  std::queue<BlobPtr> toCosim;
  /// Message queue to RPC client from the simulation.
  std::queue<BlobPtr> toClient;
};

/// The Endpoint registry is where Endpoints report their existence (register)
/// and they are looked up by RPC clients.
class EndpointRegistry {
public:
  /// Register an Endpoint. Creates the Endpoint object and owns it. Returns
  /// false if unsuccessful.
  bool registerEndpoint(std::string epId, uint64_t sendTypeId,
                        int sendTypeMaxSize, uint64_t recvTypeId,
                        int recvTypeMaxSize);

  /// Get the specified endpoint. Return nullptr if it does not exist. This
  /// method is defined inline so it can be inlined at compile time. Performance
  /// is important here since this method is used in the polling call from the
  /// simulator. Returns nullptr if the endpoint cannot be found.
  Endpoint *operator[](const std::string &epId) {
    Lock g(m);
    auto it = endpoints.find(epId);
    if (it == endpoints.end())
      return nullptr;
    return &it->second;
  }

  /// Iterate over the list of endpoints, calling the provided function for each
  /// endpoint.
  void iterateEndpoints(
      const std::function<void(std::string id, const Endpoint &)> &f) const;
  /// Return the number of endpoints.
  size_t size() const;

private:
  using Lock = std::lock_guard<std::mutex>;

  /// This object needs to be thread-safe. An object-wide mutex is sufficient.
  std::mutex m;

  /// Endpoint ID to object pointer mapping.
  std::map<std::string, Endpoint> endpoints;
};

} // namespace cosim
} // namespace esi
} // namespace circt

#endif
