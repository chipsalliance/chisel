#!/usr/bin/python3

import binascii
import random
import esi_cosim


class LoopbackTester(esi_cosim.CosimBase):
  """Provides methods to test the loopback simulations."""

  def test_list(self):
    ifaces = self.cosim.list().wait().ifaces
    assert len(ifaces) > 0

  def test_open_close(self):
    ifaces = self.cosim.list().wait().ifaces
    openResp = self.cosim.open(ifaces[0]).wait()
    assert openResp.iface is not None
    ep = openResp.iface
    ep.close().wait()

  def test_two_chan_loopback(self, num_msgs):
    to_hw = self.openEP("top.TwoChanLoopback_loopback_tohw",
                        sendType=self.schema.I1,
                        recvType=self.schema.I8)
    from_hw = self.openEP("top.TwoChanLoopback_loopback_fromhw",
                          sendType=self.schema.I8,
                          recvType=self.schema.I1)
    for _ in range(num_msgs):
      data = random.randint(0, 2**8 - 1)
      print(f"Sending {data}")
      to_hw.send(self.schema.I8.new_message(i=data))
      result = self.readMsg(from_hw, self.schema.I8)
      print(f"Got {result}")
      assert (result.i == data)

  def test_i32(self, num_msgs):
    ep = self.openEP("top.intLoopbackInst.IntTestEP.loopback",
                     sendType=self.schema.I32,
                     recvType=self.schema.I32)
    for _ in range(num_msgs):
      data = random.randint(0, 2**32 - 1)
      print(f"Sending {data}")
      ep.send(self.schema.I32.new_message(i=data))
      result = self.readMsg(ep, self.schema.I32)
      print(f"Got {result}")
      assert (result.i == data)

  def write_3bytes(self, ep):
    r = random.randrange(0, 2**24 - 1)
    data = r.to_bytes(3, 'big')
    print(f'Sending: {binascii.hexlify(data)}')
    ep.send(self.schema.UntypedData.new_message(data=data)).wait()
    return data

  def read_3bytes(self, ep):
    dataMsg = self.readMsg(ep, self.schema.UntypedData)
    data = dataMsg.data
    print(binascii.hexlify(data))
    return data

  def test_3bytes(self, num_msgs=50):
    ep = self.openEP("top.ep")
    print("Testing writes")
    dataSent = list()
    for _ in range(num_msgs):
      dataSent.append(self.write_3bytes(ep))
    print()
    print("Testing reads")
    dataRecv = list()
    for _ in range(num_msgs):
      dataRecv.append(self.read_3bytes(ep))
    ep.close().wait()
    assert dataSent == dataRecv

  def test_keytext(self, num_msgs=50):
    cStructType = self.schema.Struct17798359158705484171
    ep = self.openEP("top.twoListLoopbackInst.KeyTextEP",
                     sendType=cStructType,
                     recvType=cStructType)
    kts = []
    for i in range(num_msgs):
      kt = cStructType.new_message(
          key=[random.randrange(0, 255) for x in range(4)],
          text=[random.randrange(0, 16000) for x in range(6)])
      kts.append(kt)
      ep.send(kt).wait()

    for i in range(num_msgs):
      kt = self.readMsg(ep, cStructType)
      print(f"expected: {kts[i]}")
      print(f"got:      {kt}")
      assert list(kt.key) == list(kts[i].key)
      assert list(kt.text) == list(kts[i].text)


if __name__ == "__main__":
  import os
  import sys
  rpc = LoopbackTester(sys.argv[2], sys.argv[1])
  print(rpc.list())
  rpc.test_two_chan_loopback(25)
  rpc.test_i32(25)
  rpc.test_keytext(25)
