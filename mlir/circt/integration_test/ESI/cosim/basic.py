#!/usr/bin/python3

import random
import esi_cosim


class BasicSystemTester(esi_cosim.CosimBase):
  """Provides methods to test the 'basic' simulation."""

  def testIntAcc(self, num_msgs):
    ep = self.openEP("top.ints.TestEP",
                     sendType=self.schema.I32,
                     recvType=self.schema.I32)
    sum = 0
    for _ in range(num_msgs):
      i = random.randint(0, 77)
      sum += i
      print(f"Sending {i}")
      ep.send(self.schema.I32.new_message(i=i))
      result = self.readMsg(ep, self.schema.I32)
      print(f"Got {result}")
      assert (result.i == sum)

  def testVectorSum(self, num_msgs):
    ep = self.openEP("top.array.TestEP",
                     sendType=self.schema.ArrayOf2xUi24,
                     recvType=self.schema.ArrayOf4xSi13)
    for _ in range(num_msgs):
      # Since the result is unsigned, we need to make sure the sum is
      # never negative.
      arr = [
          random.randint(-468, 777),
          random.randint(500, 1250),
          random.randint(-468, 777),
          random.randint(500, 1250)
      ]
      print(f"Sending {arr}")
      ep.send(self.schema.ArrayOf4xSi13.new_message(l=arr))
      result = self.readMsg(ep, self.schema.ArrayOf2xUi24)
      print(f"Got {result}")
      assert (result.l[0] == arr[0] + arr[1])
      assert (result.l[1] == arr[2] + arr[3])

  def testCrypto(self, num_msgs):
    ep = self.openEP("top.structs.CryptoData",
                     sendType=self.schema.Struct15822124641382404136,
                     recvType=self.schema.Struct15822124641382404136)
    cfg = self.openEP("top.structs.CryptoConfig",
                      sendType=self.schema.I1,
                      recvType=self.schema.Struct14745270011869700302)

    cfgWritten = False
    for _ in range(num_msgs):
      blob = [random.randint(0, 255) for x in range(32)]
      print(f"Sending data {blob}")
      ep.send(
          self.schema.Struct15822124641382404136.new_message(encrypted=False,
                                                             blob=blob))

      if not cfgWritten:
        # Check that messages queue up properly waiting for the config.
        otp = [random.randint(0, 255) for x in range(32)]
        cfg.send(
            self.schema.Struct14745270011869700302.new_message(encrypt=True,
                                                               otp=otp))
        cfgWritten = True

      result = self.readMsg(ep, self.schema.Struct15822124641382404136)
      expectedResults = [x ^ y for (x, y) in zip(otp, blob)]
      print(f"Got {blob}")
      print(f"Exp {expectedResults}")
      assert (list(result.blob) == expectedResults)
