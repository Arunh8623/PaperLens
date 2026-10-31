"""A virtual clock so rate-limit tests run instantly but still check real spacing."""


class FakeTime:
    def __init__(self, start=1000.0):
        self.now = start
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s
