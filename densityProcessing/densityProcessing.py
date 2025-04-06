from functools import reduce

class DensityProcessing:
    def __init__ (self, operations):
        self.operations = operations if operations else []

    def apply(self, images):
        return [reduce(lambda data, op: op(data), self.operations, img) for img in images]