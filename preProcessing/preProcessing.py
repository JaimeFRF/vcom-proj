from functools import reduce

class PreProcessing:
    def __init__ (self, operations):
        self.operations = operations if operations else []

    def apply(self, images):
        return [reduce(lambda img, op: op(img), self.operations, img) for img in images]
        