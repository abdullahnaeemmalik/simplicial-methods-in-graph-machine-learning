class SimplexCreator ():
    def __init__ (self, dimension):
        self.input_dimension = dimension
        self.src = list()
        self.dst = list()
        for i in range (self.input_dimension +1):
            for j in range (self.input_dimension +1):
                if (i < j):
                    self.src = self.src + [i]
                    self.dst = self.dst + [j]