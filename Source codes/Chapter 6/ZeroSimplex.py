class ZeroSimplicialSet:
    def __init__(self, point):
        self.point = point

    def simplicialset(self, num):
        if num == 0:
            return([self.point])
        X=[self.point]
        for i in range(num):
            X = X + [self.point]
        return(X)

    def facemap(self, level):
        """prints out face map of the given level"""
        return f"d({self.simplicialset(level)}) = {self.simplicialset(level-1)}"

    def degeneracymap(self, level):
        """prints out degeneracy map of the given level"""
        return f"s({self.simplicialset(level)}) = {self.simplicialset(level+1)}"