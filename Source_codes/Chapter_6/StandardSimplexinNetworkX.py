import networkx as nx

class MaximalSet():
    def __init__(self,d):
        self.d = d

    def PosetΔ(self):
        return [index for index in range(0, self.d+1)]

    def SimplexΔ(self,n):
        if n == 0:
            return(self.PosetΔ())
        Xbulletn = [[v] for v in self.PosetΔ()]
        for i in range(n):
            Xbulletn = [r + [v] for r in Xbulletn for v in self.PosetΔ() if r[-1] <= v] 
        return Xbulletn

    def DegenerateOrNot(self,entryofalist):
        if len(entryofalist) == len(set(entryofalist)):
            return False
        else:
            return True

    def DegeneracyName(self,d,j):
        return "s_{} {}".format(d,j)

    def FaceName(self,d,j):
        return "d_{} {}".format(d,j)

    def MapValue(self):
        Xgraph = nx.MultiDiGraph()
        for index in range(0,self.d+1):
           for listitem in MaximalSet(self.d).SimplexΔ(index):
               firstlist = MaximalSet(self.d).SimplexΔ(index)
               secondlist = MaximalSet(self.d).SimplexΔ(index+1)
               if set(firstlist).issubset(set(secondlist)):
                   for index, (first, second) in enumerate(zip(firstlist, secondlist)):
                       if first != second:
                           Xgraph.add_edge(str(firstlist), str(secondlist), degeneracy=MaximalSet(self.d).DegeneracyName(self.d,first))
               if set(secondlist).issubset(set(firstlist)):
                    for index, (first, second) in enumerate(zip(firstlist, secondlist)): 
                        if first != second:
                            Xgraph.add_edge(str(secondlist), str(firstlist), face=MaximalSet(self.d).DegeneracyName(self.d,second))
        return Xgraph