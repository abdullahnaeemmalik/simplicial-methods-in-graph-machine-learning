class KanofDelta():
    def __init__(self,delta_set):
        self.delta_set = delta_set
        self.simplicial_set = {0: delta_set[0]}
        self.dimensions      = list(self.delta_set.keys())
        self.max_dim         = max(self.dimensions)

    def degenerate_map(self,k,j,input_simplex):
        if j > k:
            print("Error! j=",j,"must be less than the dimension k=",k)
            return
        if k == len(input_simplex) - 1:
            output_simplex = input_simplex[:j+1] + input_simplex[j:]
            return output_simplex
        else:
            print("The simplex", input_simplex, "is not part of the domain of s_{}^{}".format(j,k))

    def level_up(self,dimension):
        current_dim_simplices = self.delta_set[dimension] + self.simplicial_set.get(dimension,[])
        temp = list()
        for simplex in current_dim_simplices:
            for j in range(dimension+1):
                result = self.degenerate_map(dimension,j,simplex)
                if result not in temp:
                    temp.append(result)
        temp = temp + self.delta_set.get(dimension+1)
        self.simplicial_set.update({dimension+1:temp})


    def universal_completion(self):
        for d in range(self.max_dim):
            self.level_up(d)
        temp = list()
        for simplex in self.simplicial_set[d+1]:
            for j in range(d+2):
                result = self.degenerate_map(d+1,j,simplex)
                if result not in temp:
                    temp.append(result)
        self.simplicial_set.update({d+2:temp})