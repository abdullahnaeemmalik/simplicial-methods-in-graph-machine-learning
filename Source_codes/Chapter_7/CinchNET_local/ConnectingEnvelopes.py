import dgl

class SimplicialFinder():
    
    empty_graph = dgl.heterograph({('node', 'to', 'node'): ([], [])})
    
    
    def __init__(self,
        graph = empty_graph):
        
        assert isinstance(graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"graph\" of AdjGraph\'s init methodmust be a dgl.DGLHeteroGraph.'
        
        self.seed_graph = graph
        
        #self.seed_nodes = self.seed_graph.nodes()
        
        src, dst = self.seed_graph.edges()
        src = src.tolist()
        dst = dst.tolist()
        self.seed_edges = (src, dst)
                
        seed_edge_pairs = []
        for i, u in enumerate(src):
            v = dst[i]
            seed_edge_pairs.append((u,v))
        self.seed_edge_pairs = seed_edge_pairs
        
        set_of_nodes = set(src+dst)
        self.seed_nodes = list(set_of_nodes)
        
        assert ((len(self.seed_edge_pairs) - len(set(self.seed_edge_pairs))) == 0),\
        'Keyword argument \"graph\" of AdjGraph\'s init method must not have multiple edges'
        
        max_vertex_index = max(self.seed_nodes)
        self.seed_edge_indices = [max_vertex_index+1+i for i in range(len(self.seed_edge_pairs))] 
        self.max_vertex_index = max(self.seed_edge_indices)
        
        self.seed_edge_zero_skeleton_dict = {max_vertex_index+1+i: edge
                                             for i, edge in enumerate(self.seed_edge_pairs)}
        
        self.vertex_index_dict = {0: self.seed_nodes,
                            1: self.seed_edge_indices}
        
        zero_skeleton_dict = {}
        """This is the dictionary of simplices. The keys are tuples, with the first entry of the 
        tuple is the dimension of the simplex, the second entry is the label of the simplex.
        The value of this dictionary is the actual position of the simplex in the graph""" 
        
        for vertex_index in self.vertex_index_dict[0]:
            zero_skeleton_dict.update({(0, vertex_index): [vertex_index]})
            
        for edge_index in self.vertex_index_dict[1]:
            zero_skeleton = self.seed_edge_zero_skeleton_dict[edge_index]
            zero_skeleton = list(zero_skeleton)
            zero_skeleton_dict.update({(1, edge_index): zero_skeleton})
        self.zero_skeleton_dict = zero_skeleton_dict
    
    def levels(self):
        levels = [key for key in self.vertex_index_dict]
        return levels
    
    def height(self):
        height = max(self.levels())
        return height
    
    def connectivity_update(self):
        """This is the method that 'connects up' our AdjGraph so that it jumps up from being the adjacency graph
        of a k-connected simplicial set to the adjacency graph of a (k+1)-connected simplicial set. This is done
        by adding in (k+1)-simplicies where ever our simplicial set contains a non-degenerate boundary of a
        standard (k+1)-simplex.
        
        At a high level, this method procedes as follows:
        1. Locate all (0-simplex, k-simplex)-pairs (u, s) such that the vertex u is not contained 
        in the k-simplex s.
        2. For each such pair, let [v_0, v_1, ...., v_k] be the 0-skeleton sk_0(s) of our 
        k-simplex s.
        3. For each resulting (0-simplex, 0-simplex)-pair (u, v_i), for 0<=i<=k, query if 
        [u, v_i] is a directed edge in our original graph, for all i ([v_i,u]?)
        4. If the answer is affirmative for every pair (u, v_i) in Step 3 above, then 
        [u, v_0, v_1, ...., v_k] is the 0-skeleton of a (k+1)-simplex that needs to be 
        added to AdjGraph.
        """
        
        # record the height of AdjGraph. This becomes the integer k as used in the commentimmediately above:
        height = self.height()
        
        # Create a list of the 0-skeleta of all top-dimensional simplices in our current simpliciat set:
        top_dim_zero_skeleta = []
        for key in self.zero_skeleton_dict:
            if key[0] == height:
                top_dim_zero_skeleta.append(self.zero_skeleton_dict[key])
        
        # Create a list of the 0-skeleta of all 1-simplices, i.e., edges, in our current simplicial set:
        edge_zero_skeleta = []
        for key in self.zero_skeleton_dict:
            if key[0] == 1:
                edge_zero_skeleta.append(self.zero_skeleton_dict[key])
        
        # Begin list of all non-degenerate boundaries of standard (k+1)-simplices 
        #in our current simplicial set:
        top_dim_boundaries = []
        # Step 1: Iterate of 0-simplices in our current simplicial set:
        for src in self.vertex_index_dict[0]:
            # Step 2: iterate over k-simplices in our current simplicial set, extracting the 0-skeleton of each:
            for zero_skel in top_dim_zero_skeleta:
                # Check that our new 0-simplex doesn't already lie in the 0-skeleton of our k-simplex:
                if src in zero_skel:
                    forms_boundary_query = False
                # If it doesn't, begin Step 3: 
                else:
                    forms_boundary_query = True
                    # Check that our original graph contains all necessary edges:
                    for dst in zero_skel:
                        edge_present_query = ([src, dst] in edge_zero_skeleta)
                        forms_boundary_query *= edge_present_query
                        if forms_boundary_query == False:
                            break
                    # If it does, adjoin a new 0-skeleton to our list of (k+1)-simplices to add to AdjGraph:
                    if forms_boundary_query == True:
                        top_dim_boundaries.append([src]+zero_skel)
        
        # Update AdjGraph class attributes.
        # We've added simplices of one dimension higher, so the height increases:
        new_height = 1 + self.height()
        
        # Since we have new simplices, we have new vertices in AdjGraph, so we need to update vertex indices.
        # Fix the old maximum index for vertices in AdjGraph before we update it:
        old_max_vertex_index = self.max_vertex_index
        # Count how many new simplices we have, so how many new vertices we'll be adding to AdjGraph:
        new_simplex_count = len(top_dim_boundaries)
        # Update the attribute max_vertex_index by adding the new vertex count to our previous one:
        self.max_vertex_index += new_simplex_count
        # Create a list of indices for our new simplices, picking up at our previous largest index:
        new_indices = [old_max_vertex_index+1+i for i in range(new_simplex_count)]
        # Update our dictionary of vertex indices by introducing a new key, corresponding to our new height,
        # and define the value at this new key to be the list of all our new indices:
        self.vertex_index_dict.update({new_height: new_indices})
        # Update our dictionary of 0-skeleta associated to vertices in AdjGraph by introducing one new key
        # (simplex dimension, simplex index) for each new simplex in AdjGraph, and define the value at this
        # new key to be the 0-skeleton of this simplex:
        for i, zero_skeleton in enumerate(top_dim_boundaries):
            simplex_index = self.vertex_index_dict[new_height][i]
            self.zero_skeleton_dict.update({(new_height, simplex_index): zero_skeleton})
    def highest_dimension(self):
        for dimension, vertices in self.vertex_index_dict.items():
            if dimension == self.height():
                if vertices == []:
                    return True
        return False