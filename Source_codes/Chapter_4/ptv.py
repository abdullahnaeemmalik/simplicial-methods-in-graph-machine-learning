import dgl
import numpy as np
import itertools
import torch
from torch.sparse import *
import collections
os.environ['DGLBACKEND'] = 'pytorch'
import matplotlib.pyplot as plt
import json
import tqdm


class PseudoTV():
    
    
    src=list()
    dst=list()
    empty_graph = dgl.heterograph({('node', 'to', 'node'): (src, dst)})
    dimension = int

    assert isinstance(empty_graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"graph\" of AdjGraph\'s init methodmust be a dgl.DGLHeteroGraph.'

    def __init__(
        self, file_path, graph=empty_graph,dimension=dimension):
        self.seed_graph = graph
        self.srcs_and_dsts = self.seed_graph.edges()   
        
        """Create dictionary with dimension (keys) and place list of nodes with 
        all possible top dimensions (values)"""
        self.top_vertices_dictionary = collections.defaultdict(list)
        self.top_vertices_dictionary[0]=[int(x) for x in self.seed_graph.nodes()]
        self.top_vertices_dictionary[1]=[int(x) for x in torch.unique(self.srcs_and_dsts[1])]
        """ This dict has keys = dimensions d and values = dictionary. This 
        subdictionary has keys = nodes and values =
        vertices that make the node a top-vertex of dimension d"""

        print("Finished adding 0-top vertices and 1-top vertices in main dictionary")
        
        """compute all in_degrees. These are needed for the algorithm later on 
        when the loop runs"""
        self.in_degrees = self.seed_graph.in_degrees()
        self.out_degrees = self.seed_graph.out_degrees()
        self.dimension = dimension
        self.maximum_dimension = int(torch.max(self.in_degrees))
        
        self.file_path=file_path
        
        """Create dictionary with dimension (keys) and list of nodes with maximum
        dimension corresponding to key (values)"""
        self.pseudo_top_vertices_dict = collections.defaultdict(list)
        for v in self.seed_graph.nodes():
            """Add all zero-dimension vertices, and for now keep rest as 1-dimension vertices.
            The vertices moved to different keys as more simplices are identified"""
            if self.in_degrees[v] == 0:
                self.pseudo_top_vertices_dict[0] = self.pseudo_top_vertices_dict[0] + [int(v)]
            else:
                self.pseudo_top_vertices_dict[1] = self.pseudo_top_vertices_dict[1] + [int(v)]
                
        print("Finished adding 0-pseudo top vertices and 1-pseudo top vertices")
        
        """Create empty dictionary as above, but this time will have sets 
        (value) for each dimension (key)"""        
        self.pseudo_top_vertices_dict_of_sets = dict()
        
        """Same values as above, but keys are addresses of the sets""" 
        self._sets = dict()
        
        """ Creates dictionary from above sets that has 
        representatives (keys) and sets (values)"""
        self._partition = dict()

        """ Needed for refinement. Finds vertices in same class for 
        each iteration of refinement"""
        self.refined_partition = dict()
        
        """ Number of refinements done after TV identification
        for each vertex. Needed to find final partition index"""
        self.partition_number = 0
        
        """ Dictionary with keys = nodes as values = partition 
        index of that node. Currently at zero, since
        no refinement is done, yet! The values for each key 
        is supposed to be the partition_number"""
        self.partition_index = {int_node: 0 for int_node in self.top_vertices_dictionary[0]}
        
        """Create dictionary with keys = nodes and values = 
        one-hot tensor of top-vertex and parition index,
        both concatenated"""
        self.partition_indices_and_one_hot_tv = {int_node: [] for int_node in self.top_vertices_dictionary[0]}
        
        """ Create dictionary with keys = nodes and values = 
        one hot encoding of pseudo top dimension. That is
        ith index 1 if top maximum dimension is equal to 
        index and zero otherwise """
        self.one_hot_dict = collections.defaultdict(list)

        """ Create dictionary with keys = nodes. The values 
        for the dictionary 
        are tensors which are multi hot encoding with ith 
        index 1 for i-top dimension and zero otherwise """
        self.multi_hot_tv_dict = collections.defaultdict(list)
        
        """ Create dictionary with keys = nodes and values = 
        one hot encoding with ith index 1 if vertex is refined i
        times and zero otherwise. This is an index of number of 
        times a vertex has been partitioned. """
        self.partition_times_hot_dict = collections.defaultdict(list)
        
        """ Create dictionary with keys = nodes and values = 
        one hot encoding with ith index 1 if vertex is in the
        ith partition and zero otherwise. This captures the 
        number of partitions after refinement. """
        self.partitioned_tv = collections.defaultdict(list)
        
        """ Create a dictionary of vectors R(v) for each vertex v. 
        This gets updated at each refinement process. 
        The values are stored since then the algorithm wouldn't 
        have to create the vector each time its needed"""
        self.refinement_vectors_dict = collections.defaultdict(list)
        
        """ Boolean expression to see if refinement needs to proceed"""
        self.partition_proceeds = True
        
    def kill_diag_make_binary(self,matrix):
        
        
        diagonal_mask = (matrix._indices()[0] == matrix._indices()[1])
        off_diagonal_mask = ~diagonal_mask
        #set all zero values to one where the edge is not a loop
        matrix._values()[off_diagonal_mask] = 1.0
        #create a new sparse matrix with diagonal elements killed off
        new_indices = matrix._indices()[:, off_diagonal_mask]
        #only use original nonzero values (which were later changed to 1)
        new_values = matrix._values()[off_diagonal_mask]
        new_size = matrix.size()
        return torch.sparse_coo_tensor(indices=new_indices, values=new_values, size=new_size)
    
    def hadamard_prod(self, matrix1, matrix2):
        
        
        false_hadamard_product = matrix1 * matrix2
        false_hadamard_product = false_hadamard_product.coalesce()
        non_zero_mask = false_hadamard_product._values().nonzero().squeeze()
        non_zero_values = false_hadamard_product._values()[non_zero_mask]
        non_zero_indices = false_hadamard_product.indices()[:, non_zero_mask]
        hadamard_product = torch.sparse_coo_tensor(indices=non_zero_indices,
                                                   values=non_zero_values,
                                                   size=false_hadamard_product.size())
        """Alternate approach, and the correct one without bugs.
        Isn't used."""
        #edges_1 = matrix1._indices().transpose(0, 1)
        #edges_2 = matrix2._indices().transpose(0, 1)
        #edges = find_common_tensors(edges_1, edges_2)
        #adj_size =  len(self.seed_graph.nodes())
        #indices = edges.t().long()
        #values = torch.ones(edges.shape[0], dtype=torch.int64)
        #hadamard_product = torch.sparse_coo_tensor(indices=indices, 
                                                   #values=values, size=torch.Size([adj_size, adj_size]))
        return hadamard_product

        
    def __len__(self):
        
        
        """Return the size of the partition."""
        
        
        return len(self._sets)

    def out_nodes_as_int(self, vertex):
        
        
        """convert successors to a list with integer node values"""
        
        
        neighbors = [int(v) for v in list(self.seed_graph.successors(vertex))]
        if int(vertex) in neighbors:
            neighbors.remove(int(vertex))
        return neighbors

    def in_nodes_as_int(self, vertex):
        
        
        """convert predecessors to a list with integer node values"""
        
        
        neighbors = [int(v) for v in list(self.seed_graph.predecessors(vertex))]
        if int(vertex) in neighbors:
            neighbors.remove(int(vertex))
        return neighbors       
    
    def inductive_connecting(self):
        
        
        #Case for k=2 doesn't need to search for intersections:
        print("Computing matrices of dimension 2")
        no_diag_binary = torch.sparse.mm(self.seed_graph.adj_external(),
                                         self.seed_graph.adj_external())
        no_diag_binary = self.kill_diag_make_binary(no_diag_binary) 
        self.hadamard_product_next = self.hadamard_prod(
            self.seed_graph.adj_external(),no_diag_binary)
        print("Adding 2-vertices to the dictionary")
        row_indices_analyzed_node, col_indices_analyzed_node = self.hadamard_product_next._indices()
        for vertex in col_indices_analyzed_node.unique():
            self.top_vertices_dictionary[2] = self.top_vertices_dictionary[2] + [int(vertex)]
            self.pseudo_top_vertices_dict[2] = self.pseudo_top_vertices_dict[2] + [int(vertex)]
            if vertex in self.pseudo_top_vertices_dict[1]:
                self.pseudo_top_vertices_dict[1].remove(int(vertex))
        print("Finished adding vertices of dimension 2")
        #remove these variables from memory
        row_indices_analyzed_node = None
        col_indices_analyzed_node = None
        
        for k in tqdm(range(3,self.dimension+1), position=0, leave=False):
            no_diag_binary = torch.sparse.mm(self.seed_graph.adj_external(),no_diag_binary)
            no_diag_binary = self.kill_diag_make_binary(no_diag_binary)
            #compute A⚬A^2⚬..⚬A^k where ⚬ denotes Hadamard product
            self.hadamard_product_prev  = self.hadamard_product_next.clone().detach()
            self.hadamard_product_next = self.hadamard_prod(
                self.hadamard_product_prev,no_diag_binary)
            print("Finished computing matrices for dim",k)
            
            for v in tqdm(self.seed_graph.nodes(), position=0, leave=False):
                if v in self.top_vertices_dictionary[k-1]:
                    if self.in_degrees[v] < k:
                        continue
                    else:
                        current_in_neighbors_of_node_analyzed = self.in_nodes_as_int(v)
                        for u in tqdm(self.top_vertices_dictionary[k-1], position=0, leave=False):
                            if u == v:
                                continue
                            if u not in current_in_neighbors_of_node_analyzed:
                                continue
                            else:
                                A = self.hadamard_product_prev
                                B = self.hadamard_product_next
                                A = A.coalesce()
                                B = B.coalesce()
                                A_row, A_col = A.indices()
                                B_row, B_col = B.indices()
                                if len(B_col) == 0:
                                    print("We have reached maximum dimension, and that is",k-1)
                                    self.maximum_dimension = k-1
                                    self.top_vertices_dictionary.pop(k, None)
                                    return
                                A_u_rows = A_row[torch.where(A_col.eq(u))]
                                B_v_rows = B_row[torch.where(B_col.eq(v))]
                                common_rows = np.intersect1d(A_u_rows, B_v_rows)
                                #these are all the vertices for which there is a path with unique vertices
                                #of length 1, length 2, ..., length k-1 to both u and v
                                intersection_criterion = False
                                for i in common_rows:
                                    intersection = set(self.out_nodes_as_int(i)).intersection(
                                        set(self.in_nodes_as_int(u))).intersection(set(current_in_neighbors_of_node_analyzed))
                                    if len(intersection) > k-3:
                                        intersection_criterion = True
                                        break
                                if not(intersection_criterion):
                                    continue
                                else:
                                    self.top_vertices_dictionary[k] = self.top_vertices_dictionary[k] + [int(v)]
                                    self.pseudo_top_vertices_dict[k] = self.pseudo_top_vertices_dict[k] + [int(v)]
                                    if v in self.pseudo_top_vertices_dict[k-1]:
                                        self.pseudo_top_vertices_dict[k-1].remove(int(v))
                                        """a new top vertex v is found, so we can move out of the neighborhood of v"""
                                    break
                        
            if len(self.top_vertices_dictionary[k]) == 0:
                print("We have reached maximum dimension, and that is",k-1)
                self.maximum_dimension = k-1
                self.dimension = k-1
                self.top_vertices_dictionary.pop(k, None)
                break
        print("Now creating other initial dictionaries")
        self.pseudo_top_vertices_dict_of_sets = {key:set(self.pseudo_top_vertices_dict[key]) 
                                                for key in range(0,self.dimension+1)}
        
        self._sets = {id(self.pseudo_top_vertices_dict_of_sets[key]):self.pseudo_top_vertices_dict_of_sets[key]
                                 for key in self.pseudo_top_vertices_dict_of_sets.keys()}
        self._partition = {x:self.pseudo_top_vertices_dict_of_sets[key] 
                              for key in self.pseudo_top_vertices_dict_of_sets.keys() 
                           for x in self.pseudo_top_vertices_dict_of_sets[key]}
        print("Finished creating initial partition")
        print("Serializing dictionaries..")
        data_to_save = {"pseudo_top_vertices_dict": {str(key): value 
                                                     for key, value in self.pseudo_top_vertices_dict.items()}}
        data_to_save = {
            "pseudo_top_vertices_dict": {str(key): value 
                               for key, value in self.pseudo_top_vertices_dict.items()},
            "top_vertices_dict": {str(key): value 
                               for key, value in self.top_vertices_dictionary.items()}}
        
        
        with open('pseudo_tv.json', "w") as file:
            json.dump(data_to_save, file)
        print("Dictionary of pseudo top vertices saved as a JSON file")
    
    def partition_vector(self,vertex):
        
        
        """ Create vector R(v) for vertex v """ 
        
        
        vector = [self._partition[vertex]]
        for v in torch.sort(self.seed_graph.predecessors(vertex))[0]:
            """ The representatives have to be sorted for a meaningful comparison of vectors """
            if torch.eq(v,vertex):
                pass
            else:
                vector = vector + [self._partition[int(v)]]
        return vector
        
    def refine(self):
        
        
        """Original idea for refinement algorithm by David Eppstein. 
        Refine each set A in the partition to the two sets
        A & S, A - S.  Also produces output (A & S, A - S)
        for each changed set.  Within each pair, A & S will be
        a newly created set, while A - S will be a modified
        version of an existing set in the partition.
        
        Hit here is a dictionary with keys = addresses for 
        original partitions and values = vertices with common
        partition vector"""
        
        
        hit = self.refined_partition
        output = list()
        for A,AS in hit.items():
            A = self._sets[A]
            """Check if new partition is not the same as the old partition"""
            if AS != A:
                self._sets[id(AS)] = AS
                """ This loop finds elements that are not part of previous partition"""
                for x in AS:
                    self._partition[x] = AS
                """The elements that were not part of the partition are now A"""
                A -= AS
                output = output + [set.union(A,AS)]
        """ output here keeps track of those partitions that have been broken down"""
        refined_vertices = set().union(*output)
        """The partitioning process above, once done, should then 
        increase the partition_number, if the above
        does indeed count as another genuine refinement. If it 
        does not, then the number of refined_vertices
        is zero, and hence should not increase partition number"""
        if len(refined_vertices) == 0:
            self.partition_proceeds = False
            return
        else:
            """If there is a refinement that takes place, then we 
            increase the partition number and attach
            this as the partition index for each vertex in the 
            partition_index dictionary"""
            self.partition_number = self.partition_number+1
            for v in refined_vertices:
                self.partition_index[v] = self.partition_number

    def refinement(self):
        
        
        """Keep on refining the partitions until the partition stabilizes"""
        
        
        while self.partition_proceeds:
            print("Refining..")
            """finds vertices u and v such that R(u) = R(v) and make refined partitions here"""
            common_vertices = dict()
            for node in tqdm(self.seed_graph.nodes(), position=0, leave=False):
                self.refinement_vectors_dict[int(node)] = self.partition_vector(int(node))
            print("Finished creating list of partition vectors for", 
                  "partition iteration number",self.partition_number)
            print("Finding vertices with common partition vectors..")
            """This step could be modified for optimization. It needlessly also checks
            for partion vectors of vertices that have not been partitioned the first time"""
            for v,u in tqdm(itertools.combinations(
                self.seed_graph.nodes(),2), position=0, leave=False):
                if self.refinement_vectors_dict[int(v)] == self.refinement_vectors_dict[int(u)]:
                    Au = self._partition[int(u)]
                    common_vertices.setdefault(id(Au),set()).update([int(u),int(v)])
            self.refined_partition=common_vertices  
            self.refine()
        print("Refinement process finished")

    def add_vertex_features(self):
        
        
        """First, we start with refinement until the partition stabilizes"""
        
        
        sets_for_partition_as_list = list(self._sets.values())
        print("Adding in node features..")
        for node in tqdm(self.seed_graph.nodes(), position=0, leave=False):
            """ Fills in the following dictionaries
            1) multi_hot_tv_dict
            2) one_hot_dict
            3) partition_indices_and_one_hot_tv
            4) partition_times_hot_dict
            5) partitioned_tv
            by filling in each key (node)"""
            pihvector = [0] * (self.partition_number+1)
            mhvector = [0] * (self.dimension+1)
            ohvector = [0] * (self.dimension+1)
            ptvvector = [0] * (len(self._sets))
            node = int(node)
            for dim in range(0,self.dimension+1):
                if node in self.top_vertices_dictionary[dim]:
                    mhvector[dim] = 1
                if node in self.pseudo_top_vertices_dict[dim]:
                    ohvector[dim] = 1
            pihvector[self.partition_index[node]] = 1
            self.multi_hot_tv_dict[node] = mhvector
            self.one_hot_dict[node] = ohvector
            self.partition_times_hot_dict[node] = pihvector
            temp_vector = self.one_hot_dict[node] + self.partition_times_hot_dict[node]
            self.partition_indices_and_one_hot_tv[node] = temp_vector
            index = sets_for_partition_as_list.index(self._partition[node])
            ptvvector[index] = 1
            self.partitioned_tv[node] = ptvvector
        print("Vertex features added!")
          
    def save_dicts(self):
        
        
        print("Serializing dictionaries for vertex features..")
        data_to_save = {
            "partitioned_tv": {str(key): value 
                               for key, value in self.partitioned_tv.items()},
            "partition_indices_and_one_hot_tv": {str(key): value 
                                                 for key, value in self.partition_indices_and_one_hot_tv.items()},
            "multi_hot_tv_dict": {str(key): value 
                                  for key, value in self.multi_hot_tv_dict.items()},
            "partition_times_hot_dict" : {str(key): value 
                                          for key, value in self.partition_times_hot_dict.items()}
        }
        
        with open(self.file_path, "w") as file:
            json.dump(data_to_save, file)
        print("Dictionaries saved as a JSON file")
        
    def load_dicts(self):
        
        
        with open(self.file_path, "r") as file:
            loaded_data = json.load(file)
        
        self.partitioned_tv = {int(key): torch.tensor(value)
                               for key, value in loaded_data["partitioned_tv"].items()}
        
        self.partition_indices_and_one_hot_tv = {int(key): torch.tensor(value)
                                                 for key, value in loaded_data["partition_indices_and_one_hot_tv"].items()}
        
        self.multi_hot_tv_dict = {int(key): torch.tensor(value)
                                  for key, value in loaded_data["multi_hot_tv_dict"].items()}
        
        self.partition_times_hot_dict = {int(key): torch.tensor(value)
                                         for key, value in loaded_data["partition_times_hot_dict"].items()}
        
    def load_ptv_dict(self):
        
        
        with open('pseudo_tv.json', "r") as file:
            loaded_data = json.load(file)        
        self.pseudo_top_vertices_dict = {int(key): list(value) 
                                         for key, value in loaded_data["pseudo_top_vertices_dict"].items()}
        self.top_vertices_dictionary = {int(key): list(value) 
                                         for key, value in loaded_data["top_vertices_dict"].items()}
        print("Now creating other initial dictionaries")
        self.pseudo_top_vertices_dict_of_sets = {key:set(self.pseudo_top_vertices_dict[key]) 
                                                for key in range(0,self.dimension+1)}
        self._sets = {id(self.pseudo_top_vertices_dict_of_sets[key]):self.pseudo_top_vertices_dict_of_sets[key]
                                 for key in self.pseudo_top_vertices_dict_of_sets.keys()}
        self._partition = {x:self.pseudo_top_vertices_dict_of_sets[key] for key in self.pseudo_top_vertices_dict_of_sets.keys() for x in self.pseudo_top_vertices_dict_of_sets[key]}
        print("Finished creating initial partition")

    def make_plots(self, dict_name):
        
        
        d = self.dimension    
        # x axis values 
        x = range(0, d+1)
        print("d=",d)
        # corresponding y axis values
        if dict_name not in ['partitioned_tv','tv']:
            raise ValueError("Invalid dictionary name. Must be either partitioned_tv or tv")
        y = list()
        if dict_name == 'partitioned_tv':
            for key in self._sets.keys():
                y.append(len(self._sets[key]))
            plt.figure(figsize=(10, 6))
            plt.bar(x, y, color='blue') 
            plt.xlabel('dimension') 
            plt.ylabel('Number of elements in partition')
            plt.title('representative') 
            plt.xticks(x)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('partitionedtv_plot.png')
        if dict_name == 'tv':
            for key in self.top_vertices_dictionary.keys():
                y.append(len(self.top_vertices_dictionary[key]))
            plt.figure(figsize=(10, 6))
            plt.bar(x, y, color='blue') 
            plt.xlabel('dimension') 
            plt.ylabel('Number of partitioned top vertices')
            plt.title('dimension distribution') 
            plt.xticks(x)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig('partitionedtv_plot.png')          

class PartitionError(Exception): pass