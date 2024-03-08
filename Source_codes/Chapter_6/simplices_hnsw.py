import collections
import copy
import itertools
import json
import math
import os
import random
import sqlite3
import dgl
import torch
from tqdm import tqdm
from torch.sparse import *
from tabulate import tabulate
import numpy as np

os.environ['DGLBACKEND'] = 'pytorch'

class Unpacker():
    """input of quotiented_simplices is a dictionary with keys = (dimension, simplex_id, globe_id)
    and values = [v_0,v_1,...,v_n]. Node mapping is also a dictionary with keys = nodes
    and values = a list of pre-images. Such a list could be a singleton, but never empty
    because the quotient map is a function in the math sense. The other inputs concern the graph downstairs (range graph)
    and the graph upstairs (domain graph).
    
    This function pushes simplices from downstairs to upstairs, and does this by looking at degenerate simplice downstairs,
    then finds possible preimages for them upstairs, and does so inductively. The highest dimension for which we need
    to look for a simplex is dictated by the graph upstairs, and is decided by self.local_max_dim.
    
    Finally, the simplex_max_index takes the index from the previous section finder, and increments it for simplices
    that are found later."""
    
    def __init__(self, domain_graph,node_mapping,range_graph_edges,
                 deg_edges,quotiented_simplices,simplex_max_index,globe_number,domain_graph_edges):
        
        assert isinstance(domain_graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"domain_graph\" must be a dgl.DGLHeteroGraph.'
        assert isinstance(quotiented_simplices, dict), \
        'Keyword argument \"edges_list\" must be a dictionary.'
        assert isinstance(node_mapping, dict), \
        'Keyword argument \"node_mapping\" must be a dictionary.'
        assert isinstance(deg_edges, list), \
        'Keyword argument \"deg_edges\" must be a dictionary.'

        self.simplex_max_index     = simplex_max_index
        self.domain_graph          = domain_graph
        self.node_mapping          = node_mapping
        self.deg_edges             = deg_edges
        self.range_graph_edges     = range_graph_edges
        self.quotiented_simplices  = quotiented_simplices
        self.lifted_simplices      = dict()
        self.globe_number          = globe_number
        number_of_nodes            = len(torch.unique(torch.cat(self.domain_graph.edges())))
        in_degrees                 = self.domain_graph.in_degrees()
        out_degrees                = self.domain_graph.out_degrees()
        possible_max1              = min(int(torch.max(in_degrees)), number_of_nodes-1)
        possible_max2              = min(int(torch.max(out_degrees)), number_of_nodes-1)
        self.local_max_dim         = max(possible_max1,possible_max2)
        self.domain_graph_edges    = domain_graph_edges
        self.preimages_collection  = list()
        self.injective_nodes       = list()
        self.non_injective_nodes   = list()
        self.truquotiented_simpl   = copy.deepcopy(quotiented_simplices)
        self.new_simplices         = list()
        self.all_range_g_edges     = self.range_graph_edges + self.deg_edges
        
        # Seperate the nodes that are mapped injectively
        for node, preimages in self.node_mapping.items():
            if len(preimages) == 1:
                self.injective_nodes = self.injective_nodes + [node]
            else:
                self.non_injective_nodes = self.non_injective_nodes + [node]
        
        index = 0
        _nodes = set(self.non_injective_nodes)
        for key, quotiented_simplex in self.quotiented_simplices.items():
            if key[0] > 1:
                common_elements = _nodes.intersection(set(quotiented_simplex))
                if len(common_elements) == 0:
                    # If the simplex downstairs is the same as the simplex upstairs, then add it to the dictionary
                    key_lifted = (len(quotiented_simplex)-1, self.simplex_max_index+index, self.globe_number)
                    self.lifted_simplices.update({key_lifted:quotiented_simplex})
                    #T he simplex found is not a degenerate simplex upstairs, so we don't need to find its sections.
                    del self.truquotiented_simpl[key]
                    index = index + 1
                else:
                    # Work out the simplices upstairs that have images of the same dimension downstairs
                    degeneracies = self.generate_youngs_list(quotiented_simplex)
                    for degeneracy in degeneracies:
                        if self.simplex_verifier(degeneracy):
                            key_lift = (len(degeneracy)-1, self.simplex_max_index+index, self.globe_number)
                            self.lifted_simplices.update({key_lift:degeneracy})
                            index = index + 1
        self.simplex_max_index = self.simplex_max_index + index
        self.new_simplices = list(self.truquotiented_simpl.values()) + self.range_graph_edges
        
    def inductive_connecting(self):          
        new_zero_skeleta = self.new_simplices
        self.images = []
        for src in self.non_injective_nodes:
            for zero_skel in new_zero_skeleta:
                if len(zero_skel) > self.local_max_dim:
                    continue
                if zero_skel.count(src) >= len(self.node_mapping[src]):
                    continue
                if src not in zero_skel:
                    continue
                edge_present_query1 = True
                edge_present_query2 = True
                for dst in zero_skel:
                    if edge_present_query1:
                        if [src, dst] in self.all_range_g_edges:
                            index_to_insert = zero_skel.index(src)
                            if zero_skel[:index_to_insert]+[src]+zero_skel[index_to_insert:] not in self.images:
                                self.images.append(zero_skel[:index_to_insert]+[src]+zero_skel[index_to_insert:])
                        if [dst, src] in self.all_range_g_edges:
                            index_to_insert = zero_skel.index(dst)
                            if zero_skel[:index_to_insert-1]+[src]+zero_skel[index_to_insert-1:] not in self.images:
                                self.images.append(zero_skel[:index_to_insert-1]+[src]+zero_skel[index_to_insert-1:])
       
        if len(self.images) == 0:
            # No more images are found, so we can stop the induction process
            return
        
        else:
            self.preimages_collection = []
            self.preimages_collection_id = dict()
            for image in self.images:
                preimages = self.generate_youngs_list(image)
                for preimage in preimages:
                    self.preimages_collection_id.update({id(preimage):image})
                self.preimages_collection = self.preimages_collection + preimages
            self.add_simplices_to_dict()
            self.new_simplices = [list(image) for image in self.simplices_to_extend]
            self.inductive_connecting()
    
    def generate_youngs_list(self, simplex):
        """Takes as input the image of a quotiented simplex spits out all possible simplices in the pre-image"""
        
        indices = [range(len(self.node_mapping[key])) for key in simplex]
        combinations = itertools.product(*indices)
        result_lists = []
        for combo in combinations:
            alternative_list = [self.node_mapping[simplex[0]][combo[0]]]
            for i in range(1, len(simplex)):
                to_add = True
                p_node = self.node_mapping[simplex[i-1]][combo[i-1]]
                c_node = self.node_mapping[simplex[i]][combo[i]]
                if (p_node,c_node) in self.domain_graph_edges:
                    if c_node not in alternative_list: 
                        alternative_list.append(self.node_mapping[simplex[i]][combo[i]])
                    else:
                        to_add = False
                        break
                else:
                    to_add = False
                    break
            if to_add:
                if alternative_list not in result_lists:
                    result_lists.append(alternative_list)
        return result_lists

    def simplex_verifier(self, simplex):
        """This function takes input a k-tuple [v_1,v_2,...,v_k] for k>1 and a graph G
        and verifies if it a simplex of G. This is done by simply checking if there is 
        an edge between v_i and v_j for i<j"""
    
        is_simplex = False
        simplex_reduced = copy.deepcopy(simplex)
        
        for i in simplex:
            simplex_reduced.pop(0)
            for j in simplex_reduced:
                if (i, j) in self.domain_graph_edges:
                    continue
                else:
                    # The input is not a simplex, so we can stop our check.
                    return is_simplex
        return True
    
    def add_simplices_to_dict(self):
        index = 0
        self.simplices_to_extend = set()

        for preimage in self.preimages_collection:
            if self.simplex_verifier(preimage):
                key = (len(preimage) - 1, self.simplex_max_index + index, self.globe_number)
                self.lifted_simplices.update({key: preimage})
                index += 1
                temp_dict = {tuple(self.preimages_collection_id[id(preimage)])}
                self.simplices_to_extend = self.simplices_to_extend.union(temp_dict)
        self.simplex_max_index = self.simplex_max_index + index 
    
    def simplex_breaker(self,simplex):
        """This function takes a simplex and returns all the edges within it. I abondoned the plan to use it."""
        edges = list()
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                x = lst[i]
                y = lst[j]
                edges = edges + [[x,y]]
        return edges
		
def edges_to_identify(graph):
    """Constructing the Hadamard product here to find the list of edges to contract.
    All the edges that are parts of a 2-simplex are identified in this function
    as indices of the Hadamard product. Moreover, other edges that are not part of the Hadamard
    product, but are part of the transitive closure of existing edges are also identified.
    These are the edges that form a complete 2-simplex."""

    assert isinstance(graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"graph\" of create_hadamard must be a dgl.DGLHeteroGraph.'
    
    # First, we remove all diag entries from adj matrix A by removing all self-loops
    loopless = dgl.transforms.RemoveSelfLoop()
    graph = loopless(graph)
    
    # Then we remove all diagonals from A^2 and convert the matrix to one with binary entries
    adj_squared = torch.sparse.mm(graph.adj_external(),graph.adj_external())
    diagonal_mask = (adj_squared._indices()[0] == adj_squared._indices()[1])
    off_diagonal_mask = ~diagonal_mask
    adj_squared._values()[off_diagonal_mask] = 1.0
    new_indices = adj_squared._indices()[:, off_diagonal_mask]
    new_values = adj_squared._values()[off_diagonal_mask]
    new_size = adj_squared.size()
    squared_no_diag_binary = torch.sparse_coo_tensor(indices=new_indices, 
                                                    values=new_values, size=new_size)
    # The Hadamard product is sparse, but keeps track of entries that are zero.
    false_hadamard_product = graph.adj_external() * squared_no_diag_binary
    
    # We, therefore need to remove those entries.
    false_hadamard_product = false_hadamard_product.coalesce()
    non_zero_mask = false_hadamard_product._values().nonzero().squeeze()
    non_zero_values = false_hadamard_product._values()[non_zero_mask]
    non_zero_indices = false_hadamard_product.indices()[:, non_zero_mask]
    if non_zero_indices.dim() == 1:
        non_zero_indices = non_zero_indices.unsqueeze(0)
        non_zero_indices = non_zero_indices.view(2, -1)
    hadamard_product = torch.sparse_coo_tensor(indices=non_zero_indices,
                                               values=non_zero_values,
                                               size=false_hadamard_product.size())
    
    # The following loop finds all edges that are part of a simplex need to be collapsed.
    row_indices, col_indices = hadamard_product._indices()
    extra_edges = list()
    
    for i,j in zip(row_indices,col_indices):
        out_nodes = set([int(v) for v in list(graph.successors(i))])
        in_nodes = set([int(v) for v in list(graph.predecessors(j))])
        # These are the elements in the (reverse?) transitive closure of (i,j).
        intersection = set.intersection(out_nodes,in_nodes)
        for k in intersection:
            extra_edges = extra_edges + [(int(i),int(k))] + [(int(k),int(j))]
    return hadamard_product, extra_edges

class graph_towers():
    src=list()
    dst=list()
    empty_graph = dgl.heterograph({('node', 'to', 'node'): (src, dst)})
    ratio = 0.0
    bottom_level = 0
    
    assert isinstance(empty_graph, dgl.DGLHeteroGraph), \
        'Keyword argument \"graph\" of graph_towers\'s init method must be a dgl.DGLHeteroGraph.'
    assert isinstance(ratio, float), \
        'Keyword argument \"ratio\" of graph_towers\'s init method must be a float.'
    assert ratio<=1 and ratio>=0, \
        'Keyword argument \"ratio\" of graph_towers\'s init method must be between 0 and 1.'
    assert isinstance(bottom_level, int), \
        'Keyword argument \"bottom_level\" of graph_towers\'s init method must be an integer.'

    def __init__(self, ratio, database_name, graph=empty_graph,
                 bottom_level=bottom_level):
        
        self.seed_graph        = graph
        self.srcs_and_dsts     = self.seed_graph.edges()        
        self.ratio             = ratio
        self.updated_graph     = dgl.heterograph({('node', 'to', 'node'): ([], [])})
        self.bottom_level      = bottom_level
        self.database_name     = database_name
        self.maximum_dimension = None
        self.connection        = None
        self.cursor            = None
        self.number_of_nodes   = len(self.seed_graph.nodes())
        self.number_of_edges   = len(self.seed_graph.edges()[0])
        self.selected_edges    = None
        self.quotient_number   = 0
        self.simplex_id        = 0
        
        # Find list of edges that will be used to create a quotient graph
        self.hadamard_product, self.extra_edges  = edges_to_identify(self.seed_graph)
        rows, columns = self.hadamard_product._indices()
        self.edges_to_collapse_as_pairs = torch.cat(
            (torch.transpose(self.hadamard_product._indices(),0,1)
             ,torch.tensor(self.extra_edges)),dim=0)
        
        # Some edges from the 'extra_edges' and those given by the Hadamard product
        # are duplicated. We need to combine these in one variable.   
        self.edges_to_collapse_as_pairs = torch.unique(self.edges_to_collapse_as_pairs, dim=0)
        self.all_nodes_to_identify      = torch.cat((rows,columns),dim=0).unique()
        rows                            = None
        columns                         = None
        self._equivalenceclasses        = dict()
        self.appendage_index            = len(self.seed_graph.nodes())
        self.edge_index                 = 0        
        self.all_edges_as_pairs         = torch.stack(self.seed_graph.edges(), dim = 1).int()
        self.edges_carry_fwd            = list()
        self.edges_never_contracted     = None
        self.simplex_id                 = 0
        self._all_sets_with_indices     = dict()
        
        # Find all node classes to yield maximum class size.
        self._globes = {element:partition for partition in relation(
            self.edges_to_collapse_as_pairs) for element in partition}
        
        index = 0
        for key in self._globes.keys():
            key_check = self._all_sets_with_indices.get(id(self._globes[key]),[])
            if key_check == []:
                self._all_sets_with_indices.update({id(self._globes[key]):(self._globes[key],index)})
                index = index + 1
                
        self.max_globe_index    = index-1
        index                   = None
        self.maximum_class_size = max((len(set_value) for set_value in self._globes.values()), default=0)
        self.loop_indicator     = torch.eq(self.srcs_and_dsts[0],self.srcs_and_dsts[1])
        self.existing_loops     = (self.seed_graph.edges()[0][self.loop_indicator],
                                   self.seed_graph.edges()[1][self.loop_indicator])
        self.existing_loops     = torch.stack(self.existing_loops, dim = 1).int()
        
        # Finds globally highest dimension of simplices in the graph
        self.in_degrees        = self.seed_graph.in_degrees()
        self.out_degrees       = self.seed_graph.out_degrees()
        self.maximum_dimension = min(int(torch.max(self.in_degrees)), 
                                     int(torch.max(self.out_degrees)),self.maximum_class_size)
            
    def _close_db(self):
        self.connection.commit()
        self.connection.close()
        
    def _connect_db(self):
        self.connection = sqlite3.connect(self.database_name)
        self.cursor = self.connection.cursor()
        
    def _view_db(self,table_name):
        """This function was created for easier visualization of the database"""
        self._connect_db()
        self.cursor.execute(f"PRAGMA table_info({table_name})")
        if table_name == 'edge_details':
            columns_info = self.cursor.fetchall()
            reduced_columns = columns_info[1:10] 
            columns = [column[1] for column in reduced_columns]
            self.cursor.execute(f"SELECT {', '.join(columns)} FROM {table_name}")
            rows = self.cursor.fetchall()
        else:
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [column[1] for column in self.cursor.fetchall()]
            self.cursor.execute(f"SELECT * FROM {table_name}")
            rows = self.cursor.fetchall()
        
        table = tabulate(rows, headers=columns, tablefmt="pretty")
        print(table)
        self._close_db()
        
        
    def create_table(self):
        """This function creates the table, but does not intialize
        data. Therefore, this function only needs to be run the first time. """
        self._connect_db()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph (
                quotient_number INTEGER PRIMARY KEY,
                number_of_nodes INTEGER,
                number_of_edges INTEGER,
                number_of_simplices INTEGER,
                FOREIGN KEY (number_of_nodes) REFERENCES node_classes(node_class_id),
                FOREIGN KEY (number_of_edges) REFERENCES edge_details(edge_id),
                FOREIGN KEY (number_of_simplices) REFERENCES simplices(simplex_id)
                                            )
                            ''')
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS node_classes (
                node_class_id INTEGER PRIMARY KEY,
                quotient_id INTEGER,
                number_of_nodes INTEGER,
                globe_id INTEGER,
                nodes INTEGER)
                            ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS edge_details (
                edge_id INTEGER PRIMARY KEY,
                e_src INTEGER,
                e_dst INTEGER,
                edge_changed BOOLEAN,
                to_contract BOOLEAN,
                sampled BOOLEAN,
                contracted BOOLEAN,
                quotient_id INTEGER,
                multiplicity INTEGER,
                globe_id INTEGER,
                number_of_edges INTEGER
                                                    )
                            ''')
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS simplices (
                simplex_id INTEGER PRIMARY KEY,
                quotient_id INTEGER,
                dimension INTEGER,
                globe_id INTEGER,
                number_of_simplices INTEGER,
                simplex_vertices)
                            ''')
        self._close_db()
        
        
    def initial_db_fill(self):
        """Fills in the database with the details from the (unquotiented) graph itself
        i.e., before the quotienting process. """
        self._connect_db()
        self.cursor.execute('''INSERT INTO graph 
                            (quotient_number, number_of_nodes, number_of_edges, 
                            number_of_simplices) VALUES
                            (0, ?, ?, 0)
                            ''', (len(self.seed_graph.nodes()), 
                                  len(self.seed_graph.edges()[0]))
                           )
        
        for node in self.seed_graph.nodes():
            picked_set = self._globes.get(int(node),{node})
            picked_id  = self._all_sets_with_indices.get(id(picked_set),'X') 
            if picked_id == 'X':
                globe_id = -1
            else:
                globe_id = picked_id[1]
            self.cursor.execute('''INSERT INTO node_classes
                                (node_class_id, nodes, quotient_id, 
                                number_of_nodes, globe_id) VALUES 
                                (?, ?, 0, ?, ?)
                                ''', (int(node), int(node),  
                                      len(self.seed_graph.nodes()),
                                      globe_id))
        for edge in self.all_edges_as_pairs:
            if torch.any(torch.all(self.edges_to_collapse_as_pairs == edge, dim=1)):
                picked_set = self._globes.get(int(edge[0]),{int(edge[0])})
                picked_id  = self._all_sets_with_indices[id(picked_set)] 
                globe_id   = picked_id[1]

                self.cursor.execute('''INSERT INTO edge_details
                                    (edge_id, e_src, e_dst, edge_changed, 
                                    to_contract, quotient_id, multiplicity, sampled,
                                    number_of_edges, contracted, globe_id) VALUES 
                                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (self.edge_index, int(edge[0]), int(edge[1]), 
                                      False, True, 0, 1, False, len(self.seed_graph.edges()[0]),
                                          False, globe_id)
                                   )
            elif torch.any(torch.all(self.existing_loops == edge, dim=1)):
                self.cursor.execute('''INSERT INTO edge_details
                                    (edge_id, e_src, e_dst, edge_changed, 
                                    to_contract, quotient_id, multiplicity, sampled,
                                    number_of_edges, contracted, globe_id) VALUES 
                                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (self.edge_index, int(edge[0]), int(edge[1]), 
                                      False, False, 0, 1, False, len(self.seed_graph.edges()[0]),
                                          True, -1)
                                   )

            else:
                self.cursor.execute('''INSERT INTO edge_details
                                    (edge_id, e_src, e_dst, edge_changed, 
                                    to_contract, quotient_id, multiplicity, sampled,
                                    number_of_edges, contracted, globe_id) VALUES 
                                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    ''', (self.edge_index, int(edge[0]), int(edge[1]), 
                                      False, False, 0, 1, False, len(self.seed_graph.edges()[0]),
                                         False, -1)
                                   )
            self.edge_index = self.edge_index + 1
        
        self._close_db()
        
    def sampling(self):
        """Samples edges to collapse, keeps track of changes in db for edges remaining to collapse."""
        self._connect_db()
        # This query returns edges that have not yet been contracted.
        self.cursor.execute(f'''SELECT e_src, e_dst
                            FROM edge_details 
                            WHERE to_contract = True AND sampled = False
                            AND edge_changed = False
                            AND contracted = False AND quotient_id = {self.quotient_number}
                            ORDER BY RANDOM()
                            LIMIT {math.ceil(self.ratio * len(self.edges_to_collapse_as_pairs))}
                            ''')
        
        self.selected_edges = self.cursor.fetchall()
        # If two edges collapsing result in a third edge to collapse, we need to find it.
        self._equivalenceclasses = {element:partition 
                                    for partition in relation(self.selected_edges) 
                                    for element in partition}
        # When an edge collapses, we assign a new node label. The following list keeps track of this information.
        changed_nodes = list(self._equivalenceclasses.keys())
        """To update entries of sampled edges in db, we first find all
        the unique classes constructed above."""
        unique_sets = set()
        for value in self._equivalenceclasses.values():
            unique_sets.add(frozenset(value))
        selected_edges = []
        
        for equivalence_class in unique_sets:
            """updates the column entry `sampled' for edges that have been sampled and those
            that naturally won't be available for future quotienting"""
            self.cursor.execute('''UPDATE edge_details
                                SET sampled = True
                                WHERE e_src IN ({}) AND e_dst IN ({}) AND quotient_id = ?
                                '''.format(','.join(['?']*len(equivalence_class)),
                                           ','.join(['?']*len(equivalence_class))),
                                list(equivalence_class) + list(equivalence_class) +
                                [self.quotient_number]
                               )
            
            self.cursor.execute('''SELECT e_src, e_dst
                                FROM edge_details
                                WHERE contracted = False AND
                                globe_id IS NOT -1 AND
                                e_src IN ({}) AND e_dst IN ({})
                                AND quotient_id = ?
                                '''.format(','.join(['?']*len(equivalence_class)), 
                                            ','.join(['?']*len(equivalence_class))),
                                    list(equivalence_class) + list(equivalence_class) +
                               [self.quotient_number]
                               )
            selected_edges = selected_edges + self.cursor.fetchall()
        self.selected_edges = selected_edges

        """Find out edges to carry forward"""
        self.cursor.execute(f'''SELECT e_src, e_dst
                            FROM edge_details 
                            WHERE to_contract = True AND sampled = False 
                            AND edge_changed = False AND contracted = False
                            AND quotient_id ={self.quotient_number}
                            ''')
        
        """These are the egdes we do  not collapse after sampling and considering transitive closure"""
        self.edges_carry_fwd = self.cursor.fetchall()
        
        self.cursor.execute('''UPDATE edge_details
                            SET edge_changed = True
                            WHERE e_src IN ({}) AND e_dst IN ({});
                            '''.format(','.join(['?']*len(changed_nodes)),
                                       ','.join(['?']*len(changed_nodes))), 
                            changed_nodes + changed_nodes)
        self._close_db()
        
    def make_quotient(self):
        """This section can be combined with the db fill function above
        The only reason this is kept distinct is to keep the class modular"""
        self.new_edges_added = list()
        self.new_nodes_added = set()
        
        self._sets = {id(self._equivalenceclasses[key]):self._equivalenceclasses[key]
                      for key in self._equivalenceclasses.keys()}
        # Create mapping of class names
        self._classesnamesmapping = dict()
        
        for setid in self._sets.keys():
            self._classesnamesmapping[setid] = self.appendage_index
            self.appendage_index = self.appendage_index + 1
                    
        for key, value in self._classesnamesmapping.items():
            element_of_set = next(iter(self._sets[key]))
            globe_associated = self._globes[element_of_set]
            self._globes.update({value:globe_associated})
                    
        # Each set (node class) is assigned a new node label.
        for edge in self.selected_edges:
            nodeclass    = self._equivalenceclasses[edge[0]]
            newnodelabel = self._classesnamesmapping[id(nodeclass)]
            self.new_edges_added = self.new_edges_added + [(newnodelabel,newnodelabel)]
            self.new_nodes_added = self.new_nodes_added.union({newnodelabel})

        # Since the node labels are changed, the edges that still need to be contracted will
        # have their src and dst changed. To keep track of these un-contracted edges, we put them in the 
        # edges_carry_fwd variable
        for edge in self.edges_carry_fwd:
            srcnodeclass    = self._equivalenceclasses.get(edge[0], edge[0])
            srcnewnodelabel = self._classesnamesmapping.get(id(srcnodeclass),edge[0])
            dstnodeclass    = self._equivalenceclasses.get(edge[1], edge[1])
            dstnewnodelabel = self._classesnamesmapping.get(id(dstnodeclass),edge[1])
            self.new_edges_added = self.new_edges_added + [(srcnewnodelabel,dstnewnodelabel)]
            self.new_nodes_added = self.new_nodes_added.union({srcnewnodelabel,dstnewnodelabel})
            
        
        # Remove variables to save space
        self.edges_carry_fwd = None
        self.selected_edges  = None
        
        self.quotient_number = self.quotient_number + 1
        
    def db_fill(self):
        """This function saves the details of a quotiented graph in the database.
        Therefore, this function should be called immediately after the quotient is made.
        All the edges of the graph are added, and are given different IDs, even if the edges
        are already present in a previous quotient. However, we have different collections
        here to ensure that the edges that have been changed are discarded from the edges
        that need to be sampled. To this end, we add three new columns viz. to_contract, sampled
        and contracted."""
        # To count the multiplicities of the edges
        edge_counter        = collections.Counter(self.new_edges_added)
        number_of_edges     = len(edge_counter)
        number_of_nodes     = len(self.new_nodes_added)
        
        self._connect_db()
        self.cursor.execute('''INSERT INTO graph 
                            (quotient_number, number_of_nodes, number_of_edges, 
                            number_of_simplices) VALUES
                            (?, ?, ?, 0)
                            ''', (self.quotient_number, 
                                  number_of_nodes,
                                  number_of_edges)
                           )
        
        edge_details_data = list()
        for edge, count in edge_counter.items():
            picked_set = self._globes[int(edge[0])]
            picked_id = self._all_sets_with_indices[id(picked_set)]
            globe_id = picked_id[1]
            values = (self.edge_index, int(edge[0]), int(edge[1]),
                      False, True, self.quotient_number, count,
                      edge[0] == edge[1], number_of_edges,
                      edge[0] == edge[1], globe_id)
            edge_details_data.append(values)
            self.edge_index += 1
            
        query = '''INSERT INTO edge_details 
                    (edge_id, e_src, e_dst, edge_changed, 
                    to_contract, quotient_id, multiplicity, sampled,
                    number_of_edges, contracted, globe_id) VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''

        self.cursor.executemany(query, edge_details_data)
            
        # Removing variable to save space
        self.new_edges_added = None
        edge_details_data = None
        
        node_classes_data = []

        for setid, classlabel in self._classesnamesmapping.items():
            node_data = self._sets[setid]
            node = next(iter(node_data))
            picked_set = self._globes.get(int(node),{node})
            picked_id  = self._all_sets_with_indices.get(id(picked_set),'X')
            if picked_id == 'X':
                globe_id = -1
            else:
                globe_id = picked_id[1]
            serialized_node_data = json.dumps(list(node_data))  
            data_row = [globe_id, classlabel, self.quotient_number, number_of_nodes, serialized_node_data]
            node_classes_data.append(data_row)
            
        columns = ['globe_id', 'node_class_id', 'quotient_id', 'number_of_nodes', 'nodes']
        placeholders = ', '.join(columns)
        query = f'''INSERT INTO node_classes ({placeholders})
                    VALUES ({', '.join(['?'] * len(columns))})
                    '''
        self.cursor.executemany(query, node_classes_data)
        self._close_db()
        
                
    def empty_table(self, table_name):
        """Made this function to empty database without deleting database. 
        Helpful for debugging"""
        self._connect_db()
        self.cursor.execute(f"DELETE FROM {table_name}")
        self._close_db()
        
    def create_towers(self):
        self.initial_db_fill()
        for index in range(self.bottom_level+1):
            self.sampling()
            if len(self.selected_edges) == 0:
                self.bottom_level = self.quotient_number                
                break
            self.make_quotient()
            self.db_fill()

            
    def extract_from_db(self,quotient_number, globe_number):
        """This function extracts information about the graphs upstairs and downstairs, 
        given the quotient number and globe number. In addition, it also picks up simplices 
        from a graph below it in the hierarchy, and computes the lift of each simplex.
        These lifts are fed to the simplicial search function Unpacker"""
        
        domain_graph = dgl.heterograph({('node', 'to', 'node'): ([], [])})
                
        self._connect_db()
        # The constraint contracted = False ensures that we don't pick up loops.
        self.cursor.execute(f'''SELECT e_src, e_dst
                            FROM edge_details
                            WHERE quotient_id = {quotient_number-1} 
                            AND contracted = False
                            AND globe_id = {globe_number}
                            ''')
        domain_graph_edges = self.cursor.fetchall()
        
        self.cursor.execute(f'''SELECT * FROM simplices
                            WHERE quotient_id = {quotient_number}
                            AND globe_id = {globe_number}
                            ''')
        simplex_details = self.cursor.fetchall()
        self.cursor.execute(f'''SELECT simplex_id FROM simplices
                            ''')
        all_simplex_ids = self.cursor.fetchall()
        self.cursor.execute(f'''SELECT e_src, e_dst
                            FROM edge_details
                            WHERE quotient_id = {quotient_number}
                            AND globe_id = {globe_number}
                            ''')
        range_graph_edges_db = self.cursor.fetchall()
        self._close_db()
        nodes_to_lift = set()
        simplices_to_lift = dict()
        if len(domain_graph_edges) == 0:
            return ('empty_graph',)
        
        deg_edges = list()
        range_graph_edges=list()
        for row in range_graph_edges_db:
            e_src        = row[0]
            e_dst        = row[1]
            nodes_to_lift = nodes_to_lift.union({e_src, e_dst})
            if e_src == e_dst:
                deg_edges = deg_edges + [[e_src, e_dst]]
            else:
                range_graph_edges.append([e_src,e_dst])
        
        nodes_to_lift = list(nodes_to_lift)
        domain_graph_nodes = set()
                
        for src, dst in domain_graph_edges:
            domain_graph.add_edges(src,dst)
            domain_graph_nodes=domain_graph_nodes.union({src,dst})
            
        prev_nodes = list()
        dimensions = list()

        for row in simplex_details:
            simplex_id, dimension, index_values = row[0], row[2], row[5]
            index_values = json.loads(index_values)
            simplices_to_lift[(dimension, simplex_id)] = index_values
            if dimension == 0:
                prev_nodes = prev_nodes + [row[4]]
            dimensions = dimensions + [dimension]
        simplex_index = max(all_simplex_ids, key=lambda x: x[0], default=(0,))[0] + 1
        
        """Node IDs are kept the same as their labels. Since a node may be present in different quotient, 
        the nodes for each quotient are not present in the database. This information is extracted when
        edges are constructed.."""
        
        """Construct a dictionary of pre-images of the quotient ----> quotient+1 map on nodes. That is,
        creates a mapping of node labels to their equivalence classes."""

        in_placeholders = ', '.join(['?' for _ in nodes_to_lift])
        self._connect_db()
        self.cursor.execute(f'''SELECT node_class_id, globe_id, nodes
                            FROM node_classes
                            WHERE node_class_id IN ({in_placeholders})
                            ''', tuple(nodes_to_lift))
        node_mapping_db = self.cursor.fetchall()
        
        self._close_db()
        
        node_mapping  = dict()
        
        for row in node_mapping_db:
            key      = row[0]
            globe_id = row[1] 
            values   = row[2]
            if type(values) == str:
                values   = json.loads(values)
            node_mapping.update({key: values})
            
        for node in list(domain_graph_nodes):
            node_mapping[node] = [node]
            
        return (domain_graph, node_mapping, range_graph_edges, 
                deg_edges, simplices_to_lift, simplex_index,domain_graph_edges)
        
    def simplices_of_quotient(self,quotient_number):
        
        for globe_id in range(self.max_globe_index+1):
            simplices = dict()
            returns = self.extract_from_db(quotient_number, globe_id)
            if returns[0] == 'empty_graph':
                # We have an empty graph, so we can move on to the next iteration
                continue
            simplex_finder = Unpacker(domain_graph=returns[0],node_mapping=returns[1],range_graph_edges=returns[2],
                                      deg_edges=returns[3],quotiented_simplices=returns[4],
                                      simplex_max_index=returns[5],globe_number=globe_id,domain_graph_edges=returns[6])
            if simplex_finder.local_max_dim == 1:
                # There's no simplices to be found, so we can move on to the next iteration
                continue
            simplex_finder.inductive_connecting()
            local_simplices = simplex_finder.lifted_simplices
            simplices.update(local_simplices)
            number_of_simplices = len(simplices.values())
            if number_of_simplices == 0:
                # No simplices have been found, so we can move on to the next iteration
                continue
            
            # Otherwise, we add the found simplices to the database
            
            data_to_insert = []
            for key, element in simplices.items():
                dimension  = key[0]
                simplex_id = key[1]
                globe_id   = key[2]

                data_row = [dimension, simplex_id, quotient_number-1, number_of_simplices,
                            globe_id] + [json.dumps(list(element))]
                data_to_insert.append(data_row)
            columns = ['dimension', 'simplex_id', 'quotient_id', 'number_of_simplices', 'globe_id', 'simplex_vertices']
            placeholders = ', '.join(columns)
            query = f'''INSERT INTO simplices ({placeholders})
                    VALUES ({', '.join(['?'] * len(columns))})
                    '''
            self._connect_db()
            self.cursor.executemany(query, data_to_insert)
            self._close_db()
        
    def simplicial_search(self):        
        starting = self.bottom_level
        for q_id in range(starting, 0, -1):
            self.simplices_of_quotient(q_id)
            
            
"""found from https://stackoverflow.com/questions/42069187/
create-a-list-of-unique-numbers-by-applying-transitive-closure"""
def relation(array):

    mapping = {}

    def parent(u):
        if mapping[u] == u:
            return u
        mapping[u] = parent(mapping[u])
        return mapping[u]

    for u, v in array:
        u = int(u)
        v = int(v)
        if u not in mapping:
            mapping[u] = u
        if v not in mapping:
            mapping[v] = v
        mapping[parent(u)] = parent(v)

    results = collections.defaultdict(set)
    

    for u in mapping.keys():
        results[parent(u)].add(u)

        
    return [x for x in results.values()]

            
def find_common_tensors(tensor_A,tensor_B):
    equal_pairs = torch.all(tensor_A[:, None, :] == tensor_B[None, :, :], dim=2)
    common_pair_indices = torch.nonzero(equal_pairs, as_tuple=False)
    return tensor_A[common_pair_indices[:, 0]]
    
    
class AdjGraphOriginal():
    
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
		
"""Testing implementation with random graphs"""

import matplotlib.pyplot as plt
from time import process_time
def generate_random_graph(num_nodes, num_edges):
    src_edges =[]
    dst_edges = []
    edges = []
    
    while len(edges) < num_edges:
        src_edge = random.randint(0, num_nodes)
        dst_edge = random.randint(0, num_nodes)
        if src_edge != dst_edge and (src_edge, dst_edge) not in edges:
            src_edges.append(src_edge)
            dst_edges.append(dst_edge)
            edges.append((src_edge, dst_edge))
            
    graph = dgl.heterograph({('paper', 'cites', 'paper'): (src_edges, dst_edges)})
    return graph, edges
	
def run_experiment(runs, number_of_nodes):
    sparsity_values = list(range(2*number_of_nodes,math.comb(number_of_nodes,2)+1))
    
    all_sparsity_runs_quotients = []
    all_sparsity_runs_dynamic = []
    ratio = 0.05

    for i in tqdm(range(runs)):
        execution_times_quotients = []
        execution_times_dynamic = []
        for num_edges in sparsity_values:
            database_name = 'random{}_{}_{}.db'.format(number_of_nodes, num_edges, i)
            file_path = 'random{}_path_{}_{}'.format(number_of_nodes, num_edges, i)
            elapsed_time = 0
            elapsed_time_dynamic = 0
            dgl_G, edges = generate_random_graph(number_of_nodes, num_edges)
            random_preprocessing = graph_towers(file_path=file_path,database_name=database_name, graph=dgl_G,ratio=ratio,bottom_level = 5, max_dimension = 20)
            if random_preprocessing.maximum_class_size == 1:
                execution_times_quotients = execution_times_quotients + [elapsed_time]
                pass
            else:
                start_time = process_time()
                random_preprocessing.create_table()
                random_preprocessing.create_towers()
                random_preprocessing.simplicial_search()
                elapsed_time = process_time() - start_time
                execution_times_quotients = execution_times_quotients + [elapsed_time]
                os.remove(database_name)
            heavy_duty=AdjGraphOriginal(dgl_G)
            start_time = process_time()
            for index in range(number_of_nodes):
                heavy_duty.connectivity_update()
                if heavy_duty.highest_dimension() == True:
                    break
            elapsed_time_dynamic = process_time() - start_time
            execution_times_dynamic = execution_times_dynamic + [elapsed_time_dynamic]
        all_sparsity_runs_quotients.append(execution_times_quotients)
        all_sparsity_runs_dynamic.append(execution_times_dynamic)
    all_sparsity_runs_quotients = np.array(all_sparsity_runs_quotients)
    all_sparsity_runs_dynamic = np.array(all_sparsity_runs_dynamic)
    averages_quotients = np.mean(all_sparsity_runs_quotients, axis=0)
    averages_dynamic = np.mean(all_sparsity_runs_dynamic, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_values, averages_quotients, 
             linestyle = 'dotted', label='Graph towers algorithm', color='red',marker='o')
    plt.plot(sparsity_values, averages_dynamic, 
             linestyle = 'dotted', label='Dynamic search', color='blue', marker='x')
    plt.xlabel('Number of edges')
    plt.ylabel('Average execution time')
    plt.title('Average execution time of {} runs of a random graph with {} nodes and ratio {}'.format(runs, number_of_nodes, ratio))
    plt.xticks(sparsity_values)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('average_execution_time_num_nodes_{}_ratio_{}_lower_columns.png'.format(number_of_nodes,ratio))


"""Testing implementation with random graphs"""
def run_experiment2(runs, density):
    assert density<=1 and density>=0, \
    'Keyword argument \"density\" of graph_towers\'s init method must be between 0 and 1.'
    nodes_list = range(2,50)
    ratio = 0.1

    all_sparsity_runs_quotients = []
    all_sparsity_runs_dynamic = []

    for i in tqdm(range(runs)):
        execution_times_quotients = []
        execution_times_dynamic = []
        for num_nodes in nodes_list:
            num_edges = math.ceil(density*num_nodes*(num_nodes-1))
            database_name = 'randomdensity{}_edges{}_nodes{}_run{}.db'.format(density, num_edges, num_nodes, i)
            file_path = 'randomdensity{}_edges{}_nodes{}_run{}'.format(density, num_edges, num_nodes, i)
            elapsed_time = 0
            elapsed_time_dynamic = 0
            dgl_G, edges = generate_random_graph(num_nodes, num_edges)
            random_preprocessing = graph_towers(file_path=file_path,database_name=database_name, graph=dgl_G,ratio=ratio,bottom_level = 5, max_dimension = 20)
            if random_preprocessing.maximum_class_size == 1:
                execution_times_quotients = execution_times_quotients + [elapsed_time]
                pass
            else:
                start_time = process_time()
                random_preprocessing.create_table()
                random_preprocessing.create_towers()
                random_preprocessing.simplicial_search()
                elapsed_time = process_time() - start_time
                execution_times_quotients = execution_times_quotients + [elapsed_time]
                os.remove(database_name)
            heavy_duty=AdjGraphOriginal(dgl_G)
            start_time = process_time()
            for index in range(num_nodes):
                heavy_duty.connectivity_update()
                if heavy_duty.highest_dimension() == True:
                    break
            elapsed_time_dynamic = process_time() - start_time
            execution_times_dynamic = execution_times_dynamic + [elapsed_time_dynamic]
        all_sparsity_runs_quotients.append(execution_times_quotients)
        all_sparsity_runs_dynamic.append(execution_times_dynamic)
    all_sparsity_runs_quotients = np.array(all_sparsity_runs_quotients)
    all_sparsity_runs_dynamic = np.array(all_sparsity_runs_dynamic)
    averages_quotients = np.mean(all_sparsity_runs_quotients, axis=0)
    averages_dynamic = np.mean(all_sparsity_runs_dynamic, axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(nodes_list, averages_quotients, 
             linestyle = 'dotted', label='Graph towers algorithm', color='red',marker='o')
    plt.plot(nodes_list, averages_dynamic, 
             linestyle = 'dotted', label='Dynamic search', color='blue', marker='x')
    plt.xlabel('Number of nodes')
    plt.ylabel('Average execution time')
    plt.title('Average execution time of {} runs of a random graph with density {} and ratio {}'.format(runs, density, ratio))
    plt.xticks(nodes_list)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('average_execution_time_density{}_maxnodes{}_ratio_{}_lower_columns.png'.format(density,max(nodes_list),ratio))