# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:17:47 2018

@author: soumya
"""

import numpy as np
import csv
#import pydot
from graphviz import Digraph
dot = Digraph()

original_no_features = 0
required_no_features = 0
branch_bound = 0
#graph = pydot.Dot(graph_type = 'digraph')
opt_ans = ''


def incresing_function(require_feature):
        
    return sum(require_feature)

def generate_branch(features, require_feature,parent,label):
    global original_no_features, opt_ans, required_no_features, branch_bound    
    value = len(features)
    reduced_list = ' '.join(str(e) for e in features)
    cost = incresing_function(features)
    #node = pydot.Node(reduced_list)
    #edge = pydot.Edge(parent, reduced_list,label)
    #graph.add_edge(edge)
    dot.node(reduced_list,reduced_list + " " + str(cost))   
    dot.edge(parent,reduced_list,label)
    
    
    if cost <= branch_bound:
        return 
    if required_no_features == value:
        if cost > branch_bound:
            branch_bound = cost
            opt_ans = reduced_list
        else:
            branch_bound = branch_bound
        return
    no_of_branch = required_no_features + 1 - len(require_feature)
    f1 = features.copy()
    f2 = require_feature.copy()
    set_diff = np.setdiff1d(f1,f2)
    set_diff = np.array(set_diff)
    #print(len(set_diff))
    #print(no_of_branch)
    set_diff = np.random.choice(set_diff,no_of_branch,replace=False)
    branches = set_diff.tolist()
    branches.sort()
    w = 1
    for i in branches:
        feature = features.copy()
        feature.pop(feature.index(i))
        generate_branch(feature, branches[w: ] + require_feature, reduced_list,str(i))
        w += 1
    
    
    


def _branch_bound(features, require_feature):
    global original_no_features, required_no_features, branch_bound    
    no_of_child = required_no_features + 1
    _parent = np.array(features)
    pick_random = np.random.choice(_parent, no_of_child,replace = False)
    require_feature = pick_random.tolist()
    parent_node = ' '.join(str(e) for e in features)
    #node = pydot.Node(parent_node)
    dot.node(parent_node,parent_node + " " + str(incresing_function(features)))
    w  = 1
    list.sort(require_feature)
    for i in require_feature:
        feature = features.copy()
        feature.pop(feature.index(i))
        #reduced_list = ' '.join(str(e) for e in feature)
        #graph.node(cnode,cnode + " " + str(increasing_function(featureList)))        
        #edge  = pydot.Edge(parent_node,reduced_list)
        #graph.add_edge(edge)        
        generate_branch(feature, require_feature[w: ], parent_node,str(i))
        w += 1
    
    
    
if __name__ == "__main__":
    #features = [1,2,3,4,5,6,7,8]
    #features = [12,4,7,88,30,6]

    features = []

    with open('bb2.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            features = row

    #print(data)
    features = list(map(float, features))
    #print(data)
    require_feature = []
    original_no_features = len(features)
    required_no_features = 3
    _branch_bound(features, require_feature)
    #pydot.Node(opt_ans,style='filled', fillcolor='green')
    print(opt_ans)
    #graph.write_png('example1_graph.png')
    dot.render('outputs_bound',view=True)