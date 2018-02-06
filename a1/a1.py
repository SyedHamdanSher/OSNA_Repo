# coding: utf-8

# # CS579: Assignment 1
#
# In this assignment, we'll implement community detection and link prediction algorithms using Facebook "like" data.
#
# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.
#
# We'll cluster the resulting graph into communities, as well as recommend friends for Bill Gates.
#
# Complete the **15** methods below that are indicated by `TODO`. I've provided some sample output to help guide your implementation.


# You should not use any imports not listed here:
from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
<<<<<<< HEAD
  node2num_paths = defaultdict(int)
  node2num_paths[root] = 1
  node2distances = defaultdict(int)
  node2distances[root] = 0
  node2parents = defaultdict(list)
  def bfs_shortest_path(graph, start,node2num_paths,node2distances,node2parents,max_depth):
    explored = set()
    q=deque()
    explored.add(start)
    q.append(start)
 
    #if start == end:
      #explored.append(start)
    #return
        
    while q:
      # pop the first path from the queue
      node = q.popleft()
      for neighbor in graph.neighbors(node):
        if neighbor in explored and node2distances[neighbor] == (node2distances[node]+1):
          node2parents[neighbor].append(node)
          node2num_paths[neighbor] += 1
        elif neighbor not in explored and node2distances[node] < max_depth:
          q.append(neighbor)
          explored.add(neighbor)
          node2parents[neighbor].append(node)
          node2num_paths[neighbor] = 1
          node2distances[neighbor]=node2distances[node]+1  
      # in case there's no path between the 2 nodes
    return
  bfs_shortest_path(graph, root,node2num_paths,node2distances,node2parents,max_depth)
    
  return node2distances,node2num_paths,node2parents
 
=======
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from 
                       the root node to this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree

    In the doctests below, we first try with max_depth=5, then max_depth=2.

    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    ###TODO
    pass
>>>>>>> template/master


def complexity_of_bfs(V, E, K):

  #since every vertex and every edge will be explored in the worst case,The time complexity can be expressed as {\displaystyle O(|V|+|E|)}
  #and since we have used q = deque() q.append(start) where q.append(),newn.append(nn),q.append(newn), res.append(n1)and  q.popleft() have a time complexcity of O(1) each
  return (V+E+math.log(K))



def bottom_up(root, node2distances, node2num_paths, node2parents):
  '''flag=defaultdict(list)
    credits=defaultdict(list)
    DAG=defaultdict(list)
    flag1=[]
    flag2=[]
    #findings the leave nodes
    for K,V in node2parents.items():
        for x in V:
            flag1.append(x)
    
    for k,v in node2parents.items():
        if k not in flag1:
            flag2.append(k)
    for d in flag2:
        flag[d].append(1)
    #print(flag)
    
    for x,y in flag.items():
        for k,v in node2parents.items():
            if(x==k):
                for d in v:
                    if(len(v)>1):
                        if(k>d):
                            DAG[d,k].append(y[0]/len(v))
                        else:
                            DAG[k,d].append(y[0]/len(v))
                    else:
                        if(k>d):
                            DAG[d,k].append(y[0])
                        else:
                            DAG[k,d].append(y[0])
    #print(DAG)
    max=0
    for k,v in node2distances.items():
        if(max<v):
            max=v
    #print(max)
    for k,v in node2distances.items():
        if(k not in flag):
            flag[k].append(1)
    #print(flag)
    while(max!=0):
        
        for k1,v1 in node2distances.items():
            if(v1==max-1):
                for k,v in DAG.items():         
                    if(k1 in k):
                        #flag[k1].append(v[0])
                        flag[k1]=[flag[k1][0]+v[0]]
        for k2,v2 in node2distances.items():
            if(v2==max-1):
                for x4,y4 in flag.items():
                    for k4,v4 in node2parents.items():
                        if(x4==k4 and x4==k2):
                            for d4 in v4:
                                if(len(v4)>1):
                                    if(k4>d4):
                                        if((d4,k4))not in DAG.keys():
                                            DAG[d4,k4].append(y4[0]/len(v4))
                                    else:
                                        if((d4,k4))not in DAG.keys():
                                            DAG[k4,d4].append(y4[0]/len(v4))
                                else:
                                    if(k4>d4):
                                        if((d4,k4))not in DAG.keys():
                                            DAG[d4,k4].append(y4[0])
                                    else:
                                        if((d4,k4))not in DAG.keys():
                                            DAG[k4,d4].append(y4[0])
        max=max-1
    #print(flag)
    #print(DAG)
    for k,v in DAG.items():
        DAG[k]=v[0]'''
  flag=[]
  credits=defaultdict(int)
  nodes=defaultdict(int)
  ss=defaultdict(int)
    
  for k,v in node2distances.items():
    if k!=root:
      nodes[k]=1.0
    else:
      nodes[root]=0
    
  for k,v in sorted(node2distances.items(), key=lambda x: (-x[1])):
    if(node2num_paths[k]>1):
      n=1/node2num_paths[k]
      for nn in node2parents[k]:
        nodes[nn]=nodes[nn]+n
        nodes[k]=n  
    elif(node2num_paths[k]==1):
      for nn in node2parents[k]:
        nodes[nn]= nodes[nn]+nodes[k]
                
  for k,v in nodes.items(): 
    for nn in node2parents[k]:
      flag.append(k)
      flag.append(nn)
      flag = sorted(flag)
      credits[(flag[0],flag[1])]=nodes[k]
      flag.clear()
  return credits
    

def approximate_betweenness(graph, max_depth):
  nodelist=graph.nodes()
  BET=defaultdict(float)
  for node in nodelist:
    node2distances,node2num_paths,node2parents1=bfs(graph,node,max_depth)
    #print("Before bottom_up")
    credits=bottom_up(node, node2distances, node2num_paths, node2parents1)
    #print(node)
    for k,v in credits.items():
      BET[k[0],k[1]]=BET[k[0],k[1]]+(v)
  for k,v in BET.items():
    BET[k[0],k[1]]=v/2
  return BET


def is_approximation_always_right():
  return "no"


def partition_girvan_newman(graph, max_depth):

  graph1=graph.copy()
  #nodelist=graph1.nodes()
  components = [c for c in nx.connected_component_subgraphs(graph1)]
  #print (len(components))
  BET=approximate_betweenness(graph1, max_depth)
  #print(sorted(BET.items()),type(BET))
  #indent = '   ' * max_depth
  counter=0
  BET1=sorted(BET.items(),key=lambda tup:(-tup[1],tup[0]))
  if(max_depth!=1):
    #MAX=0
    while (len(components)==1):
      graph1.remove_edge(*(BET1[counter][0]))
      counter+=1
      components = [c for c in nx.connected_component_subgraphs(graph1)]
      '''graph1=graph.copy()
      for K,V in BET.items():
        if(MAX<V):
          MAX=V
          KK=K
        elif(MAX==V):
          if(KK>K):
            MAX=MAX
            KK==KK
          else:
            MAX=V
            KK=K
      print(indent + 'removing ' + str(KK))
      graph1.remove_edge(*KK)
      print (len(components))
      #BET = {key: value for key, value in BET.items() if key != KK}
      components = [c for c in nx.connected_component_subgraphs(graph1)]'''
    return components
  if(max_depth==1):
    i=0
    while(len(components)==1 and i<len(list(BET.keys()))):
      #graph2=graph.copy()
      graph1.remove_edge(*list(BET.keys())[i])
      #print(list(BET.keys())[i],"Removed")
      components = [c for c in nx.connected_component_subgraphs(graph1)]
      i=i+1
    return components

def get_subgraph(graph, min_degree):
  neighbors = set()
  degrees = nx.degree(graph)
  nodes=graph.nodes()
    
  for ni in nodes:
    neighbors |= set(graph.neighbors(ni))
  # plot at least the target node and his neighbors.
  result = set(nodes) | neighbors
  # add "friends of friends" up to n total nodes.
  LL=[]
  for x in neighbors:
    for k,v in degrees.items():
      if v>min_degree-1 and x==k:
        LL.append(x) 
    result=set(LL)
        
      #if len(result) > n:
        #break
  return graph.subgraph(result)




def volume(nodes, graph):
  counter=0
  for edge in graph.edges():
    if((edge[0] in nodes) or (edge[1] in nodes)):
      counter+=1
  return counter

def cut(S, T, graph):
  counter = 0
  for edge in graph.edges():
    if((edge[0] in T and edge[1] in S) or (edge[0] in S and edge[1] in T)):
      counter += 1
  return counter


def norm_cut(S, T, graph):
  NCV=((cut(S, T, graph)/volume(S, graph))+(cut(S, T, graph)/volume(T, graph)))
  return NCV


def score_max_depths(graph, max_depths):
  d=[]
  for x in max_depths:
    #print(x,"heereeeeeeeeeeeeee")
    components = partition_girvan_newman(graph, x)
    tu=tuple((x,norm_cut((list(set(components[0].nodes()))), (list(set(components[1].nodes()))), graph)))
    d.append(tu)
  return d


## Link prediction

# Next, we'll consider the link prediction problem. In particular,
# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):
  graph1=graph.copy()
  edges = sorted(graph1.edges([test_node]))
  #print(edges)
  # Sample edges to remove.
  ll=[]
  counter=0
  for edge in edges:
    if(counter<=n):
      graph1.remove_edge(*edge)
      counter+=1
    else:
      break
  return graph1



def jaccard(graph, node, k):
  neighbors = set(graph.neighbors(node))
  p=sorted(set(graph.edges()))
  #print(neighbors)
  #print(p)
  scores = []
  ll=[]
  for n in graph.nodes():
    if n != node and not graph.has_edge(node, n):
      neighbors2 = set(graph.neighbors(n))
      scores.append(((node,n), 1. * len(neighbors & neighbors2) / len(neighbors | neighbors2)))
  ll= (sorted(scores, key=lambda x: (-x[1],x[0])))
  counter=1
  scores.clear()
  for lll in ll:
    if(counter<=k):
      scores.append(lll)
      counter+=1
    else:
      break
  return scores

def similarity(graph,x,y,beta):
  l=nx.shortest_path_length(graph,source=x,target=y)
  p=list(nx.all_shortest_paths(graph,source=x,target=y))
  s=math.pow(beta,l)*len(p)
  return s


def path_score(graph, root, k, beta):
  '''def similarity(graph,x,y,beta):
    l=nx.shortest_path_length(graph,source=x,target=y)
    p=list(nx.all_shortest_paths(graph,source=x,target=y))
    s=math.pow(beta,l)*len(p)
  return s
    
  neighbors = set(graph.neighbors(root))
  p=sorted(set(graph.edges()))
  scores = []
  ll=[]
    
  for n in sorted(graph.nodes()):
    neighbors2 = set(graph.neighbors(n))
    if((root,n) not in  p and (n,root) not in p and root!=n):
      scores.append(((root,n), similarity(graph,root,n,0.5)))
  lll=sorted(scores, key=lambda x: x[1], reverse=True)
  for i in range(0,k):
    if(len(lll)>i):
      ll.append(lll[i])
  return ll'''
  scores = []
  ll=[]
  #print(math.inf,type(math.inf))
  for node in graph.nodes():
    if not graph.has_edge(root, node) and node != root:
      node2distances, node2num_paths, node2parents = bfs(graph,root,len(graph.nodes()))
      res = math.pow(beta,node2distances[node]) * node2num_paths[node]
      scores.append(((root, node), res))        
  ll= (sorted(scores, key=lambda x: (-x[1],x[0])))
  counter=1
  scores.clear()
  for lll in ll:
    if(counter<=k):
      scores.append(lll)
      counter+=1
    else:
      break
  return scores


def evaluate(predicted_edges, graph):
  edges=sorted(set(graph.edges()))
  c=0
  for x in predicted_edges:
    if x in edges:
      c=c+1
  l=len(predicted_edges)
  return (c/l)


"""
Next, we'll download a real dataset to see how our algorithm performs.
"""
def download_data():
    """
    Download the data. Done for you.
    """
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    """
    FYI: This takes ~10-15 seconds to run on my laptop.
    """
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' % evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' % evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
