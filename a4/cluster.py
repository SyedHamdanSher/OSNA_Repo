"""
cluster.py
"""

import networkx as nx
import pickle

from itertools import combinations
from networkx import edge_betweenness_centrality as betweenness

            
def readFile(filename):
    
    fp = open(filename, 'rb')
    tweets = pickle.load(fp)
    fp.close()
    return tweets
    
def jaccard(x,y):
    
    if len(x)>0 and len(y)>0:
        intersection = set.intersection(*[set(x), set(y)])
        union = set.union(*[set(x), set(y)])
        return len(intersection)/len(union)
    else:
        return 0
    
def creatGraphCluster(tweets):
	graph = nx.Graph()
	users = [t['user']['screen_name'] for t in tweets]
	#creating graph nodes
	combination = combinations(users,2)

	for u in users:
		graph.add_node(u)

	for c in combination:
		x=[]
		y=[]
		for t in tweets:
			if t['user']['screen_name'] == c[0]:
				x.append(t['user']['friends'])

			if t['user']['screen_name'] == c[1]:
				y.append(t['user']['friends'])
		coef = jaccard(x[0],y[0])

		if coef > 0.005:
			graph.add_edge(c[0],c[1],weight=coef)

	ll = []
	#remove nodes with degree less than or equal to 1
	for node,degree in graph.degree().items():
		if degree <= 1:
			ll.append(node)
	graph.remove_nodes_from(ll)

	return graph
    

def maxEdgeBetweenness(graph):
	value = betweenness(graph, weight='weight')
	return max(value, key=value.get)

def findCommunities(graph):
    components = [c for c in nx.connected_component_subgraphs(graph)]
    
    while len(components) < 8:
        edge_to_remove = maxEdgeBetweenness(graph)
        graph.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(graph)]
    return components
    
def writeToFile(fname,components):
    output = open(fname, 'wb')
    pickle.dump(components, output)
    output.close()

def main():
    filename='data.pkl'

    tweets=readFile(filename)
    
    graph=creatGraphCluster(tweets)
    
    components=findCommunities(graph)
    
    filename1='communities.pkl'
    writeToFile(filename1,components)
    

if __name__ == '__main__':
    main()
