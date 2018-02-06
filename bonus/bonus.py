import networkx as nx

def jaccard_wt(graph, node):
	scores = []
	neighbors1 = set(graph.neighbors(node))
	for n in graph.nodes():
		if n != node and not graph.has_edge(node, n):
			neighbors2 = set(graph.neighbors(n))

			numerator=0.0
			for i in neighbors1:
				for j in neighbors2:
					if i == j:
						numerator = numerator + (1.0/float(graph.degree(i)))

			denominator1=0.0
			for i in neighbors1:
				denominator1 = denominator1+graph.degree(i)
			denominator1 = 1.0/float(denominator1)
            
			denominator2=0.0
			for j in neighbors2:
				denominator2 = denominator2+graph.degree(j)
			denominator2 = 1.0/float(denominator2)
            
			score = numerator/(denominator1 + denominator2)
			scores.append(((node, n), score))
	ll= (sorted(scores, key=lambda x: (-x[1],x[0])))
	return ll


