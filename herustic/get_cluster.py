# cluster for IEEE-39 SYSTEM
# Author: Xin Shi

# import dijkstra1
import numpy as np
import networkx as nx
import matplotlib as plt
from networkx.drawing.nx_agraph import to_agraph
from numpy import inf


#################3
# N: number of nodes
# link: list of edges
# gen_location: list of locations of generators

def get_cluster(N, link,gen_location):
	g=nx.Graph()
	# print len(link)
	n = 0
	for i in link:
		if n%2 ==0:
			a = int(i.split(',')[0]) -1 
			b = int(i.split(',')[1]) -1
			g.add_edge(str(a),str(b))
		n = n + 1

	for i in gen_location:
		g.add_node(str(i))
	# print nx.node_connected_component(g, str(2))
	# print nx.node_connected_component(g, str(9))	
	# location of generators
	
	#matrix for hops
	A = np.zeros([len(gen_location),N])

	# find the hops
	p = 0
	for i in gen_location:
		for j in range(0,N):
			if str(j) in g.nodes() and i != j:
				# print i 
				# print j
				if nx.node_connected_component(g, str(i)) == nx.node_connected_component(g, str(j)):
					A[p,j] = nx.all_pairs_shortest_path_length(g)[str(i)][str(j)]
					# print 'laile'
				else:
					A[p,j] = inf
			else:
				A[p,j] = inf
			if i == j:
				A[p,j] = 0
		p = p + 1

	# do the cluster
	bus_distribution = np.zeros(N)
	#cluster
	cluster_ =  np.argmin(A, axis=0)

	print A
	print cluster_	

	nx.draw_networkx(g)
	G_draw = to_agraph(g)
	G_draw.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
	G_draw.draw('test.png')
	#nx.draw(g)


	# delete links among clusters
	delete_link = ''
	for i in range(N):
		if str(i) in g.nodes():
			for j in g.adj[str(i)].keys():
				if cluster_[i] != cluster_[int(j)]:
					# print 'really!'+ str(i) + ',' + str(j)
					delete_link = delete_link+  str(int(i)+1) + ',' + str(int(j)+1)+';' + str(int(j)+1) + ',' + str(int(i)+1) + ';'
					g.remove_edge(str(i),str(j))
	print 'delete_link'
	print delete_link
	# G_draw = to_agraph(g)
	# G_draw.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
	# G_draw.draw('test.png')
	#######all the link
	new_edges = []
	for i in g.edges():
		a = str(int(i[0]) + 1)
		b = str(int(i[1]) + 1)
		new_edges.append(str(a)+','+str(b))
		new_edges.append(str(b)+','+str(a))
	# print 'the new link:'
	# print new_edges
	# print len(new_edges.split(';'))
	return new_edges






def get_cluster_v1(N, link,gen_location):
	g=nx.Graph()
	# print len(link)
	n = 0
	for i in link:
		if n%2 ==0:
			a = int(i.split(',')[0]) -1 
			b = int(i.split(',')[1]) -1
			g.add_edge(str(a),str(b))
		n = n + 1

	for i in gen_location:
		g.add_node(str(i))
	# print nx.node_connected_component(g, str(2))
	# print nx.node_connected_component(g, str(9))	
	# location of generators
	
	#matrix for hops
	A = np.zeros([len(gen_location),N])

	# find the hops
	p = 0
	for i in gen_location:
		for j in range(N):
			if str(j) in g.nodes() and i != j:
				# print i 
				# print j
				if nx.node_connected_component(g, str(i)) == nx.node_connected_component(g, str(j)):
					A[p,j] = nx.all_pairs_shortest_path_length(g)[str(i)][str(j)]
					# print 'laile'
				else:
					A[p,j] = inf
			if i == j:
				A[p,j] = 0
		p = p + 1

	# do the cluster
	bus_distribution = np.zeros(N)
	#cluster
	cluster_ =  np.argmin(A, axis=0)

	print A
	print cluster_	

	A_cluster = []
	for i in range(len(gen_location)):
		A_cluster.append([j for j,jx in enumerate(cluster_) if jx == i])
	print 'test'
	print A_cluster


	nx.draw_networkx(g)
	G_draw = to_agraph(g)
	G_draw.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
	G_draw.draw('test.png')
	#nx.draw(g)


	# delete links among clusters
	delete_link = ''
	for i in range(N):
		if str(i) in g.nodes():
			for j in g.adj[str(i)].keys():
				if cluster_[i] != cluster_[int(j)]:
					# print 'really!'+ str(i) + ',' + str(j)
					delete_link = delete_link+  str(int(i)+1) + ',' + str(int(j)+1)+';' + str(int(j)+1) + ',' + str(int(i)+1) + ';'
					g.remove_edge(str(i),str(j))
	print 'delete_link'
	print delete_link
	# G_draw = to_agraph(g)
	# G_draw.layout('dot', args='-Nfontsize=10 -Nwidth=".2" -Nheight=".2" -Nmargin=0 -Gfontsize=8')
	# G_draw.draw('test.png')
	#######all the link
	new_edges = []
	for i in g.edges():
		a = str(int(i[0]) + 1)
		b = str(int(i[1]) + 1)
		new_edges.append(str(a)+','+str(b))
		new_edges.append(str(b)+','+str(a))
	# print 'the new link:'
	# print new_edges
	# print len(new_edges.split(';'))
	return new_edges, A_cluster







# file = open("ieee39_scenarios.dat","r")
# data = file.read().split('\n')                  

# gen_location = [2,24,3]

# N = 39

# S = int(data[3].split()[2])  # number of secnarios
# g=nx.Graph()
# Alist = []
# A = []
# for i in range(4,S+4):
# 	AAA1 = data[i].split()[2]     # set of all the links in scenario i
# 	AA1 = AAA1.split(';')
# 	A1 = set(AA1)
# 	Alist1 = []
# 	for k in range(N):
# 		Atemp = []	
# 		for j in AA1:
# 			if map(int,j.split(','))[0] == k+1:
# 				Atemp.append(map(int,j.split(','))[1])
# 		Alist1.append(Atemp)
# 	Alist.append(Alist1)
# 	A.append(AA1)

# ABB = A
# 	# 
# link = ABB[0]

# result = get_cluster(39,link,gen_location)
# print result

