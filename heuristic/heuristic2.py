from sympy import binomial
from gurobipy import *
import copy
import get_cluster
m = Model("thetamodelfull")

#######    DATA LOAD    ###############
file = open("../herustic/ieee57_scenarios.dat","r")
data = file.read().split('\n')
N = int(data[0].split()[2])                     # number of nodes
K = int(data[1].split()[2])                     # number of generators
BB = map(int, data[2].split()[2:])              # list of nodes in network
B = set(BB)                                     # set of nodes in network
L = binomial(N,2)                               # number of possible links
S = int(data[3].split()[2])                     # number of secnarios
S = S

Alist = []
A = []
for i in range(4,S+4):
#for i in range(4,5):
	AAA1 = data[i].split()[2]                    # set of all the links in scenario i
	AA1 = AAA1.split(';')
	A1 = set(AA1)
	Alist1 = []
	for k in range(N):
		Atemp = []	
		for j in AA1:
			if map(int,j.split(','))[0] == k+1:
				Atemp.append(map(int,j.split(','))[1])
		Alist1.append(Atemp)
	Alist.append(Alist1)
	A.append(AA1)


p = map(float, data[S+4].split()[2:])            # active load of each node
q = map(float, data[S+5].split()[2:])            # reactive load of each node
w = map(float, data[S+6].split()[2:])            # reactive load of each node
rtemp = map(float, data[S+7].split()[2:])        # line resistance on each link
r = [['NA' for i in range(N)] for j in range(N)] 
for i in range(N):
  for j in range(N):
    for p1 in range(0,len(rtemp)/3):
      if (rtemp[3*p1] == i+1) and (rtemp[3*p1+1] == j+1):
        r[i][j] = rtemp[3*p1+2]

xtemp = map(float,data[S+8].split()[2:])     # line resistance on each link
x = [['NA' for i in range(N)] for j in range(N)] 
for i in range(N):
  for j in range(N):
    for p1 in range(0,len(xtemp)/3):
      if (xtemp[3*p1] == i+1) and (xtemp[3*p1+1] == j+1):
        x[i][j] = xtemp[3*p1+2]

Vo = int(data[S+9].split()[2])                # Vo
VR = int(data[S+10].split()[2])               # VR
epslison = float(data[S+11].split()[2])       # epslison
Pmax = map(float, data[S+12].split()[2:])     # maximum generation of each generator
Qmax = map(float, data[S+13].split()[2:])     # maximum generation of each generator
TP = int(data[S+14].split()[2])               # TP
TQ = int(data[S+15].split()[2])               # TQ

prob = []
for i in range(S+16,S+S+16):
	prob.append(float(data[i]))

proball = sum(prob)

dd = map(float, data[S+S+16].split()[2:])     # reactive load of each node

genlocdata = data[S+S+17].split()[2].split(';')
genloc = {}
for item in genlocdata:
	itemtemp = item.split(',')
	genloc[int(itemtemp[0])-1] = int(itemtemp[1])-1

# dd = [1 for i in range(N)]  
hp = 0.1
#S = 1


####################different microgrids for different scenarios########################
AAAA_test = copy.deepcopy(A)

gen_location = [7,9,11]

for i in range(S):
	A[i] = get_cluster.get_cluster(N,A[i],gen_location)


Alist = []
for d in range(S):
	Alist1 = []
	for i in range(N):
		Atemp = []
		for j in range(N):
			if  (','.join([str(i+1),str(j+1)]) in A[d]):
				Atemp.append(j+1)
		Alist1.append(Atemp)
	Alist.append(Alist1)

############################################



#prob = [427, 414, 410, 400, 396,       388, 387, 385, 384, 381,     376, 372, 353, 323, 276,    266, 263, 246, 242, 241,      226, 198, 169, 158, 155,     147, 146, 140, 130, 128,      126, 125, 124, 124, 123,    122,  122 ,119, 116, 110, 95,   94, 73]
##########   Decision Variables   #############


s = m.addVars(N,S, vtype=GRB.BINARY,name = "load")
z = m.addVars(K,N,vtype = GRB.BINARY, name = "location")
b = m.addVars(N,N,S,vtype = GRB.BINARY,name = "link")
Pg = m.addVars(N,S,vtype = GRB.CONTINUOUS, name = "activepowergen")
P = m.addVars(N,N,S, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "activepower")
Qg = m.addVars(N,S,vtype = GRB.CONTINUOUS, name = "reactivepowergen")
Q = m.addVars(N,N,S,lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "reactivepower")
V = m.addVars(N,S, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "voltage")
delta = m.addVars(N,N,S, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "slack")

###########################################################################################
###########################################################################################
############for microgrid
# v = m.addVars(N,K, vtype=GRB.BINARY,name = "node_to_microgrid")
# c = m.addVars(N,N,K, vtype=GRB.BINARY,name = "link_to_microgrid")



############for microgrid(2)
#v = m.addVars(N,K,S, vtype=GRB.BINARY,name = "node_to_microgrid")
#c = m.addVars(N,N,K,S, vtype=GRB.BINARY,name = "link_to_microgrid")

###########################################################################################
###########################################################################################
###############################For dispatchable 

pd = m.addVars(N,S,vtype = GRB.CONTINUOUS, name = "dispatch")
ps = m.addVars(N,S,vtype = GRB.CONTINUOUS, name = "servedload")
qs = m.addVars(N,S,vtype = GRB.CONTINUOUS, name = "servedloadq")

###########################################################################




############  Constraints     #############
######### node P constraint
m.addConstrs(((quicksum(P[i-1,j-1,l] for j in Alist[l][i-1]) == - ps[i-1,l] + Pg[i-1,l]) for i in BB for l in range(S) ),"nodeP")  
######### node Q constraint
m.addConstrs(((quicksum(Q[i-1,j-1,l] for j in Alist[l][i-1]) == - qs[i-1,l] +  Qg[i-1,l]) for i in BB for l in range(S) ),"nodeQ")  

######### LINE CONSTRAINT  ############# 
for d in range(S):
  for i in range(N):
	 for j in range(N):
		  if (','.join([str(i+1),str(j+1)]) in A[d]):
			 m.addConstr(-TP*b[i,j,d] <= P[i,j,d], 'lineP1'+str(i+1)+str(j+1))
			 m.addConstr(P[i,j,d] <= TP*b[i,j,d], 'lineP2'+str(i+1)+str(j+1))
			 m.addConstr(-TQ*b[i,j,d] <= Q[i,j,d], 'lineQ'+str(i+1)+str(j+1))
			 m.addConstr(Q[i,j,d] <= TQ*b[i,j,d], 'lineQ2'+str(i+1)+str(j+1))
			 m.addConstr(P[i,j,d] == - P[j,i,d],)
			 m.addConstr(Q[i,j,d] == - Q[j,i,d])
		  else:
			 m.addConstr(P[i,j,d] == 0)
			 m.addConstr(Q[i,j,d] == 0)

######### confinement of generators
m.addConstrs(0 <= Pg[j-1,d]for j in B for d in range(S))
m.addConstrs(Pg[j-1,d] <= quicksum((z[i,j-1]*Pmax[i]) for i in range(K)) for j in BB for d in range(S))
m.addConstrs(0 <= Qg[j-1,d]  for j in B for d in range(S))
m.addConstrs(Qg[j-1,d] <= quicksum((z[i,j-1]*Qmax[i]) for i in range(K))  for j in BB for d in range(S))

########### CONSTRAINTS OF VOLTAGE
for d in range(S):
  for i in range(N):
	 for j in range(N):
		  if (','.join([str(i+1),str(j+1)]) in A[d]):
			 m.addConstr(V[i,d] == V[j,d] + (r[i][j]*P[i,j,d] + x[i][j]*Q[i,j,d])/Vo + delta[i,j,d])
			 m.addConstr((-1 + b[i,j,d])*Vo <= delta[i,j,d])
			 m.addConstr( delta[i,j,d] <= (1 - b[i,j,d])*Vo)
###############################################3

m.addConstrs(Vo*quicksum(z[i,j-1] for i in range(K)) <= V[j-1,d]  for j in BB for d in range(S))
m.addConstrs( V[j-1,d] <= Vo for j in BB for d in range(S))

# m.addConstrs(Vo*z[1,i-1] <= V[i-1,d]  for i in BB for d in range(S))
# m.addConstrs(V[i-1,d] <= Vo + (1 - z[1,i-1])*((1 + epslison)*VR - Vo)  for i in BB for d in range(S))
###############################################3
m.addConstrs( (1 - epslison)*VR <= V[j-1,d] for j in BB for d in range(S))
m.addConstrs(  V[j-1,d] <= (1 + epslison)*VR  for j in BB for d in range(S))

############ LOGICAL CONSTRAINTS
m.addConstrs(quicksum(z[i,j] for j in range(N)) == 1 for i in range(K))

##############new 
m.addConstrs(quicksum(z[i,j] for i in range(K)) <= 1 for j in range(N))



# m.addConstr(z[0,11] ==  1)
# m.addConstr(z[1,9] == 1)
# m.addConstr(z[2,9] == 1)
# m.addConstr(z[3,11] == 1)
#m.addConstr(z[4,10] == 1)

m.addConstr(z[0,7] == 1)
m.addConstr(z[1,9] == 1)
m.addConstr(z[2,11] == 1)

###########################################################################################
###########################################################################################
############for microgrid

# m.addConstrs(quicksum(v[i,k] for k in range(K)) == 1 for i in range(N))
# m.addConstrs(v[i,k] >= z[k,i] for i in range(N) for k in range(K))

# for i in range(N): 
# 	for j in range(N):
# 		m.addConstrs(c[i,j,k] <= v[i,k] for k in range(K))
# 		m.addConstrs(c[i,j,k] <= v[j,k] for k in range(K))
# 		m.addConstrs(c[i,j,k] >= v[i,k] + v[j,k] - 1 for k in range(K))
# 		if (','.join([str(i+1),str(j+1)]) in A[d]):
# 			 m.addConstrs(b[i,j,d] == quicksum(c[i,j,k] for k in range(K)) for d in range(S))
# #########################################


############for microgrid(2)

#m.addConstrs(quicksum(v[i,k,d] for k in range(K)) == 1 for i in range(N) for d in range(S))
#m.addConstrs(v[i,k,d] >= z[k,i] for i in range(N) for k in range(K) for d in range(S))
#
#for d in range(S):
#	for i in range(N): 
#		for j in range(N):
#			  if (','.join([str(i+1),str(j+1)]) in A[d]):
#				 m.addConstrs(c[i,j,k,d] <= v[i,k,d] for k in range(K))
#				 m.addConstrs(c[i,j,k,d] <= v[j,k,d] for k in range(K))
#				 m.addConstrs(c[i,j,k,d] >= v[i,k,d] + v[j,k,d] - 1 for k in range(K))
#				 m.addConstr(b[i,j,d] == quicksum(c[i,j,k,d] for k in range(K)))
#########################################

###############################For dispatchable 

m.addConstrs(dd[i]*hp*p[i] + (1 - dd[i])*p[i] <= pd[i,d] <= p[i] for i in range(N) for d in range(S))

m.addConstrs(ps[i,d] <= s[i,d]*p[i] for i in range(N) for d in range(S))

m.addConstrs(0 <= ps[i,d] <= pd[i,d] for i in range(N) for d in range(S))

m.addConstrs(ps[i,d] >= pd[i,d] - (1- s[i,d])*p[i] for i in range(N) for d in range(S))

for i in range(N):
	if p[i] != 0:
		m.addConstrs(qs[i,d] == q[i]/p[i]*ps[i,d] for d in range(S))


###########################################################################


###########################################################################################
###########################################################################################





############     OBJECTIVE VALUE    ##############
m.setObjective(quicksum ( quicksum(prob[j]/proball*w[i-1]*ps[i-1,j] for i in BB) for j in range(S)) , GRB.MAXIMIZE)
#m.setObjective(quicksum ( quicksum(w[i-1]*ps[i-1,j] for i in BB) for j in range(S)) , GRB.MAXIMIZE)
# m.setObjective(quicksum ( w[i-1]*ps[i-1,0]for i in BB ) , GRB.MAXIMIZE)

#m.setObjective(w[0]*s[0]*p[0] , GRB.MAXIMIZE)

m.Params.IntFeasTol = 1e-05
m.Params.FeasibilityTol = 1e-06
m.Params.OptimalityTol = 1e-06
m.Params.MIPGap = 1e-02

################################################

m.optimize()
print('\nCost: %g' % m.objVal)


served = [0]*S

for i in m.getVars():
	#print('%s %g' % (i.varName, i.x))
	if i.varName.split('[')[0] == 'location':
		if int(i.x) == 1:
			print('%s %g' % (i.varName, i.x))
	# if i.varName.split('[')[0] == 'activepowergen':
	# 	# if int(i.x) > 0:
	# 	print('%s %g' % (i.varName, i.x))
	if i.varName.split('[')[0] == 'activepowergen':
		if int(i.x) > 0:
			for j in range(S):
				if i.varName.split(',')[1].split(']')[0] == str(j):
					served[j] = i.x + served[j]
			# print('%s %g' % (i.varName, i.x))
all_served = 0
for i in range(S):
	all_served = all_served + served[i]*prob[i]

print 'used generation:'
print all_served/proball



# plot the histogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 10})

histplot = []

location = 0
for i in prob:
    for j in range(int(i)):
        histplot.append(served[location])
    location = location + 1
plt.yscale('log')
plt.xlabel('Total loads served')
plt.ylabel('Frequency')


plt.hist(np.array(histplot),20)


