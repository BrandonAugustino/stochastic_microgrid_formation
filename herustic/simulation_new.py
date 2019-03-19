# import sys
# sys.path.append('/usr/local/lib/python2.7/')


from sympy import binomial   
from gurobipy import *
import random
import copy
import operator

#############################    DATA LOAD    ##################################

file = open("..herustic/ieee57_s5_python.dat","r")    # read data for the optimization model
#dataforfullmodeltest1.dat
data = file.read().split('\n')                  # split the data accoding to \n, 
N = int(data[0].split()[2])                     # number of nodes
K = int(data[1].split()[2])                     # number of generators
BB = map(int, data[2].split()[2:])              # list of nodes in network
B = set(BB)                                     # set of nodes in network
L = binomial(N,2)                               # number of maximum possible link      
AAA = data[3].split()[2]                        # set of all the links 
AA = AAA.split(';')                             # split the link column into link pairs
AAfull = AA                                     # Denote the AAfull as full link in the network
# print 'test'
# print AA
A = set(AA)                                     # A is the set of all the full link

Alist = []                                      # Alist is a two dimensional array which contains adjacent nodes of each node               
for i in range(N):                              # get the adjacent nodes for each node
	Atemp = []
	for j in AA:
		if map(int,j.split(','))[0] == i+1:
			Atemp.append(map(int,j.split(','))[1])
	Alist.append(Atemp)

p = map(float, data[4].split()[2:])             # active load of each node
q = map(float, data[5].split()[2:])             # reactive load of each node
w = map(float, data[6].split()[2:])             # weight of each node
rtemp = map(float, data[7].split()[2:])         # line resistance on each link
r = [['NA' for i in range(N)] for j in range(N)]          # denote r as a N*N array whose elements are NA
for i in range(N):                                        # set r[i][j] equal to the corresponding resistance
	for j in range(N):
		for p1 in range(0,len(rtemp)/3):
			if (rtemp[3*p1] == i+1) and (rtemp[3*p1+1] == j+1):
				r[i][j] = rtemp[3*p1+2]

xtemp = map(float,data[8].split()[2:])          # line reactance on each link
x = [['NA' for i in range(N)] for j in range(N)]          # denote x as a N*N array whose elements are NA
for i in range(N):                                        # set x[i][j] equal to the corresponding reactance
	for j in range(N):
		for p1 in range(0,len(xtemp)/3):
			if (xtemp[3*p1] == i+1) and (xtemp[3*p1+1] == j+1):
				x[i][j] = xtemp[3*p1+2]

Vo = int(data[9].split()[2])                    # Vo nominal voltage magnitude
VR = int(data[10].split()[2])                   # VR rated voltage magnitude
epslison = float(data[11].split()[2])           # epslison: tolerance of the voltage difference 
Pmax = map(float, data[12].split()[2:])         # maximum active power generation of each generator
Qmax = map(float, data[13].split()[2:])         # maximum reactive power generation of each generator
TP = int(data[14].split()[2])                   # TP: transmission capacity of line of active power flow
TQ = int(data[15].split()[2])                   # TQ: transmission capacity of line of reactive power flow
S = int(data[16].split()[2])                    # number of scenarios

##################################################################################

###############   For cascading part   #################
Umax = 200                                      # power flow constraints for both active power and reactive power for cascading
TPL = 1000*TP                                   # relaxization of TP
TQL = 1000*TQ                                   # relaxization of TQ
h_h = 0.1
h_q = 0.1

##############    Generate the potential scenarios   ###############

simset = {}                                     # set of possible scenarios
simmark = {}                                    # set of corresponding freqency
# for i in range(len(AA)/2):                      # generate all the scenarios that two links were missing
#   AAtemp = AAA.split(';')
#   temp1 = AAA.split(';')[2*i]
#   temp2 = AAA.split(';')[2*i+1]
#   AAtemp.remove(temp1)
#   AAtemp.remove(temp2)
#   simset[str(i)] = AAtemp
#   simmark[str(i)] = 0

############     Choose one line to be broken randomly     ################
print('\n AA:')                                 # print the set of links before one of them is broken
print AA                                        
print len(AA)                                   # print number of links before broken happens

# Interations begins
for Iteration in range(100):                  
	AAA = data[3].split()[2]                      # set of all the links in scenario 1
	AA = AAA.split(';')                           # Take set of links from data again to make sure the correctness of the data
	A = set(AA)
	Alist = []
	for i in range(N):
		Atemp = []
		for j in AA:
			if map(int,j.split(','))[0] == i+1:
				Atemp.append(map(int,j.split(','))[1])
		Alist.append(Atemp)


	print('\n ')
	print('\nIteration begins:' + str(Iteration))     # Announment of the Iteration
	print('\n ')
	# linkremove1 = random.randint(0, len(AA)-1)        # choose one link from the set randomly       
	# if (linkremove1%2 == 0):                          # find the paired link
	#   linkremove2 = linkremove1 + 1
	# else:
	#   linkremove2 = linkremove1 - 1
	# AAnew = copy.deepcopy(AA)                         # deepcopy AAnew as the set of all links
	# linkmark1 = AA[linkremove1]                       # find the links
	# linkmark2 = AA[linkremove2]
	# AAnew.remove(AA[linkremove1])                     # remove the broken links from set of full links
	# AAnew.remove(AA[linkremove2])
	# AA = copy.deepcopy(AAnew)                         # Now AA is the set of rest links
	# A = set(AA)                                       # Now A is the set of rest links
	# Alist = []                                        # Alist is two-dimensional array contains adjacant nodes of each nodes for rest links
	# for i in range(N):                                # find the adjacant nodes for each node
	#   Atemp = []
	#   for j in AA:
	#     if map(int,j.split(','))[0] == i+1:
	#       Atemp.append(map(int,j.split(','))[1])
	#   Alist.append(Atemp)


	#########################   Decision Variables   #######################

	m = Model("Powersystembroken")                                                # name of the model
	thelb = [-200 for i in range(N*N)]                                            # lower bound for variable P and Q (powerflow)
	Vlb = [-500000 for i in range(N)]                                             # lower bound for variable V (voltage)
	s = m.addVars(N, vtype = GRB.BINARY, name = "load")                           # if the node is loaded or not
	z = m.addVars(K, N, vtype = GRB.BINARY, name = "location")                    # location of each generator
	b = m.addVars(N, N, vtype = GRB.BINARY, name = "link")                        # if the link is opened 
	Pg = m.addVars(N,vtype = GRB.CONTINUOUS, name = "activepowergen")             # the active power capacity of each generator
	P = m.addVars(N,N, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "activepower")  # active power on each link
	Qg = m.addVars(N,vtype = GRB.CONTINUOUS, name = "reactivepowergen")           # the reactive power capacity of each generator
	Q = m.addVars(N,N,lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "reactivepower") # reactive power on each link
	V = m.addVars(N, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "voltage")          # slack variable for a constaint
	delta = m.addVars(N,N, lb = -GRB.INFINITY,vtype = GRB.CONTINUOUS, name = "slack")

	#############################################################################################


	print('\n test for A before optimize:')                                       # print the set of links after one link is broken
	print A
	print len(A)


	#########################    Constaints    ###############################3

	##########  node P constraint   #######
	m.addConstrs(((quicksum(P[i-1,j-1] for j in Alist[i-1]) == -s[i-1]*p[i-1] + Pg[i-1]) for i in BB ),"nodeP")  

	##########  node Q constraint  ########
	m.addConstrs(((quicksum(Q[i-1,j-1] for j in Alist[i-1]) == -s[i-1]*q[i-1] + Qg[i-1]) for i in BB ),"nodeQ")  

	##########  power flow capcaity  CONSTRAINT  ##########
	for i in range(N):
		for j in range(N):
			if (','.join([str(i+1),str(j+1)]) in A):
				m.addConstr(-TP*b[i,j] <= P[i,j], 'lineP1'+str(i+1)+str(j+1))
				m.addConstr(P[i,j] <= TP*b[i,j], 'lineP2'+str(i+1)+str(j+1))
				m.addConstr(-TQ*b[i,j] <= Q[i,j], 'lineQ'+str(i+1)+str(j+1))
				m.addConstr(Q[i,j] <= TQ*b[i,j], 'lineQ2'+str(i+1)+str(j+1))
				m.addConstr(P[i,j] == - P[j,i])
				m.addConstr(Q[i,j] == - Q[j,i])
			else:
				m.addConstr(P[i,j] == 0)
				m.addConstr(Q[i,j] == 0)
				m.addConstr(b[i,j] == 0)

	######### capcacity of power generation for each generator  ###########
	m.addConstrs(0 <= Pg[j-1]for j in B)
	m.addConstrs(Pg[j-1] <= quicksum((z[i,j-1]*Pmax[i]) for i in range(K)) for j in B)  
	m.addConstrs(0 <= Qg[j-1]  for j in B)
	m.addConstrs(Qg[j-1] <= quicksum((z[i,j -1]*Qmax[i]) for i in range(K))  for j in B)  

	####################     CONSTRAINTS OF VOLTAGE     #######################
	for i in range(N):
		for j in range(N):
			if (','.join([str(i+1),str(j+1)]) in A):
				# print 'test'
				# print  str(i) + ',' + str(j)
				# print r[i][j]
				# print ','.join([str(i+1),str(j+1)]) in A
				m.addConstr(V[i] == V[j] + (r[i][j]*P[i,j] + x[i][j]*Q[i,j])/Vo + delta[i,j])
				m.addConstr((-1 + b[i,j])*Vo <= delta[i,j])
				m.addConstr( delta[i,j] <= (1 - b[i,j])*Vo)

	m.addConstrs(Vo*quicksum(z[i,j-1] for i in range(K)) <= V[j-1]  for j in BB )
	m.addConstrs(V[j-1] <= Vo  for j in BB )

	m.addConstrs( (1 - epslison)*VR <= V[j-1] for j in BB )
	m.addConstrs(  V[j-1] <= (1 + epslison)*VR  for j in BB )  # Can be eliminated

	############   LOGICAL CONSTRAINTS   ###################
	m.addConstrs(quicksum(z[i,j] for j in range(N)) == 1 for i in range(K))

	##############new 
	m.addConstrs(quicksum(z[i,j] for i in range(K)) <= 1 for j in range(N))


	############ location of fixed generators ##############
	# m.addConstr(z[0,2] == 1)
	m.addConstr(z[1,9] == 1)
	# m.addConstr(z[2,9] == 1)


	############     OBJECTIVE VALUE    ##############
	m.setObjective(quicksum(w[i-1]*s[i-1]*p[i-1] for i in BB), GRB.MAXIMIZE)


	##############    Parameters for the optimization model    #############
	m.Params.IntFeasTol = 1e-05
	m.Params.FeasibilityTol = 1e-06
	m.Params.OptimalityTol = 1e-06
	m.Params.MIPGap = 1e-02
	########################################################################

	# print('\nThe broken link is ')                              # print the broken links
	# print linkmark1
	# print linkmark2

	print('\n The links before optimization are')               # print the set of links just before optimization
	print AA
	print('\n')
	################################################  Result print ######################

	m.optimize()                                                # Do the optimization
	print('\nCost: %g' % m.objVal)                              # print the cost


	######################## Print the information of location of generators and if we load nodes  ##########333
	#for i in m.getVars()[0:147]:
	#  print('%s %g' % (i.varName, i.x))
	#############################################################################################

	brokenline = {}                                             # dictionary of potentially broken links: contain links and number of power flows
	mark = 0                                                    # number of broken links
	for i in m.getVars():
		if (i.varName[0:11] == 'activepower') and (i.varName[11] != 'g'):             # get the active power flow for each link
			if abs(i.x) >= 0*Umax:
				mark = mark + 1                                                           # number of links that are potentially broken
				strwork = map(int,i.varName[:-1].split('[')[1].split(','))                # get the name of this broken link
				stritem = str(strwork[0]+1) + ',' + str(strwork[1]+1)                     # get the name of this broken link
				brokenline[stritem] = i.x                                                 # set the corresponding value as the power flow
		if (i.varName[0:13] == 'reactivepower') and (i.varName[13] != 'g'):           # get the active power flow for each link
			if abs(i.x) >= 0*Umax:              
				mark = mark + 1                                                       
				strwork = map(int,i.varName[:-1].split('[')[1].split(','))                # get the name of this potentially broken link
				stritem = str(strwork[0]+1) + ',' + str(strwork[1]+1)                     # get the crresponding value as power flow
				brokenline[stritem] = i.x                                                 # get the reactive power flow for each link ####################problem !!!!!!!!!!!!!


	print('\nNumber of links that will be broken:')                                 # print number of links that will potentially be broken
	print mark
	print('\nThe set of links that are broken afterward:')                          # print set of links that are potentially broken afterward
	print brokenline
	# print('\nThe broken link before:')                                              # print the potentially broken link before
	# print linkmark1
	# print linkmark2

	################################   delete the optimization model  #########################
	
	del m

	#####################    Iteration begins(to find the balanced network)    ################

	markofiter = 0                                                                  # number of iterations for finding balanced network
	realbroken = {}                                                               # set of real broken links
	brokenprob = random.randint(0, 100)/100.0                                      # the probability that one line will break if the corresponding pro ir larger than this
	for i in brokenline:                                                          # link broken if the power flow is larger than the benm
		if h_h*abs(brokenline[i])/Umax >= brokenprob:                               # need to increase randomness
			realbroken[i] = brokenline[i]                                             # record read broken links in realbroken
	while (realbroken != {}):                                                       # stop the iteration if there is no potentially broken link
		#realbroken = {}                                                               # set of real broken links
		markofiter = markofiter + 1                                                   # number of iteration increases
		print('\nNew Interation Begins' + str(markofiter))                            # print the number of iterations
		# brokenprob = random.randint(0, 40)/100.0                                      # the probability that one line will break if the corresponding pro ir larger than this
		# for i in brokenline:                                                          # link broken if the power flow is larger than the benm
		#   if 0.3*abs(brokenline[i])/Umax >= brokenprob:                               # need to increase randomness
		#     realbroken[i] = brokenline[i]                                             # record read broken links in realbroken
		AAnew = copy.deepcopy(AA)                                                     # set AAnew as the set of links in the last iteration


		################## print to check #################
		print('\nprobability')                                                        # print the threshold (probability)
		print brokenprob
		print('\nbroken line:')                                                       # print the potential broken lines
		print brokenline
		print('\nrealbroken')                                                         # print the real lines that will be broken in this iteration
		print realbroken
		if (realbroken == {}):                                                        # if there is no broken line in this iteration, that means network is balanced, stop iteration
			break  

		print('\n test for A before optimize:')                                       # print the set of links after substracting broken lines of last iterations (before this iteration)
		print AAnew
		print len(AAnew)
		############################################

		for i in realbroken:                                                          # remove broken lines from last iteration
			AAnew.remove(i)
		AA = copy.deepcopy(AAnew)                                                     # set AA as the set of links after removing broken lines
		A = set(AA)                                                                   # set A as a set object
		Alist = []
		for i in range(N):                                                            # set Alist as adjacent nodes of each node
			Atemp = []
			#Aset = set()
			for j in AA:
				if map(int,j.split(','))[0] == i+1:
					Atemp.append(map(int,j.split(','))[1])
			#Aset = set(Atemp)
			Alist.append(Atemp)
		print('\n test for A before optimize after removing:')                        # print all the links after removing broken lines
		print A
		print len(A)

######################    Define the variables     #####################          # begin next optimization for the network with links been removed

		m = Model("iteration")                                                        # set name of model of this optimization
		thelb = [-200 for i in range(N*N)]                                            # lower bound of power flow
		Vlb = [-500000 for i in range(N)]                                             # lower bound of voltage
		s = m.addVars(N, vtype = GRB.BINARY, name = "load")
		z = m.addVars(K, N, vtype = GRB.BINARY, name = "location")
		b = m.addVars(N, N, vtype = GRB.BINARY, name = "link")
		Pg = m.addVars(N,vtype = GRB.CONTINUOUS, name = "activepowergen")
		P = m.addVars(N,N, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "activepower")
		Qg = m.addVars(N,vtype = GRB.CONTINUOUS, name = "reactivepowergen")
		Q = m.addVars(N,N,lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "reactivepower")
		V = m.addVars(N, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "voltage")
		delta = m.addVars(N,N, lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = "slack")

		######### node P constraint  ##############
		m.addConstrs(((quicksum(P[i-1,j-1] for j in Alist[i-1]) == -s[i-1]*p[i-1] + Pg[i-1]) for i in BB ),"nodeP")  

		########## node Q constraint  ##############
		m.addConstrs(((quicksum(Q[i-1,j-1] for j in Alist[i-1]) == -s[i-1]*q[i-1] + Qg[i-1]) for i in BB ),"nodeQ")  

		########## LINE CONSTRAINT   #############
		for i in range(N):
			for j in range(N):
				if (','.join([str(i+1),str(j+1)]) in A):
					m.addConstr(-TP*b[i,j] <= P[i,j], 'lineP1'+str(i+1)+str(j+1))
					m.addConstr(P[i,j] <= TP*b[i,j], 'lineP2'+str(i+1)+str(j+1))
					m.addConstr(-TQ*b[i,j] <= Q[i,j], 'lineQ'+str(i+1)+str(j+1))
					m.addConstr(Q[i,j] <= TQ*b[i,j], 'lineQ2'+str(i+1)+str(j+1))
					m.addConstr(P[i,j] == - P[j,i])
					m.addConstr(Q[i,j] == - Q[j,i])
				else:
					m.addConstr(P[i,j] == 0)
					m.addConstr(Q[i,j] == 0)
					m.addConstr(b[i,j] == 0)

		for j in range(N):
			if  (','.join([str(i+1),str(j+1)])  in realbroken):
				m.addConstr(b[i,j] == 0)
				m.addConstr(b[j,i] == 0)

		######### confinement of generators ##############
		m.addConstrs(0 <= Pg[j-1]for j in B)
		m.addConstrs(Pg[j-1] <= quicksum((z[i,j-1]*Pmax[i]) for i in range(K)) for j in B)  
		m.addConstrs(0 <= Qg[j-1]  for j in B)
		m.addConstrs(Qg[j-1] <= quicksum((z[i,j-1]*Qmax[i]) for i in range(K))  for j in B) 

		########## CONSTRAINTS OF VOLTAGE #############
		for i in range(N):
			for j in range(N):
				if (','.join([str(i+1),str(j+1)]) in A):
					m.addConstr(V[i] == V[j] + (r[i][j]*P[i,j] + x[i][j]*Q[i,j])/Vo + delta[i,j])
					m.addConstr((-1 + b[i,j])*Vo <= delta[i,j])
					m.addConstr( delta[i,j] <= (1 - b[i,j])*Vo)
		m.addConstrs(Vo*quicksum(z[i,j-1] for i in range(K)) <= V[j-1]  for j in BB )
		m.addConstrs(V[j-1] <= Vo  for j in BB )

					
		m.addConstrs( (1 - epslison)*VR <= V[j-1] for j in BB )
		m.addConstrs(  V[j-1] <= (1 + epslison)*VR  for j in BB )  # Can be eliminated

		############ LOGICAL CONSTRAINTS ################
		m.addConstrs(quicksum(z[i,j] for j in range(N)) == 1 for i in range(K))
		##############new 
		m.addConstrs(quicksum(z[i,j] for i in range(K)) <= 1 for j in range(N))

		############ location of fixed generators ##############
		# m.addConstr(z[0,2] == 1)
		m.addConstr(z[1,9] == 1)
		# m.addConstr(z[2,9] == 1)

		m.Params.IntFeasTol = 1e-05
		m.Params.FeasibilityTol = 1e-06
		m.Params.OptimalityTol = 1e-06
		m.Params.MIPGap = 1e-02

		############     OBJECTIVE VALUE    ##############
		m.setObjective(quicksum(w[i-1]*s[i-1]*p[i-1] for i in BB), GRB.MAXIMIZE)

		############# Do the optimization ###############
		m.optimize()

		print('\nCost: %g' % m.objVal)                                                      # print the objective value

		brokenline = {}                                                                     # brokenline is the set of all potential broken lines
		mark = 0                                                                            # mark is the number of potential broken lines
		for i in m.getVars():                                                               # find the potential broken lines
			if (i.varName[0:11] == 'activepower') and (i.varName[11] != 'g') :                # find broken lines caused by active power flow
				if abs(i.x) >= 0*Umax:
					mark = mark + 1                                                               # number of links that are broken
					strwork = map(int,i.varName[:-1].split('[')[1].split(','))                    # active power flow of the broken line 
					stritem = str(strwork[0]+1) + ',' + str(strwork[1]+1)                         # name of the broken line
					brokenline[stritem] = i.x                                                     # set the broken line
			if (i.varName[0:13] == 'reactivepower') and (i.varName[13] != 'g'):               # find the broken lines caused by reactive power flow
				if abs(i.x) >= 0*Umax:
					mark = mark + 1                                                               # number of potential broken lines
					strwork = map(int,i.varName[:-1].split('[')[1].split(','))                    # reactive power flow of the broken line 
					stritem = str(strwork[0]+1) + ',' + str(strwork[1]+1)                         # name of the broken line
					brokenline[stritem] = i.x                                                     # set the broken line          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!problem
		realbroken = {}                                                               # set of real broken links
		brokenprob = random.randint(0, 100)/100.0                                      # the probability that one line will break if the corresponding pro ir larger than this
		for i in brokenline:                                                          # link broken if the power flow is larger than the benm
			if h_q*abs(brokenline[i])/Umax >= brokenprob:                               # need to increase randomness
				realbroken[i] = brokenline[i]                                             # record read broken links in realbroken



		print('\nNumber of links that will be broken:')                                     # print the number of lines that will be broken
		print mark
		print('\nThe set of links that are broken afterward:')                              # print the links that have potential to be broken
		print brokenline
		del m                                                                               # delete the model for next iteration and modeling

	simsetadd = {}                                                                        # scenarios will be added in this iteration
	simmarkadd = {}                                                                       # frequency of the added scenario

	if AA in simset.values():                                                             # if AA showed before, just add 1 to the corresponding number
		for i in simset:
			if  (simset[i] == AA):
				simmark[i] = simmark[i] + 1
	else:                                                                                 # if AA did not show before, add this scenario to the whole set of scenarios
		simsetadd[str(len(simset))] = AA
		simmarkadd[str(len(simset))] = 1
		simmark.update(simmarkadd)
		simset.update(simsetadd)

	print('\nTest for AA')                                                                # print the final balanced network for this simulation
	print AA
	print len(AA)

#############################     print the result: scenarios and their corresponding frequency   #################

print('\n List of Scenarios(by broken lines):')                                         # print the all scenarios(by the broken lines)
for i in simset:
	if (simmark[i] >=1):
		print list(set(AAfull) - set(simset[i]))                                            # get the broken lines for this scenario
		print simmark[i]
		print('\n') 
print('\n Final Result:')                                                               # get the 2-frequency scenarios
for i in simset:
	if (simmark[i] >=2):
		print list(set(AAfull) - set(simset[i]))
		print simmark[i]
		print('\n')
simmark_sorted = sorted(simmark.items(), key=operator.itemgetter(1),reverse=True)       # sort the list of scenarios according to their frequency


##########################     prepare to write data into file result3.dat     ########################

f = open("result3.dat","w")                                                                                    
for i in simmark_sorted:                                                                # put the list of scenarios and corresponding frequency into file "result3.dat"(broken lines)
	if int(i[1])>0:
		print >>f , (set(AAfull) - set(simset[i[0]]))
		print >>f, i[1]
		print >>f, '\n'
f.close()

f = open("result3a.dat","w")                                                            # put the list of scenarios and corresponding frequency into file "result3a.dat"(all)
for i in simmark_sorted:
	if int(i[1])>0:
		tempa = simset[i[0]][0]
		for j in range(1,len(simset[i[0]])):
			tempa = tempa + ';' + simset[i[0]][j]
		print >>f , tempa
		print >>f, i[1]
		print >>f, '\n'
f.close()


f = open("result3b.dat","w")                                                            # put the list of scenarios and corresponding frequency into file "result3a.dat"(all)
for i in simmark_sorted:
	if int(i[1])>0:
		tempa = simset[i[0]][0]
		for j in range(1,len(simset[i[0]])):
			tempa = tempa + ';' + simset[i[0]][j]
		tempa = 'AAA' + ' ' + '=' + ' ' + tempa
		print >>f , tempa
		#print >>f, '\n'
print >>f, '\n'
for i in simmark_sorted:
	if int(i[1])>0:
		print >>f, i[1]
f.close()


print('\n Number of scenarios:')
print len(simset)

