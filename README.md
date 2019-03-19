# stochastic_microgrid_formation
This repository contains Python scripts for the Stochastic microgrid formation herustic presented in "A Heuristic Approach to the Post-disturbance and Stochastic Pre-disturbance Microgrid Formation Problem" by Kwami S. A. Sedzro, Xin Shi, Alberto J. Lamadrid and Luis F. Zuluaga. Currently there are 7 files in the repository. 

1) "simulation_new.py" is used to generate potential scenarios according to the generation scheme described in the paper.
2) "MILP.py" is the python code for the MILP formulation
3) "herustic1.py" is the python code for the first step of the heuristic method.
4) "get_cluster.py" is the python code for the second step of the heuristic method. 
5) "herustic2.py" is the python code for the third step of the heuristic method.
6) "ieee30_scenarios.dat is the data file corresponding to the IEEE 30-bus system considered in the case study of the paper.
7) "ieee57_scenarios.dat is the data file corresponding to the IEEE 57-bus system considered in the case study of the paper.

The code is constructed and solved using the gurobipy module, thus in order to execute the script one must (i) Ensure that gurobi is installed on the computer they are using to run the code (ii) Install gurobipy:

https://www.gurobi.com/documentation/8.1/quickstart_mac/the_gurobi_python_interfac.html

Additionally, to run the code, execute the files in the following sequence: 

Step 0: (optional) Run "simulation_new.py" in order to generate the disaster scenario for a given data set

Step 1: Run "herustic1.py" in order to solve for the best locations for the Distributed Generators. 

Step 2: Make sure to record the solution (i.e. the generator locations), and use this to populate the vector "gen_location" on line 83 of "herustic2.py". Currently, the code is set up to solve the IEEE 57-bus system, where part 1 of the herustic locates the generators at buses 7, 9 and 11. Thus, before making any changes, one would find

          gen_location = [7, 9, 11]
on line 83 of "herustic2.py".

Step 3: Run "herustic2.py" to obtain the partitioning (microgrids) found by the herustic.
