# Capstone-Project-2
This is an Optimization problem which made use of knowledge in Essential Mathematics for Data Science and Libraries in Python to solve this problem.

CAPSTONE PROJECT 1
OPTIMIZATION PROBLEM
¶
AVOIDING THE POLICE
Introduction:
Project Description
This project is about a new Taxi Driver in Takoradi who is determined to use the shortest route on every trip. Trips to and around Dadzie street, a town in Takoradi has the most financial returns. Unfortunatel, the shortest route as mentioned above is heavily patrolled by the police, and with the fines and bribes charged for simply driving through, the shortest route may not be the best choice. Hence, the aim here is to find the route that maximizes the probability of not being stopped by the police. The figure below shows the possible routes from the market circle (1) to Dadzie street (7) and the associated probabilities of not being stopped on each segment.

avoiding%20the%20police%202.PNG

Problem:
Finding the Optimal solution that helps the Driver decide or choose a route that maximizes the probability of not being stopped by the police, thereby avoiding any fines and bribes that comes with that.

Goal:
Selecting the route that **Maximizes** the Probability of not being fined by the Police. 
Problem Solution Steps:
The steps below were used to find the optimal routes with the highest probability.

Step 1. Import all necessary Libraries.
The first step is to import relevant components of the Dijkstra Algorithm.

#importing all libraries to be used

import dijkstra as ds
import numpy as np
from dijkstra import Graph
from dijkstra import DijkstraSPF
Step 2. Converting all Street (nodes) numbers from 1-7 to A-G
A, B, C, D, E, F, G = nodes = list("ABCDEFG")

graph = Graph()
graph.add_edge(A, B, -np.log10(0.2))
graph.add_edge(A, C, -np.log10(0.9))
graph.add_edge(B, C, -np.log10(0.6))
graph.add_edge(B, D, -np.log10(0.8))
graph.add_edge(C, D, -np.log10(0.1))
graph.add_edge(C, E, -np.log10(0.3))
graph.add_edge(D, E, -np.log10(0.4))
graph.add_edge(D, F, -np.log10(0.35))
graph.add_edge(E, G, -np.log10(0.25))
graph.add_edge(F, G, -np.log10(0.5))

dijkstra = DijkstraSPF(graph, A)
Step 3. Printing out each Label and Distance
#Printing the labels anad their corresponding distance

print("%-5s %-5s" % ("label", "distance"))
for u in nodes:
    print("%-5s %8d" % (u, dijkstra.get_distance(u)))
label distance
A            0
B            0
C            0
D            0
E            0
F            1
G            1
Step 4. Extracting the path From A to G, the path is:
print(" -> ".join(dijkstra.get_path(G)))
A -> C -> E -> G




SAFETY AT SEKYERE
¶
Introduction:
Project Description
The District Chief Executive (DCE) of Sekyere East wishes to promote safety in his community by installing Emergency Telephones at selected locations. The DCE wishes to maximize the effectiveness of the Telephones by placing them at the intersections of the streets. There are Eleven (11) main streets in all in the community as shown below :

sekyere%20East.PNG

Problem:
Finding the Optimal solution required to meet the DCE's ambition of promoting safety in the community whiles cutting down cost but ensuring that a single Telephone is able to serve at least two(2) streets.

Goal:
Finding the minimum number of telephones that can serve each of the community’s main streets. 
Mathematical Formulation of Problem :
Objective Function/Model:
Minimize Z = X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8

where Xi's are the Telephones to be installed at the eight(8) streets labelled 1-8.

Inequality Constraints:
The constraints of the problem require installing at least one Telephone on each of the 11 streets (A to K) becomes:

X1 + X2 >= 1 ............STREET A
X2 + X3 >= 1 ............STREET B
X4 + X5 >= 1 ............STREET C
X7 + X8 >= 1 ............STREET D
X6 + X7 >= 1 ............STREET E
X2 + X6 >= 1 ............STREET F
X1 + X6 >= 1 ............STREET G
X4 + X7 >= 1 ............STREET H
X2 + X4 >= 1 ............STREET I
X5 + X8 >= 1 ............STREET J
X3 + X5 >= 1 ............STREET K

Xi's = (0,1), i = 1,2,... 8

Problem Solution Steps:
The steps below were used to find the optimal solution to the problem.

Step 1. Import Pyomo.
The first step is to import relevant components of the Pyomo Library.

# Importing Pyomo - model for solving optimization problem 

from pyomo.environ import *
Step 2. Create the Model Object
Pyomo provides two distinct methods to create models. Here I use ConcreteModel() to create a model instance.

# creating the first model object for storing the model insntance. 

model = ConcreteModel()
Step 3. Add the Decision Variables, Objective Function and Constraints to the Model Object.
The first major component of a Pyomo model are the decision variables which are added as fields to the model. In this case I use model.x1, model.x2, ... model.x8

# declare decision variables

model.x1 = Var(domain=NonNegativeReals)
model.x2 = Var(domain=NonNegativeReals)
model.x3 = Var(domain=NonNegativeReals)
model.x4 = Var(domain=NonNegativeReals)
model.x5 = Var(domain=NonNegativeReals)
model.x6 = Var(domain=NonNegativeReals)
model.x7 = Var(domain=NonNegativeReals)
model.x8 = Var(domain=NonNegativeReals)
Next, my objective function is specified as an algebraic expression involving the decision variables. Here, I store my objective function as model.telephone and use the argument sense as minimize.

# declare objective function to be minimized

model.telephone = Objective(expr = model.x1 + model.x2 + model.x3 + model.x4 + model.x5 + model.x6 + model.x7 + model.x8, sense=minimize)
Constraints are then added s fields to the model, each constraint created using the Constraint().

# declare constraints

model.StreetA = Constraint(expr = model.x1 + model.x2 >= 1)
model.StreetB = Constraint(expr = model.x2 + model.x3 >= 1)
model.StreetC = Constraint(expr = model.x4 + model.x5 >= 1)
model.StreetD = Constraint(expr = model.x7 + model.x8 >= 1)
model.StreetE = Constraint(expr = model.x6 + model.x7 >= 1)
model.StreetF = Constraint(expr = model.x2 + model.x6 >= 1)
model.StreetG = Constraint(expr = model.x1 + model.x6 >= 1)
model.StreetH = Constraint(expr = model.x4 + model.x7 >= 1)
model.StreetI = Constraint(expr = model.x2 + model.x4 >= 1)
model.StreetJ = Constraint(expr = model.x5 + model.x8 >= 1)
model.StreetK = Constraint(expr = model.x3 + model.x5 >= 1)
Step 4. Create a solver object and solve.
The Pyomo library includes a SolverFactory() class used to specify a solver. Because the problem is a linear programming problem, I use the glpk solver.

# using the solver :

results = SolverFactory('glpk').solve(model)
results.write()
if results.solver.status:
    model.pprint()
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem: 
- Name: unknown
  Lower bound: 4.0
  Upper bound: 4.0
  Number of objectives: 1
  Number of constraints: 12
  Number of variables: 9
  Number of nonzeros: 23
  Sense: minimize
# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver: 
- Status: ok
  Termination condition: optimal
  Statistics: 
    Branch and bound: 
      Number of bounded subproblems: 0
      Number of created subproblems: 0
  Error rc: 0
  Time: 0.036015987396240234
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 0
  number of solutions displayed: 0
8 Var Declarations
    x1 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   0.0 :  None : False : False : NonNegativeReals
    x2 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   1.0 :  None : False : False : NonNegativeReals
    x3 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   0.0 :  None : False : False : NonNegativeReals
    x4 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   0.0 :  None : False : False : NonNegativeReals
    x5 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   1.0 :  None : False : False : NonNegativeReals
    x6 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   1.0 :  None : False : False : NonNegativeReals
    x7 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   1.0 :  None : False : False : NonNegativeReals
    x8 : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :     0 :   0.0 :  None : False : False : NonNegativeReals

1 Objective Declarations
    telephone : Size=1, Index=None, Active=True
        Key  : Active : Sense    : Expression
        None :   True : minimize : x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8

11 Constraint Declarations
    StreetA : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x1 + x2 :  +Inf :   True
    StreetB : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x2 + x3 :  +Inf :   True
    StreetC : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x4 + x5 :  +Inf :   True
    StreetD : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x7 + x8 :  +Inf :   True
    StreetE : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x6 + x7 :  +Inf :   True
    StreetF : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x2 + x6 :  +Inf :   True
    StreetG : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x1 + x6 :  +Inf :   True
    StreetH : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x4 + x7 :  +Inf :   True
    StreetI : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x2 + x4 :  +Inf :   True
    StreetJ : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x5 + x8 :  +Inf :   True
    StreetK : Size=1, Index=None, Active=True
        Key  : Lower : Body    : Upper : Active
        None :   1.0 : x3 + x5 :  +Inf :   True

20 Declarations: x1 x2 x3 x4 x5 x6 x7 x8 telephone StreetA StreetB StreetC StreetD StreetE StreetF StreetG StreetH StreetI StreetJ StreetK
# display solution

print('\nNumber of Telephone to Install = ', model.telephone())

print('\nDecision Variables : ')
print('\nFor Telephone to be installed the decision variable should have a value of Xi = 1, otherwise do not Install')
print('x1 = ', int(model.x1()))
print('x2 = ', int(model.x2()))
print('x3 = ', int(model.x3()))
print('x4 = ', int(model.x4()))
print('x5 = ', int(model.x5()))
print('x6 = ', int(model.x6()))
print('x7 = ', int(model.x7()))
print('x8 = ', int(model.x8()))

print('\nEND OF SOLUTION')
Number of Telephone to Install =  4.0

Decision Variables : 

For Telephone to be installed the decision variable should have a value of Xi = 1, otherwise do not Install
x1 =  0
x2 =  1
x3 =  0
x4 =  0
x5 =  1
x6 =  1
x7 =  1
x8 =  0

END OF SOLUTION













