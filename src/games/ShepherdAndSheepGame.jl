# This file defines the dynamics and cost functions for the Shepherd and Sheep example.

# Two player dynamics
# Dynamics (Euler-discretized double integrator equations with Δt = 0.1s).
# State for each player is layed out as [x, ẋ, y, ẏ].

using LinearAlgebra

num_players = 2
num_states = 8
num_ctrls = [2, 2]

shepherd_Ã(dt) = [1 dt  0  0;
                  0  1  0  0;
                  0  0  1 dt;
                  0  0  0  1]
shepherd_A(dt) = vcat(hcat(shepherd_Ã(dt), zeros(4, 4)),
                      hcat(zeros(4, 4), shepherd_Ã(dt)))

B₁(dt) = vcat([0   0;
               dt  0;
               0   0;
               0   dt],
               zeros(4, 2))
B₂(dt) = vcat(zeros(4, 2),
              [0   0;
               dt  0;
               0   0;
               0   dt])

ShepherdAndSheepDynamics(dt) = LinearDynamics(shepherd_A(dt), [B₁(dt), B₂(dt)])


# Costs reflecting the preferences above.
Q₁ = zeros(8, 8)
Q₁[5, 5] = 1.0
Q₁[7, 7] = 1.0
c₁ = QuadraticCost(Q₁)
add_control_cost!(c₁, 1, 1 * diagm([1, 1]))
add_control_cost!(c₁, 2, zeros(2, 2))

Q₂ = zeros(8, 8)
Q₂[1, 1] = 1.0
Q₂[5, 5] = 1.0
Q₂[1, 5] = -1.0
Q₂[5, 1] = -1.0
Q₂[3, 3] = 1.0
Q₂[7, 7] = 1.0
Q₂[3, 7] = -1.0
Q₂[7, 3] = -1.0
c₂ = QuadraticCost(Q₂)
add_control_cost!(c₂, 2, 1 * diagm([1, 1]))
add_control_cost!(c₂, 1, zeros(2, 2))

# Gets a vector of costs, one per player.
ShepherdAndSheepCosts() = [c₁, c₂]


# Nonlinear, but still quadratic, version of this game.

ShepherdAndSheepWithUnicycleDynamics() = UnicycleDynamics(2)

# P1 wants P2 to go to origin in position
Q₃ = zeros(8, 8)
Q₃[5, 5] = 1.0
Q₃[6, 6] = 1.0
c₃ = QuadraticCost(Q₃)
add_control_cost!(c₃, 1, .1 * diagm([1, 1]))
add_control_cost!(c₃, 2, zeros(2, 2))

# P2 wants to go to P1's position
Q₄ = zeros(8, 8)
Q₄[1, 1] = 1.0
Q₄[5, 5] = 1.0
Q₄[1, 5] = -1.0
Q₄[5, 1] = -1.0
Q₄[2, 2] = 1.0
Q₄[6, 6] = 1.0
Q₄[2, 6] = -1.0
Q₄[6, 2] = -1.0

# Q₄[3, 3] = 1.0
# Q₄[7, 7] = 1.0
# Q₄[3, 7] = -1.0
# Q₄[7, 3] = -1.0
c₄ = QuadraticCost(Q₄)
add_control_cost!(c₄, 2, .1 * diagm([1, 1]))
add_control_cost!(c₄, 1, zeros(2, 2))

# Gets a vector of costs, one per player.
UnicycleShepherdAndSheepWithQuadraticCosts() = [c₃, c₄]


# Nonquadratic, but still linear, version of this game.
costs = ShepherdAndSheepCosts()
costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

# P1 - avoid crossing the line [x; y] - [-1/2; -1/2] = 0
indices1 = [1, 3] # x, y
x_offset_1 = zeros(num_states)
x_offset_1[indices1] .= -(1.0/2)
x_zero_int_1 = zeros(num_states)
x_zero_int_1[indices1] .= 10.
log_cost_p1 = LogBarrierCost(indices1, x_offset_1, x_zero_int_1)

# P2 - avoid crossing the line [x; y] - [-1/2; -1/2] = 0
indices2 = [5, 7] # x, y
x_offset_2 = zeros(num_states)
x_offset_2[indices2] .= -(1.0/2)
x_zero_int_2 = zeros(num_states)
x_zero_int_2[indices2] .= 10.
log_cost_p2 = LogBarrierCost(indices2, x_offset_2, x_zero_int_2)

# Make the weighted cost.
p1_new_cost = WeightedCost([1.0, 0.5], [deepcopy(costs[1]), log_cost_p1])
# new_costs = [p1_new_cost, costs[2]]

p2_new_cost = WeightedCost([1.0, 0.5], [deepcopy(costs[2]), log_cost_p2])
# new_costs = [costs[1], p2_new_cost]

ShepherdAndSheepWithLogBarrierQuadraticCosts() = [p1_new_cost, p2_new_cost]

export ShepherdAndSheepDynamics, ShepherdAndSheepWithUnicycleDynamics, ShepherdAndSheepCosts, UnicycleShepherdAndSheepWithQuadraticCosts, ShepherdAndSheepWithLogBarrierQuadraticCosts
