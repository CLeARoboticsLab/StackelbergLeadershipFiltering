# This file defines the dynamics and cost functions for the Shepherd and Sheep example.

# Two player dynamics
# Dynamics (Euler-discretized double integrator equations with Δt = 0.1s).
# State for each player is layed out as [x, ẋ, y, ẏ].

using LinearAlgebra

num_players = 2
num_states = 8
num_ctrls = [2, 2]

# INDICES that assume two-player games
xidx(dyn::LinearDynamics, p_idx::Int) = 4 * (p_idx - 1) + 1
yidx(dyn::LinearDynamics, p_idx::Int) = 4 * (p_idx - 1) + 3
xidx(dyn::UnicycleDynamics, p_idx::Int) = 4 * (p_idx - 1) + 1
yidx(dyn::UnicycleDynamics, p_idx::Int) = 4 * (p_idx - 1) + 2
export xidx, yidx

shepherd_Ã() = [0 1.  0  0;
                0  0  0  0;
                0  0  0 1.;
                0  0  0  0]
shepherd_A() = vcat(hcat(shepherd_Ã(), zeros(4, 4)),
                      hcat(zeros(4, 4), shepherd_Ã()))

B₁() = vcat([0   0;
             1  0;
             0   0;
             0   1],
             zeros(4, 2))
B₂() = vcat(zeros(4, 2),
            [0   0;
             1  0;
             0   0;
             0   1])

ShepherdAndSheepDynamics() = ContinuousLinearDynamics(shepherd_A(), [B₁(), B₂()])

# Gets a vector of costs, one per player.
ShepherdAndSheepCosts(dyn::Dynamics) = begin
    
    # Costs reflecting the preferences above.
    p1x_idx = xidx(dyn, 1)
    p1y_idx = yidx(dyn, 1)
    p2x_idx = xidx(dyn, 2)
    p2y_idx = yidx(dyn, 2)

    Q₁ = zeros(8, 8)
    Q₁[p2x_idx, p2x_idx] = 1.0
    Q₁[p2y_idx, p2y_idx] = 1.0
    c₁ = QuadraticCost(1*Q₁)
    add_control_cost!(c₁, 1, .1 * diagm([1, 1]))
    add_control_cost!(c₁, 2, zeros(2, 2))

    Q₂ = zeros(8, 8)
    Q₂[p1x_idx, p1x_idx] = 1.0
    Q₂[p2x_idx, p2x_idx] = 1.0
    Q₂[p1x_idx, p2x_idx] = -1.0
    Q₂[p2x_idx, p1x_idx] = -1.0
    Q₂[p1y_idx, p1y_idx] = 1.0
    Q₂[p2y_idx, p2y_idx] = 1.0
    Q₂[p1y_idx, p2y_idx] = -1.0
    Q₂[p2y_idx, p1y_idx] = -1.0
    c₂ = QuadraticCost(1*Q₂)
    add_control_cost!(c₂, 2, .1 * diagm([1, 1]))
    add_control_cost!(c₂, 1, zeros(2, 2))

    return [c₁, c₂]
end


# Nonlinear, but still quadratic, version of this game.

ShepherdAndSheepWithUnicycleDynamics(dt) = UnicycleDynamics(2, dt)

# # P1 wants P2 to go to origin in position
# Q₃ = zeros(8, 8)
# Q₃[5, 5] = 1.0
# Q₃[6, 6] = 1.0
# c₃ = QuadraticCost(Q₃)
# add_control_cost!(c₃, 1, .1 * diagm([1, 1]))
# add_control_cost!(c₃, 2, zeros(2, 2))

# # P2 wants to go to P1's position
# Q₄ = zeros(8, 8)
# Q₄[1, 1] = 1.0
# Q₄[5, 5] = 1.0
# Q₄[1, 5] = -1.0
# Q₄[5, 1] = -1.0
# Q₄[2, 2] = 1.0
# Q₄[6, 6] = 1.0
# Q₄[2, 6] = -1.0
# Q₄[6, 2] = -1.0

# # Q₄[3, 3] = 1.0
# # Q₄[7, 7] = 1.0
# # Q₄[3, 7] = -1.0
# # Q₄[7, 3] = -1.0
# c₄ = QuadraticCost(Q₄)
# add_control_cost!(c₄, 2, 1 * diagm([1, 1]))
# add_control_cost!(c₄, 1, zeros(2, 2))

# # Gets a vector of costs, one per player.
# UnicycleShepherdAndSheepWithQuadraticCosts() = [c₃, c₄]


# # Nonquadratic, but still linear, version of this game.
# costs = ShepherdAndSheepCosts()
# costs = [QuadraticCostWithOffset(costs[1]), QuadraticCostWithOffset(costs[2])]

# # P1 - avoid crossing the line [x; y] - [-1/2; -1/2] = 0
# indices1 = [1, 3] # x, y
# x_offset_1 = zeros(num_states)
# x_offset_1[indices1] .= -(1.0/2)
# x_zero_int_1 = zeros(num_states)
# x_zero_int_1[indices1] .= 10.
# log_cost_p1 = LogBarrierCost(indices1, x_offset_1, x_zero_int_1)

# # P2 - avoid crossing the line [x; y] - [-1/2; -1/2] = 0
# indices2 = [5, 7] # x, y
# x_offset_2 = zeros(num_states)
# x_offset_2[indices2] .= -(1.0/2)
# x_zero_int_2 = zeros(num_states)
# x_zero_int_2[indices2] .= 10.
# log_cost_p2 = LogBarrierCost(indices2, x_offset_2, x_zero_int_2)

# # Make the weighted cost.
# p1_new_cost = WeightedCost([1.0, 0.5], [deepcopy(costs[1]), log_cost_p1])
# # new_costs = [p1_new_cost, costs[2]]

# p2_new_cost = WeightedCost([1.0, 0.5], [deepcopy(costs[2]), log_cost_p2])
# # new_costs = [costs[1], p2_new_cost]

# ShepherdAndSheepWithLogBarrierQuadraticCosts() = [p1_new_cost, p2_new_cost]


# A game with linear dynamics, where
# - P1 has log barriers on state at x₂=±bound_val, y₂=±bound_val. This should converge to the origin, but can be adjusted based on
#   relative log barrier weights. There is also a quadratic cost with almost no state cost (enough to be PD) and some control cost.
# - P2 has quadratic cost and seeks it's position to be close to P1.
#   TODO(hamzah) - introduce a relative logarithmic cost which accounts for maximum allowed difference.

ShepherdAndSheepWithLogBarrierCost(dyn::Dynamics, agent_idx, bounded_agent_idx, x_bounds, y_bounds) = begin
    zero_us = [zeros(2) for ii in 1:2]
    time_range = (0., 0.05)

    x_idx = xidx(dyn, bounded_agent_idx)
    y_idx = yidx(dyn, bounded_agent_idx)

    # 1. bound P2 x position below.
    c1 = AbsoluteLogBarrierCost(x_idx, x_bounds[1], true)
    # 2. bound P2 x position above.
    c2 = AbsoluteLogBarrierCost(x_idx, x_bounds[2], false)
    # 3. bound P2 y position below.
    c3 = AbsoluteLogBarrierCost(y_idx, y_bounds[1], true)
    # 4. bound P2 y position above.
    c4 = AbsoluteLogBarrierCost(y_idx, y_bounds[2], false)
    # 5. quadratic control cost on P1 by P1
    # c5 = QuadraticCost(zeros(num_states, num_states))
    # add_control_cost!(c5, 1, 10 * diagm([1, 1]))
    # add_control_cost!(c5, 2, zeros(2, 2))
    # c5 = QuadraticCostWithOffset(c₁)
    Q₃ = zeros(8, 8)
    # Q₃[p2x_idx, p2x_idx] = 10.0
    # Q₃[p2y_idx, p2y_idx] = 10.0
    # qc = QuadraticCost(Q₃)
    # add_control_cost!(qc, 1, 1 * diagm([1, 1]))
    # add_control_cost!(qc, 2, zeros(2, 2))

    costs = [c1, c2, c3, c4, QuadraticCostWithOffset(ShepherdAndSheepCosts(dyn)[agent_idx])]
    weights = ones(5)
    weights[5] = 1.
    return WeightedCost(weights, costs)
end

ShepherdAndSheepWithLogBarrierOverallCosts(dyn::Dynamics, bound_val::Real) = ShepherdAndSheepWithLogBarrierOverallCosts(dyn, (-bound_val, bound_val), (-bound_val, bound_val))
ShepherdAndSheepWithLogBarrierOverallCosts(dyn::Dynamics, x_bounds, y_bounds) = begin
    return [ShepherdAndSheepWithLogBarrierCost(dyn, 1, 2, x_bounds, y_bounds),
            # ShepherdAndSheepWithLogBarrierCost(dyn, 2, 1, x_bounds, y_bounds)]
            QuadraticCostWithOffset(ShepherdAndSheepCosts(dyn)[2])]
end


export ShepherdAndSheepDynamics
export ShepherdAndSheepWithUnicycleDynamics
export ShepherdAndSheepCosts
export UnicycleShepherdAndSheepWithQuadraticCosts
# export ShepherdAndSheepWithLogBarrierQuadraticCosts
export ShepherdAndSheepWithLogBarrierOverallCosts
