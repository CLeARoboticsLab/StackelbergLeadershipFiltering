# Quadratic cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
mutable struct PureQuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    Rs
end
PureQuadraticCost(Q) = PureQuadraticCost(Q, Dict{Int, Matrix{eltype(Q)}}())

# TODO(hamzah) Add better tests for the PureQuadraticCost struct and associated functions.

# Method to add R^{ij}s to a Cost struct.
export add_control_cost!
function add_control_cost!(c::PureQuadraticCost, other_player_idx, Rij)
    c.Rs[other_player_idx] = Rij
end

# TODO(hamzah): Use Julia's conversion and promotion for this? Or at least for affine costs.
function quadraticize_costs(cost::PureQuadraticCost, time_range, x, us)
    return cost
end

# Helpers that get the homogenized Q and R matrices for this cost.
# TODO(hmzh) - remove these when I get rid of pure quadratic cost.
function get_homogenized_state_cost_matrix(c::PureQuadraticCost)
    return c.Q
end

function get_homogenized_control_cost_matrix(c::PureQuadraticCost, player_idx::Int)
    return c.Rs[player_idx]
end


# Evaluate cost on a state/control trajectory at a particule time.
function compute_cost(c::PureQuadraticCost, time_range, xh::AbstractVector{Float64}, uhs::AbstractVector{<:AbstractVector{Float64}})
    x = xh[1:size(c.Q, 1)]
    cost = (1/2.) * x' * c.Q * x
    if !isempty(c.Rs)
        cost += (1/2.) * sum(uhs[jj][1:size(Rij, 1)]' * Rij * uhs[jj][1:size(Rij, 1)] for (jj, Rij) in c.Rs)
    end
    return cost
end

# Export all the cost types/structs.
export PureQuadraticCost

# Export all the cost types/structs.
export add_control_cost!, quadraticize_costs, compute_cost
