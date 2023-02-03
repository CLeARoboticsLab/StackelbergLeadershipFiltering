# Cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    Rs
    is_homogenized::Bool
end
QuadraticCost(Q) = QuadraticCost(Q, Dict{Int, Matrix{eltype(Q)}}(), false) # quadratic costs are non homogenized by default

# TODO(hamzah) Add better tests for the QuadraticCost struct and associated functions.

# Method to add R^{ij}s to a Cost struct.
export add_control_cost!
function add_control_cost!(c::QuadraticCost, other_player_idx, Rij)
    c.Rs[other_player_idx] = Rij
end

function affinize_costs(cost::QuadraticCost, time_range, x, us)
    return cost
end

# Evaluate cost on a state/control trajectory at a particule time.
function compute_cost(c::QuadraticCost, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return x' * c.Q * x + sum(us[jj]' * Rij * us[jj] for (jj, Rij) in c.Rs)
end

# Export all the cost types/structs.
export QuadraticCost

# Export all the cost types/structs.
export add_control_cost!, affinize_costs, compute_cost
