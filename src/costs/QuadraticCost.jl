# Quadratic cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    Rs
    is_pure::Bool
end
QuadraticCost(Q) = QuadraticCost(Q, Dict{Int, Matrix{eltype(Q)}}(), true)

# function construct_q_cost()
#     Q_dim = size(Q, 1)
#     @assert all(Q .== Q') # check symmetry
#     @assert all(Q[1:, size(Q, 1)] .== vcat(zeros(Q_dim-1), 1)) # check that there are no linear terms and constant term
#     # quadratic costs are non homogenized by default
# end

# TODO(hamzah) Add better tests for the QuadraticCost struct and associated functions.

# Method to add R^{ij}s to a Cost struct.
export add_control_cost!
function add_control_cost!(c::QuadraticCost, other_player_idx, Rij)
    c.Rs[other_player_idx] = Rij
end

# TODO(hamzah): Use Julia's conversion and promotion for this? Or at least for affine costs.
function quadraticize_costs(cost::QuadraticCost, time_range, x, us)
    return cost
end

# Evaluate cost on a state/control trajectory at a particule time.
function compute_cost(c::QuadraticCost, time_range, xh::AbstractVector{Float64}, uhs::AbstractVector{<:AbstractVector{Float64}})
    @assert size(xh, 1) == size(c.Q, 1)
    cost = xh' * c.Q * xh
    # println("1: ", time_range, cost, c.Rs)
    if !isempty(c.Rs)
        cost += sum(uhs[jj]' * Rij * uhs[jj] for (jj, Rij) in c.Rs)
        # println("2+: ", time_range, cost)
    end
    return cost
end

# Export all the cost types/structs.
export QuadraticCost

# Export all the cost types/structs.
export add_control_cost!, quadraticize_costs, compute_cost
