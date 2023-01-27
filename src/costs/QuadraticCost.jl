# Cost for a single player.
# Form is: x^T_t Q^i x + \sum_j u^{jT}_t R^{ij} u^j_t.
# For simplicity, assuming that Q, R are time-invariant, and that dynamics are
# linear time-invariant, i.e. x_{t+1} = A x_t + \sum_i B^i u^i_t.
mutable struct QuadraticCost <: Cost
    Q::AbstractMatrix{Float64}
    Rs
end
QuadraticCost(Q) = QuadraticCost(Q, Dict{Int, Matrix{eltype(Q)}}())

# TODO(hamzah) Add better tests for the QuadraticCost struct and associated functions.

# Method to add R^{ij}s to a Cost struct.
export add_control_cost!
function add_control_cost!(c::QuadraticCost, other_player_idx, Rij)
    c.Rs[other_player_idx] = Rij
end

function quadraticize_costs(cost::QuadraticCost, time_range, x, us)
    return cost
end

# Evaluate cost on a state/control trajectory.
# - xs[:, time]
# - us[player][:, time]
function evaluate(c::QuadraticCost, xs, us)
    horizon = last(size(xs))

    total = 0.0
    for tt in 1:horizon
        total += xs[:, tt]' * c.Q * xs[:, tt]
        total += sum(us[jj][:, tt]' * Rij * us[jj][:, tt] for (jj, Rij) in c.Rs)
    end
    return total
end

# Export all the cost types/structs.
export QuadraticCost

# Export all the cost types/structs.
export quadraticize_costs, evaluate
