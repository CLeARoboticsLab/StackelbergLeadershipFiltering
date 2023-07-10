# This file contains utility functions for the homogenization of vectors, dynamics matrices, and cost matrices.

#########################
# Vector homogenization #
#########################

function homogenize_vector(v::AbstractVector{T}) where {T}
    return vcat(v, 1)
end

function homogenize_vector(vs::AbstractMatrix{T}) where {T}
    return vcat(vs, ones(1, size(vs, 2)))
end
export homogenize_vector


###############################################
# Matrix homogenization for dynamics matrices #
###############################################

# This function assumes right multiplication; if left multiplication is done, then matrix should be transposed.
function homogenize_dynamics_matrix(M::AbstractMatrix{T}; m=zeros(size(M, 1))::AbstractVector{T}, ρ=0.0::T) where {T}
    M_dim2 = size(M, 2)
    # TODO(hamzah): This is buggy code and should really be dependent on whether we are homogenizing A or B.
    cm = (size(M, 1) == size(M, 2)) ? 1. : 0.
    return vcat(hcat(       M        ,  m),
                hcat(zeros(1, M_dim2), cm))
end
export homogenize_dynamics_matrix


###########################################
# Matrix homogenization for cost matrices #
###########################################

# Produces a symmetric matrix.
function homogenize_cost_matrix(M::AbstractMatrix{T}, m=zeros(size(M, 1))::AbstractVector{T}, cm=0.0::T, ρ=nothing) where {T}
    return vcat(hcat(M ,  m), hcat(m', cm))
end


###################################################
# Convenience homogenization tools for LQ systems #
###################################################

function regularize_matrix(M, reg_param, ensure_pd)
    # If already positive definite, no need to alter anything.
    if iszero(reg_param) || isposdef(M)
        return M
    end

    # If desired, ensure positive definiteness (after third step) by spectrally shifting by magnitude of smallest eigenvalue.
    reg_M = deepcopy(M)
    if ensure_pd
        eig_min = minimum(eigvals(M))
        @assert eig_min ≤ 1e-8 string("eigenval: ", eig_min) # some leeway for small positives...
        reg_M = reg_M + abs(eig_min) * I
    end

    # Finally, spectral shift by the regularization parameter and return.
    return reg_M + reg_param * I
end

# Helpers that get the homogenized Q and R matrices for a quadratic cost matrix.
function get_homogenized_state_cost_matrix(c::QuadraticCost; reg_param=0.0, ensure_pd=false)
    Qh = homogenize_cost_matrix(c.Q, c.q, c.cq)
    return regularize_matrix(Qh, reg_param, ensure_pd)
end

function get_homogenized_control_cost_matrix(c::QuadraticCost, player_idx::Int; reg_param=0.0, ensure_pd=false)
    Rh = homogenize_cost_matrix(c.Rs[player_idx], c.rs[player_idx], c.crs[player_idx])
    return regularize_matrix(Rh, reg_param, ensure_pd)
end

export get_homogenized_state_cost_matrix, get_homogenized_control_cost_matrix


# Helpers that get the homogenized A and B for a linear dynamical system.
function get_homogenized_state_dynamics_matrix(dyn::LinearDynamics)
    return homogenize_dynamics_matrix(dyn.A; m=dyn.a)
end

function get_homogenized_control_dynamics_matrix(dyn::LinearDynamics, player_idx::Int)
    return homogenize_dynamics_matrix(dyn.Bs[player_idx])
end

export get_homogenized_state_dynamics_matrix, get_homogenized_control_dynamics_matrix


###########################################
# Matrix homogenization for cost matrices #
###########################################


# For homogenized dimensions
function xhdim(sys_info::SystemInfo)
    return sys_info.num_x + 1
end

function uhdim(sys_info::SystemInfo)
    return sum(uhdim(sys_info, ii) for ii in 1:num_agents(sys_info))
end

function uhdim(sys_info::SystemInfo, player_idx)
    return sys_info.num_us[player_idx] + 1
end

# homgenized dimension sizes
function xhdim(dyn::Dynamics)
    return xdim(dyn.sys_info) + 1
end

function uhdim(dyn::Dynamics)
    return sum(uhdim(dyn, ii) for ii in 1:num_agents(dyn))
end

function uhdim(dyn::Dynamics, player_idx)
    return udim(dyn.sys_info, player_idx) + 1
end
export xhdim, uhdim
