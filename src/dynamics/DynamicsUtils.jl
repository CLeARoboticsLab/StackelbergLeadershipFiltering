# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# - propagate_dynamics(dyn, time_range, x, us) - this function propagates the dynamics to the next timestep with state and controls.
# - propagate_dynamics(dyn, time_range, x, us, v) - this function propagates the dynamics to the next timestep with state, controls, realized process noise.
# - Fx(dyn, time_range, x, us) - first-order derivatives wrt state x
# - Fus(dyn, time_range, x, us) - first-order derivatives wrt state us

# No dynamics should require homogenized inputs. The functions themselves should transform the inputs/outputs as needed.

# Every Dynamics struct must have
# - a sys_info field of type SystemInfo.
abstract type Dynamics end

# A type that every nonlinear dynamics struct (unique per use case) can inherit from. These need to have the same
# functions as the Dynamics type.
abstract type NonlinearDynamics <: Dynamics end

# This function assumes right multiplication; if left multiplication is done, then matrix should be transposed.
function homogenize_dynamics_matrix(M::AbstractMatrix{Float64}; m=zeros(size(M, 1))::AbstractVector{Float64}, ρ=0.0)
    M_dim2 = size(M, 2)
    cm = (size(M, 1) == size(M, 2)) ? 1. : 0.
    return vcat(hcat(       M        ,  m),
                hcat(zeros(1, M_dim2), cm))
end
export homogenize_dynamics_matrix

# Homogenize state - by default, this adds a 1 to the bottom. If a custom one is needed, define it elsewhere.
function homogenize_state(dyn::Dynamics, xs::AbstractArray{Float64})
    xhs = homogenize_vector(xs)
    @assert size(xhs, 1) == xhdim(dyn)
    return xhs
end

function homogenize_ctrls(dyn::Dynamics, us::AbstractVector{<:AbstractArray{Float64}})
    num_players = num_agents(dyn)
    uhs = [homogenize_vector(us[ii]) for ii in 1:num_players]
    for ii in 1:num_players
        @assert size(uhs[ii], 1) == uhdim(dyn, ii)
    end
    return uhs
end
export homogenize_state, homogenize_ctrls

# By default, generate no process noise. Allow 
function generate_process_noise(dyn::Dynamics, rng)
    return zeros(vdim(dyn))
end

# A function that produces a first-order Taylor linearization of the dynamics.
function linearize_dynamics(dyn::Dynamics,
                            time_range,
                            x0::AbstractVector{Float64},
                            u0s::AbstractVector{<:AbstractVector{Float64}})
    t₀, t = time_range
    dt = t - t₀
    @assert t₀ ≤ t

    A = get_A(dyn, time_range, x0, u0s)
    Bs = get_Bs(dyn, time_range, x0, u0s)
    return LinearDynamics(A, Bs)
end

function get_A(dyn::Dynamics,
               time_range,
               x0::AbstractVector{Float64},
               u0s::AbstractVector{<:AbstractVector{Float64}})
    diff_x = x -> propagate_dynamics(dyn, time_range, x, u0s)
    A = ForwardDiff.jacobian(diff_x, x0)
    return A - I
end

function get_Bs(dyn::Dynamics,
                time_range,
                x0::AbstractVector{Float64},
                u0s::AbstractVector{<:AbstractVector{Float64}})
    u0s_combined = vcat(u0s...)
    diff_us = us -> propagate_dynamics(dyn, time_range, x0, split(us, num_us(dyn)))
    combined_Bs = ForwardDiff.jacobian(diff_us, u0s_combined)
    Bs = transpose.(split(combined_Bs', num_us(dyn)))
    return Bs
end

# Export the types of dynamics.
export Dynamics, NonlinearDynamics, generate_process_noise, linearize_dynamics


# Dimensionality helpers.
function num_agents(dyn::Dynamics)
    return num_agents(dyn.sys_info)
end

function xdim(dyn::Dynamics)
    return xdim(dyn.sys_info)
end

function udim(dyn::Dynamics)
    return udim(dyn.sys_info)
end

function num_us(dyn::Dynamics)
    return num_us(dyn.sys_info)
end

function udim(dyn::Dynamics, player_idx)
    return udim(dyn.sys_info, player_idx)
end

function vdim(dyn::Dynamics)
    return vdim(dyn.sys_info)
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

export num_agents, xdim, udim, num_us, vdim, xhdim, uhdim
