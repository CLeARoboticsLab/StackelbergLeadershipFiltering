# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# - linearize_dynamics(dyn, x, us) - this function linearizes the dynamics given the state and controls.
# - propagate_dynamics(dyn, time_range, x, us) - this function propagates the dynamics to the next timestep with state and controls.
# - propagate_dynamics(dyn, time_range, x, us, v) - this function propagates the dynamics to the next timestep with state, controls, realized process noise.
# - homogenize_state(dyn, xs) - needs to be defined if dynamics requires linear/constant terms
# - homogenize_ctrls(dyn, us) - needs to be defined if dynamics requires linear/constant terms

# Every Dynamics struct must have
# - a sys_info field of type SystemInfo and
# - is_homogenized boolean.
abstract type Dynamics end

# A type that every nonlinear dynamics struct (unique per use case) can inherit from. These need to have the same
# functions as the Dynamics type.
abstract type NonlinearDynamics <: Dynamics end

# Homogenize state - by default, this adds a 1 to the bottom. If a custom one is needed, define it elsewhere.
function homogenize_state(dyn::Dynamics, xs::AbstractMatrix{Float64})
    return vcat(xs, ones(1, size(xs, 2)))
end

function homogenize_ctrls(dyn::Dynamics, us::AbstractVector{<:AbstractMatrix{Float64}})
    num_players = num_agents(dyn)
    return [vcat(us[ii], ones(1, size(us[ii], 2))) for ii in 1:num_players]
end

# By default, generate no process noise. Allow 
function generate_process_noise(dyn::Dynamics, rng)
    return zeros(vdim(dyn))
end

# Export the types of dynamics.
export Dynamics, NonlinearDynamics, generate_process_noise


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

function udim(dyn::Dynamics, player_idx)
    return udim(dyn.sys_info, player_idx)
end

function vdim(dyn::Dynamics)
    return vdim(dyn.sys_info)
end

export num_agents, xdim, udim
