# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# - propagate_dynamics(dyn, time_range, x, us) - this function propagates the dynamics to the next timestep with state and controls.
# - propagate_dynamics(dyn, time_range, x, us, v) - this function propagates the dynamics to the next timestep with state, controls, realized process noise.
# - Fx(dyn, time_range, x, us) - first-order derivatives wrt state x
# - Fus(dyn, time_range, x, us) - first-order derivatives wrt state us
# - plot_states_and_controls(dyn, times, xs, us) - produce Plots.jl versions of plots for all states in the dynamics provided

# Every Dynamics struct must have
# - a sys_info field of type SystemInfo.
abstract type Dynamics end

# A type that every nonlinear dynamics struct (unique per use case) can inherit from. These need to have the same
# functions as the Dynamics type.
abstract type NonlinearDynamics <: Dynamics end

# By default, generate no process noise. Allow 
function generate_process_noise(dyn::Dynamics, rng)
    return zeros(vdim(dyn))
end

# A function that produces a first-order Taylor linearization of the dynamics.
function linearize_dynamics(dyn::Dynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    t₀, t = time_range
    dt = t - t₀
    @assert t₀ ≤ t

    A = I + dt * Fx(dyn, time_range, x, us)
    Bs = Fus(dyn, time_range, x, us)
    return LinearDynamics(A, Bs)
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

function udim(dyn::Dynamics, player_idx)
    return udim(dyn.sys_info, player_idx)
end

function vdim(dyn::Dynamics)
    return vdim(dyn.sys_info)
end

export num_agents, xdim, udim, vdim
