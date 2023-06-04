# Utilities for managing linear and nonlinear dynamics.

# Every Dynamics is assumed to have the following functions defined on it:
# Unique to each struct:
# - dx(dyn, time_range, x, us, v) - computes differntial dynamics, with process noise.
# - propagate_dynamics(dyn, time_range, x, us, v) - this function propagates the dynamics to the next timestep with state, controls, realized process noise.
# - Fx(dyn, time_range, x, us) - first-order derivatives wrt state x
# - Fus(dyn, time_range, x, us) - first-order derivatives wrt state us
# - linearize_discretize(dyn, time_range, x, us) - linearizes and discretizes a continuous-time system.

# Defined on Dynamics
# - dx(dyn, time_range, x, us) - same as the one above, but without process noise.
# - propagate_dynamics(dyn, time_range, x, us) - this function propagates the dynamics to the next timestep with state and controls.

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

function dx(dyn::Dynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]

    return dx(dyn, time_range, x, us, nothing)
end

# A function definition that does not accept process noise input and reroutes to the type-specific propagate_dynamics that does.
function propagate_dynamics(dyn::Dynamics,
                            time_range,
                            x::AbstractVector{Float64},
                            us::AbstractVector{<:AbstractVector{Float64}})
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]

    return propagate_dynamics(dyn, time_range, x, us, nothing)
end

# A function that produces a continuous-time first-order Taylor linearization of the dynamics.
function linearize(dyn::Dynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    t₀, t = time_range
    @assert t₀ ≤ t

    # TODO(hamzah) Add in forward diff usage here, and a way to linearize discretized.
    A = Fx(dyn, time_range, x, us)
    Bs = Fus(dyn, time_range, x, us)
    return ContinuousLinearDynamics(A, Bs)
end

function linearize_discretize(dyn::Dynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    t₀, t = time_range
    @assert t₀ ≤ t

    @assert !is_continuous(dyn) "Input dynamics must have dt > 0 for discretization."
    # TODO(hamzah) Add in forward diff usage here, and a way to linearize discretized.
    A = Fx(dyn, time_range, x, us)
    Bs = Fus(dyn, time_range, x, us)
    cont_dyn = ContinuousLinearDynamics(A, Bs)
    return discretize(cont_dyn, sampling_time(dyn))
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

function sampling_time(dyn::Dynamics)
    return sampling_time(dyn.sys_info)
end

function is_continuous(dyn::Dynamics)
    return iszero(sampling_time(dyn))
end

export num_agents, xdim, udim, vdim, sampling_time, is_continuous
