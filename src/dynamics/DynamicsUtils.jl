using ForwardDiff

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

# TODO(hamzah)  - combine these functions into one.
function dx(dyn::Dynamics, time_range, x::AbstractVector{TX}, us::AbstractVector{<:AbstractVector{Float64}}) where {TX}
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]

    adjusted_us = [TX.(u) for u in us]
    return dx(dyn, time_range, x, adjusted_us, nothing)
end

function dx(dyn::Dynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{TU}}) where {TU}
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]

    return dx(dyn, time_range, TU.(x), us, nothing)
end

# A function definition that does not accept process noise input and reroutes to the type-specific propagate_dynamics that does.
function propagate_dynamics(dyn::Dynamics, time_range, x, us)
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]

    return propagate_dynamics(dyn, time_range, x, us, nothing)
end

# A function that produces a continuous-time first-order Taylor linearization of the dynamics.
function linearize(dyn::Dynamics, time_range, x, us)
    t₀, t = time_range
    @assert t₀ ≤ t

    # TODO(hamzah) Add in forward diff usage here.
    A = get_A(dyn, time_range, x, us)
    Bs = get_Bs(dyn, time_range, x, us)
    return ContinuousLinearDynamics(A, Bs)
end

function get_A(dyn::Dynamics, time_range, x0, u0s)
    diff_x = x -> dx(dyn, time_range, x, u0s)
    A = ForwardDiff.jacobian(diff_x, x0)
    return A
end

function get_Bs(dyn::Dynamics, time_range, x0, u0s)
    u0s_combined = vcat(u0s...)
    diff_us = us -> dx(dyn, time_range, x0, split(us, udims(dyn)))
    combined_Bs = ForwardDiff.jacobian(diff_us, u0s_combined)
    Bs = transpose.(split(combined_Bs', udims(dyn)))
    return Bs
end

function linearize_discretize(dyn::Dynamics, time_range, x, us)
    t₀, t = time_range
    @assert t₀ ≤ t

    @assert !is_continuous(dyn) "Input dynamics must have dt > 0 for discretization."
    # TODO(hamzah) Add in forward diff usage here, and a way to linearize discretized.
    A = get_A(dyn, time_range, x, us)
    Bs = get_Bs(dyn, time_range, x, us)
    cont_dyn = ContinuousLinearDynamics(A, Bs)
    return discretize(cont_dyn, sampling_time(dyn))
end

# Export the types of dynamics.
export Dynamics, NonlinearDynamics, generate_process_noise, linearize_dynamics


# A function definition that uses RK4 integration to provide the next step.
# No process noise for now. Some aggregate dynamics (i.e. DynamicsWithHistory) will define their own.
function step_rk4(dyn::Dynamics, time_range, x, us, v=nothing)
    # Ensure that there should not be any process noise.
    @assert vdim(dyn) == 0
    @assert time_range[1] ≤ time_range[2]
    δt = time_range[2] - time_range[1]

    k₁ = dx(dyn, time_range, x, us, v)
    k₂ = dx(dyn, time_range, x + (k₁/2) * δt, us, v)
    k₃ = dx(dyn, time_range, x + (k₂/2) * δt, us, v)
    k₄ = dx(dyn, time_range, x + k₃ * δt, us, v)

    return x + (1/6) * (k₁ + 2*k₂ + 2*k₃ + k₄) * δt
end

export step_rk4


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

function udims(dyn::Dynamics)
    return udims(dyn.sys_info)
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
