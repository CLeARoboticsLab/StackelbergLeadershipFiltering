using SparseArrays

# This util is meant to be used where the state can be decomposed into N players which each act according to unicycle
# dynamics. A unicycle dynamics model with one actor has a 4-element state including 2D-position, angle, and speed.
# There are two controls, turn rate and acceleration.
# X_i = [px_i py_i theta_i v_i]'
# U_i = [omega_i alpha_i]'
# Unicycle dynamics indepedent of other actors can then be composed into a single state by concatenation and with some
# minor changes to the control matrices B_i.
struct UnicycleDynamics <: NonlinearDynamics
    sys_info::SystemInfo
end

# Constructor
UnicycleDynamics(num_players::Int) = UnicycleDynamics(SystemInfo(num_players, 4*num_players, 2*ones(num_players)))

function propagate_dynamics(dyn::UnicycleDynamics,
                            time_range,
                            x::AbstractVector{T},
                            us::AbstractVector{<:AbstractVector{T}}) where T
    # In this nonlinear system, no need to homogenize the inputs because we don't matrix multiply anywhere.
    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2 * N
    end

    # Convert the inputs to the necessary type.

    x_tp1 = x
    dt = time_range[2] - time_range[1]

    for ii in 1:N
        start_idx = 4 * (ii-1)
        px = x[start_idx + 1]
        py = x[start_idx + 2]
        theta = x[start_idx + 3]
        vel = x[start_idx + 4]

        turn_rate = us[ii][1]
        accel = us[ii][2]

        # x_tp1[start_idx+1:start_idx+4] += dt * [vel * cos(theta); vel * sin(theta); turn_rate; accel]
        x_tp1[start_idx+1] += dt * vel * cos(theta)
        x_tp1[start_idx+2] += dt * vel * sin(theta)
        x_tp1[start_idx+3] += dt * turn_rate
        x_tp1[start_idx+4] += dt * accel
        # ; vel * sin(theta); turn_rate; accel

        # Wrap angle before propagation
        x_tp1[start_idx+3] = wrap_angle(x_tp1[start_idx+3])
    end

    return x_tp1
end

# TODO: Unicycle dynamics doesn't currently support process noise.
function propagate_dynamics(dyn::UnicycleDynamics,
                            time_range,
                            x::AbstractVector{T},
                            us::AbstractVector{<:AbstractVector{T}},
                            v::AbstractVector{Float64}) where T
    throw(MethodError("propagate_dynamics not implemented with process noise for UnicycleDynamics"))
end

# These are the continuous derivatives of the unicycle dynamics with respect to x and u.

function Fx(dyn::UnicycleDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2 * N
    end
    num_states = xdim(dyn)
    As = [sparse(zeros(num_states, num_states)) for ii in 1:N]
    for ii in 1:N
        start_idx = num_states * (ii-1)
        theta = x[start_idx + 3]
        v = x[start_idx + 4]

        # Compute the state and controls for each actor.
        s = sin(theta)
        c = cos(theta)
        As[ii][1:2, 3:4] = [-v*s c; v*c s]
    end
    return Matrix(blockdiag(As...))
end

function Fus(dyn::UnicycleDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    N = num_agents(dyn)
    @assert N == length(us)
    @assert size(x, 1) == 4 * N
    for ii in 1:N
        @assert size(us[ii], 1) == 2 * N
    end
    num_states = xdim(dyn)
    Bs = [zeros(xdim(dyn), udim(dyn, ii)) for ii in 1:N]
    prev_time, curr_time = time_range
    dt = curr_time - prev_time
    for ii in 1:N
        start_idx = num_states * (ii-1)
        Bs[ii][start_idx+3:start_idx+4, 1:2] = dt * [1 0; 0 1]
    end
    return Bs
end

export UnicycleDynamics, propagate_dynamics, linearize_dynamics, Fx, Fus
