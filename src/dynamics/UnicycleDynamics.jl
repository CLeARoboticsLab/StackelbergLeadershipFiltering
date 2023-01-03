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

function propagate_dynamics(dyn::UnicycleDynamics,
                            t,
                            x::AbstractVector{Float64},
                            us::AbstractVector{<:AbstractVector{Float64}})
    N = dyn.sys_info.num_agents
    @assert N == length(us)
    @assert xdim(dyn) == 4 * N
    @assert udim(dyn) == 2 * N

    x_dot = zeros(xdim(dyn))

    for ii in 1:N
        start_idx = 4 * (ii-1)
        px = x[start_idx + 1]
        py = x[start_idx + 2]
        theta = x[start_idx + 3]
        vel = x[start_idx + 4]

        turn_rate = us[ii][1]
        accel = us[ii][2]

        x_dot[start_idx+1:start_idx+4] = [vel * cos(theta); vel * sin(theta); turn_rate; accel]
    end

    return x_dot
end

# TODO: Unicycle dynamics doesn't currently support process noise.
function propagate_dynamics(dyn::UnicycleDynamics,
                            t,
                            x::AbstractVector{Float64},
                            us::AbstractVector{<:AbstractVector{Float64}},
                            v::AbstractVector{Float64})
    throw(MethodError("propagate_dynamics not implemented with process noise for UnicycleDynamics"))
end

function linearize_dynamics(dyn::UnicycleDynamics, t, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    N = dyn.sys_info.num_agents
    @assert N == length(us)
    @assert xdim(dyn) == 4 * N

    As = [sparse(zeros(4, 4)) for ii in 1:N]
    Bs = [zeros(xdim(dyn), 2) for ii in 1:N]

    for ii in 1:N
        @assert udim(dyn, ii) == 2

        start_idx = 4 * (ii-1)
        theta = x[start_idx + 3]
        v = x[start_idx + 4]

        # Compute the state and controls for each actor.
        s = sin(theta)
        c = cos(theta)
        As[ii][1:2, 3:4] = [-v*s c; v*c s]
        Bs[ii][start_idx+3:start_idx+4, 1:2] = [1 0; 0 1]
    end
    # Combine the As into one large A matrix and add in the zeroth order term of the Taylor expansion.
    A = I + Matrix(blockdiag(As...))

    return LinearDynamics(A, Bs, dyn.sys_info)
end

export UnicycleDynamics, propagate_dynamics, linearize_dynamics
