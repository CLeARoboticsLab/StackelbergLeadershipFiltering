# TODO(hamzah) Add better tests for the LinearDynamics struct and associated functions.
# Describes the dynamics x_{t+1} = Ax_t + a + ∑Buᵢ_t + Dv_t; x = state, u = inputs, v = process noise
struct LinearDynamics <: Dynamics
    A  # linear state dynamics term
    a  # constant state dynamics term
    Bs # controls
    D  # process noise
    sys_info::SystemInfo
end

# TODO(hamzah) Add [:,:] as necessary for auto-sizing - fixes bug if 1D vector passed in when a 2D matrix is expected.
# Constructor for linear dynamics that auto-generates the system info and has no process noise.
# TODO(hamzah): Add an optional bs argument that defaults to zero vector of proper size.
LinearDynamics(A, Bs; a=zeros(size(A, 1))) = LinearDynamics(A, a, Bs, nothing,
                                                SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)]))

# Constructor for linear dynamics that is provided the system info and has no process noise.
LinearDynamics(A, Bs, sys_info::SystemInfo; a=zeros(size(A, 1))) = LinearDynamics(A, a, Bs, nothing, sys_info)

# Constructor for linear dynamics that auto-generates the system info with process noise.
LinearDynamics(A, Bs, D; a=zeros(size(A, 1))) = LinearDynamics(A, a, Bs, D,
                                                               SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)], size(D, 2)))

# Helpers that get the homogenized A and B matrices.
function get_homogenized_state_dynamics_matrix(dyn::LinearDynamics)
    return homogenize_dynamics_matrix(dyn.A; m=dyn.a)
end

function get_homogenized_control_dynamics_matrix(dyn::LinearDynamics, player_idx::Int)
    return homogenize_dynamics_matrix(dyn.Bs[player_idx])
end

export get_homogenized_state_dynamics_matrix, get_homogenized_control_dynamics_matrix


# Get the linear term.
function get_linear_state_dynamics(dyn::LinearDynamics)
    return dyn.A
end

function get_constant_state_dynamics(dyn::LinearDynamics)
    return dyn.a
end

function get_control_dynamics(dyn::LinearDynamics, player_idx::Int)
    return dyn.Bs[player_idx]
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

function propagate_dynamics(dyn::LinearDynamics,
                            time_range,
                            x::AbstractVector{Float64},
                            us::AbstractVector{<:AbstractVector{Float64}},
                            v::Union{Nothing, AbstractVector{Float64}})
    N = num_agents(dyn)

    # Assertions to confirm sizes.
    @assert size(x, 1) == xdim(dyn)
    for ii in 1:N
        @assert size(us[ii], 1) == udim(dyn, ii)
    end

    # Incorporate the dynamics based on the state and the controls.
    dt = time_range[2] - time_range[1]
    x_next = dyn.A * x + dyn.a * dt + sum(dyn.Bs[ii] * us[ii] for ii in 1:N)

    if dyn.D != nothing && v != nothing
        x_next += dyn.D * v
    end

    # Remove the extra dimension before returning the propagated state.
    return x_next
end

function linearize_dynamics(dyn::LinearDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return dyn
end


# These are the continuous derivative matrices of the f function.
function Fx(dyn::LinearDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return dyn.A - I
end

function Fus(dyn::LinearDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return dyn.Bs
end

# TODO(hmzh) - may need dfdv as well


export LinearDynamics, propagate_dynamics, linearize_dynamics, Fx, Fus

using Plots
# TODO(hamzah) - refactor this to be tied DoubleIntegrator Dynamics instead of Linear Dynamics.
function plot_states_and_controls(dyn::LinearDynamics, times, xs, us)
    @assert num_agents(dyn) == 2
    @assert xdim(dyn) == 8
    @assert udim(dyn, 1) == 2
    @assert udim(dyn, 2) == 2
    x₁ = xs[:, 1]

    title1 = "pos. traj."
    q1 = plot(legend=:outertopright, title=title1, xlabel="x (m)", ylabel="y (m)")
    plot!(q1, xs[1, :], xs[3, :], label="P1 pos")
    plot!(q1, xs[5, :], xs[7, :], label="P2 pos")

    q1 = scatter!([x₁[1]], [x₁[3]], color="blue", label="start P1")
    q1 = scatter!([x₁[5]], [x₁[7]], color="red", label="start P2")

    title2a = "x-pos"
    q2a = plot(legend=:outertopright, title=title2a, xlabel="t (s)", ylabel="x (m)")
    plot!(times, xs[1,:], label="P1 px")
    plot!(times, xs[5,:], label="P2 px")

    title2b = "y-pos"
    q2b = plot(legend=:outertopright, title=title2b, xlabel="t (s)", ylabel="y (m)")
    plot!(times, xs[3,:], label="P1 py")
    plot!(times, xs[7,:], label="P2 py")

    title3 = "x-vel"
    q3 = plot(legend=:outertopright, title=title3, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[2,:], label="P1 vx")
    plot!(times, xs[6,:], label="P2 vx")

    title4 = "y-vel"
    q4 = plot(legend=:outertopright, title=title4, xlabel="t (s)", ylabel="vel. (mps)")
    plot!(times, xs[4,:], label="P1 vy")
    plot!(times, xs[8,:], label="P2 vy")

    title5 = "x-accel"
    q5 = plot(legend=:outertopright, title=title5, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][1, :], label="P1 ax")
    plot!(times, us[2][1, :], label="P2 ax")

    title6 = "y-accel"
    q6 = plot(legend=:outertopright, title=title6, xlabel="t (s)", ylabel="accel (mpss)")
    plot!(times, us[1][2, :], label="P1 ay")
    plot!(times, us[2][2, :], label="P2 ay")

    return q1, q2a, q2b, q3, q4, q5, q6
end

export plot_states_and_controls

