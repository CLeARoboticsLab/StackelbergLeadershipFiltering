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
function get_homogenized_state_dynamics_matrix(dyn::Dynamics)
    return homogenize_dynamics_matrix(dyn.A; m=dyn.a)
end

function get_homogenized_control_dynamics_matrix(dyn::Dynamics, player_idx::Int)
    return homogenize_dynamics_matrix(dyn.Bs[player_idx])
end

export get_homogenized_state_dynamics_matrix, get_homogenized_control_dynamics_matrix


# Get the linear term.
function get_linear_dynamics_term(dyn::Dynamics)
    return dyn.A
end

function get_constant_dynamics_term(dyn::Dynamics)
    return dyn.a
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
    x_next = dyn.A * x + dyn.a + sum(dyn.Bs[ii] * us[ii] for ii in 1:N)

    if dyn.D != nothing && v != nothing
        x_next += dyn.D * v
    end

    # Remove the extra dimension before returning the propagated state.
    return x_next
end

function linearize_dynamics(dyn::LinearDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return dyn
end

export LinearDynamics, propagate_dynamics, linearize_dynamics
