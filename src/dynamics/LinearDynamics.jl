# TODO(hamzah) Add better tests for the LinearDynamics struct and associated functions.
struct LinearDynamics <: Dynamics
    A  # linear state; can have a constant vector term attached, of dimension n+1 by n+1
    Bs # controls
    D  # process noise
    sys_info::SystemInfo
end

# TODO(hamzah) Add [:,:] as necessary for auto-sizing - fixes bug if 1D vector passed in when a 2D matrix is expected.
# Constructor for linear dynamics that auto-generates the system info and has no process noise.
# TODO(hamzah): Add an optional bs argument that defaults to zero vector of proper size.
LinearDynamics(A, Bs; a=zeros(size(A, 1))) = LinearDynamics(
                                                homogenize_dynamics_matrix(A; m=a),
                                                [homogenize_dynamics_matrix(B) for B in Bs],
                                                nothing,
                                                SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)]))

# Constructor for linear dynamics that is provided the system info and has no process noise.
LinearDynamics(A, Bs, sys_info::SystemInfo; a=zeros(size(A, 1))) = LinearDynamics(
                                                                    homogenize_dynamics_matrix(A; m=a),[
                                                                    homogenize_dynamics_matrix(B) for B in Bs],
                                                                    nothing,
                                                                    sys_info)

# Constructor for linear dynamics that auto-generates the system info with process noise.
LinearDynamics(A, Bs, D; a=zeros(size(A, 1))) = LinearDynamics(
                                                    homogenize_dynamics_matrix(A; m=a),
                                                    [homogenize_dynamics_matrix(B) for B in Bs],
                                                    D,
                                                    SystemInfo(length(Bs), last(size(A)), [last(size(Bs[i])) for i in 1:length(Bs)], size(D, 2)))

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

    xh = homogenize_state(dyn, x)
    uhs = homogenize_ctrls(dyn, us)

    # Assertions to confirm sizes.
    @assert size(xh, 1) == xhdim(dyn)
    for ii in 1:N
        @assert size(uhs[ii], 1) == uhdim(dyn, ii)
    end

    xh_next = dyn.A * xh
    for i in 1:N
        uhi = reshape(uhs[i], uhdim(dyn, i))
        xh_next += dyn.Bs[i] * uhi
    end

    if dyn.D != nothing && v != nothing
        xh_next += dyn.D * v
    end

    # Remove the extra dimension before returning the propagated state.
    return xh_next[1:xdim(dyn)]
end

function linearize_dynamics(dyn::LinearDynamics, time_range, x::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    return dyn
end

export LinearDynamics, propagate_dynamics, linearize_dynamics
