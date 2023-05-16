# This file defines a dynamics object that has some amount of history. The dynamics propagation for these historical
# states is as follows:
# X_t = [x_{t-T}; x_{t-T+1}; ...; x_t]
# X_{t+1} = f(X_t, u_t) = [x_{t-T+1}; ...; x_t; f(x_t, u_t)]

struct DynamicsWithHistory <: Dynamics
    dyn::Dynamics        # The underlying dynamics that we are tracking.
    num_hist::Int        # Number of historical states to track.
    sys_info::SystemInfo # Information about the system
end
DynamicsWithHistory(dyn::Dynamics, num_hist::Int) = DynamicsWithHistory(dyn, num_hist, SystemInfo(num_agents(dyn),
                                                                                                  num_hist * xdim(dyn),
                                                                                                  [udim(dyn, ii) for ii in 1:num_agents(dyn)]))
# Define helpers.
function get_underlying_dynamics(dyn::DynamicsWithHistory)
    return dyn.dyn
end

# hist_idx is the index of the desired state in the history with 1 as the current state.
function get_state(dyn::DynamicsWithHistory, X::AbstractVector{Float64}, hist_idx::Int)
    start_idx = (dyn.num_hist - hist_idx) * xdim(dyn.dyn) + 1
    end_idx = (dyn.num_hist - hist_idx + 1) * xdim(dyn.dyn)
    @assert start_idx > 0
    @assert end_idx <= xdim(dyn)
    return X[start_idx:end_idx]
end

function get_current_state(dyn::DynamicsWithHistory, X::AbstractVector{Float64})
    return get_state(dyn, X, 1)
end

export get_underlying_dynamics, get_state, get_current_state


function propagate_dynamics(dyn::DynamicsWithHistory, time_range, X::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    x_t = get_current_state(dyn, X)
    num_states = xdim(dyn.dyn)
    num_states_w_hist = xdim(dyn)

    # Shift first num_hist-1 states down into the history and propagate the dynamics to get the new one.
    end_idx = (dyn.num_hist - 1) * num_states
    X_new = vcat(X[num_states+1:num_states+end_idx],
                 propagate_dynamics(dyn.dyn, time_range, x_t, us))

    @assert all(X[num_states+1:num_states+end_idx] .== X_new[1:end_idx])
    @assert size(X_new, 1) == num_states_w_hist
    return X_new
end

# These are the continuous derivative matrices of the f function. It does not include the derivatives of previous states.
function Fx(dyn::DynamicsWithHistory, time_range, X::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    x_t = get_current_state(dyn, X)
    num_states = xdim(dyn.dyn)
    num_states_w_hist = xdim(dyn)

    out = zeros(num_states_w_hist, num_states_w_hist)

    # Get the derivative of the actual transition of the last state.
    start_idx = (dyn.num_hist - 1) * num_states + 1
    end_idx = dyn.num_hist * num_states
    out[start_idx:end_idx, start_idx:end_idx] = Fx(dyn.dyn, time_range, x_t, us)

    return out
end

function Fus(dyn::DynamicsWithHistory, time_range, X::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    x_t = get_current_state(dyn, X)
    num_states = xdim(dyn.dyn)
    num_states_w_hist = xdim(dyn)

    dfdus = Fus(dyn.dyn, time_range, x_t, us)
    return [vcat(zeros(num_states_w_hist-num_states, udim(dyn, ii)), dfdus[ii]) for ii in 1:num_agents(dyn.dyn)]
end

# We define a custom linearize_dynamics function for this object because this is a pseudo-dynamics object and would not
# work properly with the generally defined one.
function linearize_dynamics(dyn::DynamicsWithHistory, time_range, X::AbstractVector{Float64}, us::AbstractVector{<:AbstractVector{Float64}})
    x_t = get_current_state(dyn, X)
    num_states = xdim(dyn.dyn)
    num_states_w_hist = xdim(dyn)

    # Get the first derivative matrix to adjust.
    A = Fx(dyn, time_range, X, us)

    # Produce matrix that shifts history forward one time step.
    nm1_dims = (dyn.num_hist - 1) * num_states
    A[1:nm1_dims, num_states+1:num_states_w_hist] = eye(nm1_dims)

    Bs = Fus(dyn, time_range, X, us)

    return LinearDynamics(A, Bs)
end

export propagate_dynamics, linearize_dynamics, Fx, Fus
