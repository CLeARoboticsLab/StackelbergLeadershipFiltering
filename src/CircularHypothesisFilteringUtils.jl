# CircularHypothesisFilteringUtils.jl contains functions for setting up the circular reference controls.

# state has size 8 []

# TODO(hamzah): Add a boolean flag that implements adding noise.
function dyn_ode(x, u)
    return [x[4] * cos(x[3]),
            x[4] * sin(x[3]),
            u[1],
            u[2]]
end

function rk4_dynamics(timestep, x, u)
    k1 = dyn_ode(x, u)
    k2 = dyn_ode(x + 0.5 * k1, u)
    k3 = dyn_ode(x + 0.5 * k2, u)
    k4 = dyn_ode(x + k3, u)
    return x + timestep * (k1 + 2 * (k2 + k3) + k4) / 6
end

# Gets a circular reference trajectory CCW around the origin - function allows specification of angles.
# Defaults to 10 meter radius.
function get_circular_reference_trajectory(horizon, num_states, num_ctrls; timestep_s=0.1, radius_m=10)

    # Compute useful constants.
    desired_speed = 1
    desired_ang_rate = desired_speed / radius_m

    # Initialize the references.
    ref_traj = zeros((num_states, horizon))
    ref_ctrls = cat(desAngRate * ones((1, horizon)),
                                zeros((1, horizon)); dims=1)

    # Start at the leftmost point of the circle pointed upward.
    ref_traj[:, 1] = [radius_m, 0, π/2, 0]

     # Initial controls - accelerate to desired speed and angular rate in one step.
    ref_ctrls[:, 1] = [desired_ang_rate, desired_speed / timestep_s]

    # Propgate forward the circular dynamics to fill the reference states.
    for tt in 1:horizon
        ref_traj[t + 1] = rk4_dynamics(timestep_s, refTraj[t], refCtrls[t])
    end

    return ref_traj, ref_ctrls
end

# Linearizes a dynamics matrix around a reference trajectory at time t.
function construct_A(x_ref, timestep_s)
    cos_ref = cos(x_ref[2])
    sin_ref = sin(x_ref[2])
    return [[1, 0, -x_ref[3] * sin_ref * timestep_s, cos_ref * timestep_s],
            [0, 1,  x_ref[3] * cos_ref * timestep_s, sin_ref * timestep_s],
            [0, 0,                                1,                    0],
            [0, 0,                                0,                    1]]
end

# Linearizes a dynamics matrix around reference trajectories at time t.
function construct_big_A(x_ref_1, x_ref_2, timestep_s)
    size_1 = size(x_ref_1, 1)
    size_2 = size(x_ref_2, 1)
    joint_size = size_1 + size_2
    A = zeros((joint_size, joint_size))

    A[1:size_1, 1:size_1] = construct_A(x_ref_1, timestep);
    A[size_1+1:joint_size, size_1+1:joint_size] = construct_A(x_ref_2, timestep);

    return A;
end

# Linearizes a control matrix around a reference trajectory at time t.
export construct_B
function construct_B(x_ref, timestep_s)
    return [[         0,          0],
            [         0,          0],
            [timestep_s,          0],
            [         0, timestep_s]];
end

# Linearizes a control matrix around a reference trajectory at time t.
export construct_big_B
function construct_big_B(x_ref, timestep_s)
    B = [[         0,          0],
         [         0,          0],
         [timestep_s,          0],
         [         0, timestep_s]];
    return repmatnd(B, 2, 1);
end
