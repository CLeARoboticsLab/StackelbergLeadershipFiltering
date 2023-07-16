using StackelbergControlHypothesesFiltering

include("PassingScenarioConfig.jl")

function create_passing_scenario_dynamics(num_players, sample_time)
    return UnicycleDynamics(num_players, sample_time)
end

# Specify some constants.
const NUM_PASSING_SCENARIO_SUBCOSTS = 14

# A helper to combine functions into a weighted sum.
function combine_cost_funcs(funcs, weights)
    @assert size(funcs) == size(weights)

    # Create a weighted cost function ready for use with autodiff.
    g(si, x, us, t) = begin
        f_eval = [f(si, x, us, t) for f in funcs]
        return weights' * f_eval
    end

    return g
end

function create_passing_scenario_costs(cfg, si, w_p1, w_p2, goal_p1, goal_p2)
    @assert length(w_p1) == NUM_PASSING_SCENARIO_SUBCOSTS
    @assert length(w_p2) == NUM_PASSING_SCENARIO_SUBCOSTS

    # Passing Scenario cost definition
    # 1. distance to goal (no control cost)

    # TODO(hamzah) - should this be replaced with a tracking trajectory cost?
    const_1 = 1.
    Q1 = zeros(8, 8)
    Q1[1, 1] = const_1 * 1.
    Q1[2, 2] = const_1 * 0.
    Q1[3, 3] = const_1 * 1.
    Q1[4, 4] = const_1 * 1.
    q_cost1 = QuadraticCost(Q1)
    add_control_cost!(q_cost1, 1, zeros(udim(si, 1), udim(si, 1)))
    add_control_cost!(q_cost1, 2, zeros(udim(si, 2), udim(si, 2)))
    c1a = QuadraticCostWithOffset(q_cost1, goal_p1)

    Q2 = zeros(8, 8)
    Q2[5, 5] = const_1 * 1.
    Q2[6, 6] = const_1 * 0.
    Q2[7, 7] = const_1 * 1.
    Q2[8, 8] = const_1 * 1.
    q_cost2 = QuadraticCost(Q2)
    add_control_cost!(q_cost2, 1, zeros(udim(si, 1), udim(si, 1)))
    add_control_cost!(q_cost2, 2, zeros(udim(si, 2), udim(si, 2)))
    c2a = QuadraticCostWithOffset(q_cost2, goal_p2)

    # 2. avoid collisions
    avoid_collisions_cost_fn(si, x, us, t) = begin
        # This log barrier avoids agents getting within some configured radius of one another.
       return - log(norm([1 1 0 0 -1 -1 0 0] * x, cfg.dist_norm_order) - cfg.collision_radius_m)
    end

    c1b = PlayerCost(avoid_collisions_cost_fn, si)
    c2b = PlayerCost(avoid_collisions_cost_fn, si)

    # 3. enforce speed limit and turning limit
    c1c_i = AbsoluteLogBarrierCost(4, cfg.speed_limit_mps, false)
    c1c_ii = AbsoluteLogBarrierCost(4, -cfg.speed_limit_mps, true)

    c1c_iii = AbsoluteLogBarrierCost(3, cfg.θ₀+cfg.max_heading_deviation, false)
    c1c_iv = AbsoluteLogBarrierCost(3, cfg.θ₀-cfg.max_heading_deviation, true)

    c2c_i = AbsoluteLogBarrierCost(8, cfg.speed_limit_mps, false)
    c2c_ii = AbsoluteLogBarrierCost(8, -cfg.speed_limit_mps, true)
    c2c_iii = AbsoluteLogBarrierCost(7, cfg.θ₀+cfg.max_heading_deviation, false)
    c2c_iv = AbsoluteLogBarrierCost(7, cfg.θ₀-cfg.max_heading_deviation, true)

    # 4, 5. minimize and bound control effort - acceleration should be easier than rotation
    c1de = QuadraticCost(zeros(8, 8), zeros(8), 0.)
    R11 = [1. 0; 0 0.1]
    add_control_cost!(c1de, 1, R11)
    add_control_cost!(c1de, 2, zeros(udim(si, 2), udim(si, 2)))


    max_rotvel = cfg.max_rotational_velocity_radps
    max_accel = cfg.max_acceleration_mps

    c1de_i = AbsoluteLogBarrierControlCost(1, [1.; 0.], max_rotvel, false)
    c1de_ii = AbsoluteLogBarrierControlCost(1, [1.; 0.], -max_rotvel, true)
    c1de_iii = AbsoluteLogBarrierControlCost(1, [0.; 1.], max_accel, false)
    c1de_iv = AbsoluteLogBarrierControlCost(1, [0.; 1.], -max_accel, true)

    c2de = QuadraticCost(zeros(8, 8))
    R22 = [1. 0; 0 0.1]
    add_control_cost!(c2de, 2, R22)
    add_control_cost!(c2de, 1, zeros(udim(si, 1), udim(si, 1)))

    c2de_i = AbsoluteLogBarrierControlCost(2, [1.; 0.], max_rotvel, false)
    c2de_ii = AbsoluteLogBarrierControlCost(2, [1.; 0.], -max_rotvel, true)
    c2de_iii = AbsoluteLogBarrierControlCost(2, [0.; 1.], max_accel, false)
    c2de_iv = AbsoluteLogBarrierControlCost(2, [0.; 1.], -max_accel, true)

    # 6. log barriers on the x dimension ensure that the vehicles don't exit the road
    # TODO(hamzah) - remove assumption of straight road
    c1f_i = AbsoluteLogBarrierCost(1, get_right_lane_boundary_x(cfg), false)
    c1f_ii = AbsoluteLogBarrierCost(1, get_left_lane_boundary_x(cfg), true)

    c2f_i = AbsoluteLogBarrierCost(5, get_right_lane_boundary_x(cfg), false)
    c2f_ii = AbsoluteLogBarrierCost(5, get_left_lane_boundary_x(cfg), true)

    # 7. Add a bump in cost for crossing the centerline.
    c1g = GaussianCost([1], [get_center_line_x(cfg)], ones(1, 1))
    c2g = GaussianCost([5], [get_center_line_x(cfg)], ones(1, 1))

    costs_p1 = [c1a,
                c1b,
                c1c_i,
                c1c_ii,
                c1c_iii,
                c1c_iv,
                c1de,
                c1de_i,
                c1de_ii,
                c1de_iii,
                c1de_iv,
                c1f_i,
                c1f_ii,
                c1g]
    @assert length(costs_p1) == NUM_PASSING_SCENARIO_SUBCOSTS
    

    costs_p2 = [c2a,
                c2b,
                c2c_i,
                c2c_ii,
                c2c_iii,
                c2c_iv,
                c2de,
                c2de_i,
                c2de_ii,
                c2de_iii,
                c2de_iv,
                c2f_i,
                c2f_ii,
                c2g]
     @assert length(costs_p2) == NUM_PASSING_SCENARIO_SUBCOSTS

    g1 = combine_cost_funcs(get_as_function.(costs_p1), w_p1)
    sum_cost_p1 = PlayerCost(g1, si)

    g2 = combine_cost_funcs(get_as_function.(costs_p2), w_p2)
    sum_cost_p2 = PlayerCost(g2, si)

    return [sum_cost_p1, sum_cost_p2]
end
