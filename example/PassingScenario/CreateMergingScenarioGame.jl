using StackelbergControlHypothesesFiltering

include("MergingScenarioConfig.jl")

function create_merging_scenario_dynamics(num_players, sample_time)
    return UnicycleDynamics(num_players, sample_time)
end

# Specify some constants.
const NUM_MERGING_SCENARIO_SUBCOSTS = 13

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

function create_merging_scenario_costs(cfg::MergingScenarioConfig, si, w_p1, w_p2, goal_p1, goal_p2; large_number=1e6, p1_on_left=true)
    @assert length(w_p1) == NUM_MERGING_SCENARIO_SUBCOSTS
    @assert length(w_p2) == NUM_MERGING_SCENARIO_SUBCOSTS

    # Passing Scenario cost definition
    # 1. distance to goal (no control cost)

    # TODO(hamzah) - should this be replaced with a tracking trajectory cost?
    const_1 = 1.
    Q1 = zeros(8, 8)
    Q1[1, 1] = const_1 * 0.
    Q1[2, 2] = const_1 * 0.5#2.
    Q1[3, 3] = const_1 * 1.
    Q1[4, 4] = const_1 * 1.
    q_cost1 = QuadraticCost(Q1)
    add_control_cost!(q_cost1, 1, zeros(udim(si, 1), udim(si, 1)))
    add_control_cost!(q_cost1, 2, zeros(udim(si, 2), udim(si, 2)))
    c1a = QuadraticCostWithOffset(q_cost1, goal_p1)

    Q2 = zeros(8, 8)
    Q2[5, 5] = const_1 * 0.
    Q2[6, 6] = const_1 * 0.5
    Q2[7, 7] = const_1 * 1.
    Q2[8, 8] = const_1 * 1.
    q_cost2 = QuadraticCost(Q2)
    add_control_cost!(q_cost2, 1, zeros(udim(si, 1), udim(si, 1)))
    add_control_cost!(q_cost2, 2, zeros(udim(si, 2), udim(si, 2)))
    c2a = QuadraticCostWithOffset(q_cost2, goal_p2)


    merging_trajectory_position_cost_p1(si, x, us, t) = begin
        dist_along_lane = x[2]
        L₁ = cfg.region1_length_m 
        L₂ = cfg.region2_length_m 
        w = cfg.lane_width_m

        if dist_along_lane ≤ L₁ # separate lanes
            goal_pos = (p1_on_left) ? -w/2 : w/2
        # elseif dist_along_lane ≤ L₁ + L₂ # linear progression to smaller lane
        #     lin_width_at_dist_proportion = 1 - (dist_along_lane - L₁)/L₂
        #     goal_pos = -w/4 - lin_width_at_dist_proportion * w/4
        else # in final region 
            goal_pos = 0.
        end

        return 1//2 * (x[1] - goal_pos)^2
    end

    merging_trajectory_position_cost_p2(si, x, us, t) = begin
        dist_along_lane = x[6]
        L₁ = cfg.region1_length_m 
        L₂ = cfg.region2_length_m 
        w = cfg.lane_width_m

        if dist_along_lane ≤ L₁ # separate lanes
            goal_pos = (p1_on_left) ? w/2 : -w/2
            # goal_pos = w/2
        # elseif dist_along_lane ≤ L₁ + L₂ # linear progression to smaller lane
        #     lin_width_at_dist_proportion = 1 - (dist_along_lane - L₁)/L₂
        #     goal_pos = w/4 + lin_width_at_dist_proportion * w/4
        else # in final region 
            goal_pos = 0.
        end

        return 1//2 * (x[5] - goal_pos)^2
    end

    c1a_i = PlayerCost(merging_trajectory_position_cost_p1, si)
    c2a_i = PlayerCost(merging_trajectory_position_cost_p2, si)

    # 2. avoid collisions
    avoid_collisions_cost_fn(si, x, us, t) = begin
        # This log barrier avoids agents getting within some configured radius of one another.
        # TODO(hamzah) - this may accidentally be using 1-norm.
        dist_to_boundary = norm([1 1 0 0 -1 -1 0 0] * x, cfg.dist_norm_order) - cfg.collision_radius_m
        return (dist_to_boundary ≤ 0) ? large_number : -log(dist_to_boundary)
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
    R11 = [1. 0; 0 1.]
    Z22 = zeros(udim(si, 1), udim(si, 1))
    add_control_cost!(c1de, 1, Z22)
    add_control_cost!(c1de, 2, Z22)

    max_rotvel = cfg.max_rotational_velocity_radps
    max_accel = cfg.max_acceleration_mps

    c1de_i = AbsoluteLogBarrierControlCost(1, [1.; 0.], max_rotvel, false)
    c1de_ii = AbsoluteLogBarrierControlCost(1, [1.; 0.], -max_rotvel, true)
    c1de_iii = AbsoluteLogBarrierControlCost(1, [0.; 1.], max_accel, false)
    c1de_iv = AbsoluteLogBarrierControlCost(1, [0.; 1.], -max_accel, true)

    c2de = QuadraticCost(zeros(8, 8))
    R22 = [1. 0; 0 1.]
    add_control_cost!(c2de, 2, Z22)
    add_control_cost!(c2de, 1, Z22)

    c2de_i = AbsoluteLogBarrierControlCost(2, [1.; 0.], max_rotvel, false)
    c2de_ii = AbsoluteLogBarrierControlCost(2, [1.; 0.], -max_rotvel, true)
    c2de_iii = AbsoluteLogBarrierControlCost(2, [0.; 1.], max_accel, false)
    c2de_iv = AbsoluteLogBarrierControlCost(2, [0.; 1.], -max_accel, true)

    # 6. log barriers on the x dimension ensure that the vehicles don't exit the road
    # TODO(hamzah) - remove assumption of straight road
    stay_within_lanes_p1(si, x, us, t) = begin
        dist_along_lane = x[2]
        L₁ = cfg.region1_length_m 
        L₂ = cfg.region2_length_m 
        w = cfg.lane_width_m

        if dist_along_lane ≤ L₁ # separate lanes
            upper_bound = (p1_on_left) ? 0. : w
            lower_bound = (p1_on_left) ? -w : 0.
        elseif dist_along_lane ≤ L₁ + L₂ # linear progression to smaller lane
            lin_width_at_dist_proportion = 1 - (dist_along_lane - L₁)/L₂
            upper_bound = w/2 + lin_width_at_dist_proportion * w/2
            lower_bound = -w/2 - lin_width_at_dist_proportion * w/2
        else # in final region 
            upper_bound = w/2
            lower_bound = -w/2
        end

        # println(x[1], " ", lower_bound," ", upper_bound)
        violates_bound = x[1] ≥ upper_bound || x[1] ≤ lower_bound
        return violates_bound ? large_number : -log(upper_bound-x[1]) -log(x[1]-lower_bound)
    end

    stay_within_lanes_p2(si, x, us, t) = begin
        dist_along_lane = x[6]
        L₁ = cfg.region1_length_m 
        L₂ = cfg.region2_length_m 
        w = cfg.lane_width_m

        if dist_along_lane ≤ L₁ # separate lanes
            upper_bound = (p1_on_left) ? w : 0.
            lower_bound = (p1_on_left) ? 0. : -w
        elseif dist_along_lane ≤ L₁ + L₂ # linear progression to smaller lane
            lin_width_at_dist_proportion = 1 - (dist_along_lane - L₁)/L₂
            upper_bound = w/2 + lin_width_at_dist_proportion * w/2
            lower_bound = -w/2 - lin_width_at_dist_proportion * w/2
        else # in final region 
            upper_bound = w/2
            lower_bound = -w/2
        end

        violates_bound = x[5] ≥ upper_bound || x[5] ≤ lower_bound
        return violates_bound ? large_number : -log(upper_bound-x[5]) -log(x[5]-lower_bound)
    end

    c1f = PlayerCost(stay_within_lanes_p1, si)
    c2f = PlayerCost(stay_within_lanes_p2, si)

    # player 1 stays ahead of player 2
    stay_ahead_cost(si, x, us, t) = begin
        y1 = x[2]
        y2 = x[6]
        dy = y2 - y1
        return 0.5 * (tanh(dy) + 1) * dy^2
    end
    c1g = PlayerCost(stay_ahead_cost, si)


    costs_p1 = [c1a,
                c1a_i,
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
                c1f]
    @assert length(costs_p1) == NUM_MERGING_SCENARIO_SUBCOSTS
    

    costs_p2 = [c2a,
                c2a_i,
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
                c2f]
     @assert length(costs_p2) == NUM_MERGING_SCENARIO_SUBCOSTS

    g1 = combine_cost_funcs(get_as_function.(costs_p1), w_p1)
    sum_cost_p1 = PlayerCost(g1, si)

    g2 = combine_cost_funcs(get_as_function.(costs_p2), w_p2)
    sum_cost_p2 = PlayerCost(g2, si)

    return [sum_cost_p1, sum_cost_p2]
end
