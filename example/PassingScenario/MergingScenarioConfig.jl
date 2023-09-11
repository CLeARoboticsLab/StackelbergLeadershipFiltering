using Parameters

# This file contains configuration variables related to the road and constraints.

# The configuration defines a straight two-way lane that defines a right-hand coordinate system where y points in the
# direction of the lane, and x points to the right side of the road.
DEFAULT_LANE_WIDTH = 2.3
DEFAULT_CENTER_LINE_X = 0.
Y_DIRECTION_HEADING = pi/2
DEFAULT_MAX_HEADING_DEVIATION = pi/3

DEFAULT_MAX_ROTVEL_RADPS = 2. # arbitrarily chosen
DEFAULT_MAX_ACCEL_MPS = 9.  # this corresponds to 0 to 60 in 2.5s, which is close to the record

DEFAULT_NORM_ORDER = 2

@with_kw struct MergingScenarioConfig{TF, TI}
    # Specify the lane size and dimensions.
    lane_width_m::TF = DEFAULT_LANE_WIDTH
    # cl_x::TF = DEFAULT_CENTER_LINE_X

    # Length of separated two lane region.
    region1_length_m::TF = 30.

    # Length of merging section.
    region2_length_m::TF = 30.

    # Define legal constraints on state.
    speed_limit_mps::TF = 25.
    θ₀::TF = Y_DIRECTION_HEADING
    max_heading_deviation::TF = DEFAULT_MAX_HEADING_DEVIATION # max difference from θ₀

    # radius of circle defined to cause collision
    collision_radius_m::TF = 0.
    dist_norm_order::TI = DEFAULT_NORM_ORDER

    # control constraints
    max_rotational_velocity_radps::TF = DEFAULT_MAX_ROTVEL_RADPS
    max_acceleration_mps::TF = DEFAULT_MAX_ACCEL_MPS
end

# function get_center_line_x(cfg::PassingScenarioConfig)
#     return cfg.cl_x
# end

# function get_left_lane_boundary_x(cfg::PassingScenarioConfig)
#     return cfg.cl_x - cfg.lane_width_m
# end

# function get_right_lane_boundary_x(cfg::PassingScenarioConfig)
#     return cfg.cl_x + cfg.lane_width_m
# end

function get_validator(si, cfg::MergingScenarioConfig)
    function check_valid(xs, us, ts; p1_on_left)
        T = length(ts)
        for tt in 1:T
            x = xs[:, tt]
            u1 = us[1][:, tt]
            u2 = us[2][:, tt]

            no_collision = norm([1 1 0 0 -1 -1 0 0] * x, cfg.dist_norm_order) > cfg.collision_radius_m

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

            p1_within_lane_lines = x[1] > lower_bound && x[1] < upper_bound
            if dist_along_lane ≤ L₁
                p2_within_lane_lines = x[5] > -upper_bound && x[5] < -lower_bound
            else
                p2_within_lane_lines = x[5] > lower_bound && x[5] < upper_bound
            end

            within_speed_limit = abs(x[4]) < cfg.speed_limit_mps && abs(x[8]) < cfg.speed_limit_mps
            within_heading_limit = abs(x[3] - cfg.θ₀) < cfg.max_heading_deviation && abs(x[7] - cfg.θ₀) < cfg.max_heading_deviation
            within_angvel_limits = abs(u1[1]) < cfg.max_rotational_velocity_radps && abs(u2[1]) < cfg.max_rotational_velocity_radps
            within_accel_limits = abs(u1[2]) < cfg.max_acceleration_mps && abs(u2[2]) < cfg.max_acceleration_mps

            satisfies_all_constraints_at_tt = (no_collision && p1_within_lane_lines && p2_within_lane_lines 
                                               && within_speed_limit  && within_heading_limit
                                               && within_angvel_limits && within_accel_limits)
            if !satisfies_all_constraints_at_tt
                println("$(tt) - no collision: $(no_collision) $(norm([1 1 0 0 -1 -1 0 0] * x, cfg.dist_norm_order)), 
                                 lane_lines (p1): $(p1_within_lane_lines) $(x[1]),
                                 lane_lines (p2): $(p2_within_lane_lines) $(x[5]),
                                 speed limit: $(within_speed_limit) $(x[4]) $(x[8]),
                                 heading_limit: $(within_heading_limit) $(x[3]) $(x[7]),
                                 rotvel limit: $(within_angvel_limits) $(u1[1]) $(u2[1]),
                                 accel. limit: $(within_accel_limits) $(u1[2]) $(u2[2])")

                # return false
            end
        end
        return true
    end
    return check_valid
end
