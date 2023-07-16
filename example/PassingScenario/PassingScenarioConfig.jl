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

@with_kw struct PassingScenarioConfig{TF, TI}
    # Specify the lane size and dimensions.
    lane_width_m::TF = DEFAULT_LANE_WIDTH
    cl_x::TF = DEFAULT_CENTER_LINE_X

    # Define legal constraints on state.
    speed_limit_mps::TF = 35.
    θ₀::TF = Y_DIRECTION_HEADING
    max_heading_deviation::TF = DEFAULT_MAX_HEADING_DEVIATION # max difference from θ₀

    # radius of circle defined to cause collision
    collision_radius_m::TF = 0.
    dist_norm_order::TI = DEFAULT_NORM_ORDER

    # control constraints
    max_rotational_velocity_radps::TF = DEFAULT_MAX_ROTVEL_RADPS
    max_acceleration_mps::TF = DEFAULT_MAX_ACCEL_MPS
end

function get_center_line_x(cfg::PassingScenarioConfig)
    return cfg.cl_x
end

function get_left_lane_boundary_x(cfg::PassingScenarioConfig)
    return cfg.cl_x - cfg.lane_width_m
end

function get_right_lane_boundary_x(cfg::PassingScenarioConfig)
    return cfg.cl_x + cfg.lane_width_m
end

