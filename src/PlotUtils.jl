# TODO(hamzah) Implement the following plots in here if not trivial. To be rolled into a plots package for the lab later on.
# - probabilites (multiple actors)
# - state vs. time (multiple actors)
# - position in two dimensions (multiple actors)

# // Plots the trajectories on a plane.
# void plot_trajectory(
#     const std::vector<Vector> &xs,
#     const std::vector<Vector> &reference_trajectory1,
#     const double radius1_m,
#     const std::vector<Vector> &reference_trajectory2,
#     const double radius2_m)
# {
#     if (xs.size() != reference_trajectory1.size() || xs.size() != reference_trajectory2.size())
#     {
#         throw std::runtime_error("inconsistent sizes for plot_trajectory.");
#     }

#     // Construct x,y vectors to plot.
#     auto x1_coords = std::vector<double>(xs.size());
#     auto y1_coords = std::vector<double>(xs.size());
#     auto x2_coords = std::vector<double>(xs.size());
#     auto y2_coords = std::vector<double>(xs.size());
#     auto x_ref1_coords = std::vector<double>(reference_trajectory1.size());
#     auto y_ref1_coords = std::vector<double>(reference_trajectory1.size());
#     auto x_ref2_coords = std::vector<double>(reference_trajectory2.size());
#     auto y_ref2_coords = std::vector<double>(reference_trajectory2.size());
#     for (auto i = 0ul; i < xs.size(); ++i)
#     {
#         x1_coords[i] = xs[i][0];
#         y1_coords[i] = xs[i][1];
#         x_ref1_coords[i] = reference_trajectory1[i][0];
#         y_ref1_coords[i] = reference_trajectory1[i][1];

#         x2_coords[i] = xs[i][4];
#         y2_coords[i] = xs[i][5];
#         x_ref2_coords[i] = reference_trajectory2[i][0];
#         y_ref2_coords[i] = reference_trajectory2[i][1];
#     }

#     // Create a Plot object
#     sciplot::Plot plot;

#     // Set the x and y labels
#     plot.xlabel("x (m)");
#     plot.ylabel("y (m)");

#     // Set the x and y ranges
#     auto radius_m = std::max(radius1_m, radius2_m);
#     plot.xrange(-1.25 * radius_m, 1.25 * radius_m);
#     plot.yrange(-1.25 * radius_m, 1.25 * radius_m);

#     // Set the legend to be on the bottom along the horizontal
#     plot.legend()
#         .atOutsideBottom()
#         .displayHorizontal()
#         .displayExpandWidthBy(2);

#     // Plot curves.
#     plot.drawCurve(x1_coords, y1_coords).label("x1").lineWidth(1);
#     plot.drawCurve(x2_coords, y2_coords).label("x2").lineWidth(1);
#     plot.drawCurve(x_ref1_coords, y_ref1_coords).label("P1 reference");
#     plot.drawCurve(x_ref2_coords, y_ref2_coords).label("P2 reference");

#     // Show the plot in a pop-up window
#     plot.show();
# }

# // Plots the states on a graph.
# // TODO: Add plots relative to the second reference trajectory.
# void plot_states(const std::vector<Vector> &xs, const std::vector<Vector> &reference_trajectory1, const std::vector<Vector> &reference_trajectory2, double timestep)
# {
#     const auto horizon = xs.size();

#     // Create a time vector.
#     std::vector<double> t(horizon);
#     for(std::size_t i = 0; i < horizon; ++i)
#     {
#         t[i] = timestep*i;
#     }

#     if (xs.size() != reference_trajectory1.size() || reference_trajectory2.size() != xs.size())
#     {
#         throw std::runtime_error("inconsistent sizes for plot_trajectory.");
#     }

#     // Construct x,y vectors to plot.
#     auto x_coords = std::vector<double>(xs.size());
#     auto y_coords = std::vector<double>(xs.size());
#     auto vel_coords = std::vector<double>(xs.size());
#     auto theta_coords = std::vector<double>(xs.size());

#     auto x_ref1_coords = std::vector<double>(reference_trajectory1.size());
#     auto y_ref1_coords = std::vector<double>(reference_trajectory1.size());
#     auto vel_ref1_coords = std::vector<double>(reference_trajectory1.size());
#     auto theta_ref1_coords = std::vector<double>(reference_trajectory1.size());

#     auto x_ref2_coords = std::vector<double>(reference_trajectory1.size());
#     auto y_ref2_coords = std::vector<double>(reference_trajectory1.size());
#     auto vel_ref2_coords = std::vector<double>(reference_trajectory1.size());
#     auto theta_ref2_coords = std::vector<double>(reference_trajectory1.size());
#     for (auto i = 0ul; i < xs.size(); ++i)
#     {
#         x_coords[i] = xs[i][0];
#         y_coords[i] = xs[i][1];
#         theta_coords[i] = xs[i][2];
#         vel_coords[i] = xs[i][3];

#         x_ref1_coords[i] = reference_trajectory1[i][0];
#         y_ref1_coords[i] = reference_trajectory1[i][1];
#         theta_ref1_coords[i] = reference_trajectory1[i][2];
#         vel_ref1_coords[i] = reference_trajectory1[i][3];

#         x_ref2_coords[i] = reference_trajectory2[i][0];
#         y_ref2_coords[i] = reference_trajectory2[i][1];
#         theta_ref2_coords[i] = reference_trajectory2[i][2];
#         vel_ref2_coords[i] = reference_trajectory2[i][3];
#     }

#     // Create a Plot object
#     sciplot::Plot plot;

#     // Set the x and y labels
#     plot.xlabel("time (s)");
#     plot.ylabel("distance (m)");

#     // Set the legend to be on the bottom along the horizontal
#     plot.legend()
#         .atOutsideBottom()
#         .displayHorizontal()
#         .displayExpandWidthBy(2);

#     // Plot curves.
#     plot.drawCurve(t, x_coords).label("x-pos").lineWidth(1);
#     plot.drawCurve(t, y_coords).label("y-pos").lineWidth(1);
#     plot.drawCurve(t, x_ref1_coords).label("x-pos reference 1");
#     plot.drawCurve(t, y_ref1_coords).label("y-pos reference 1");
#     plot.drawCurve(t, x_ref2_coords).label("x-pos reference 2");
#     plot.drawCurve(t, y_ref2_coords).label("y-pos reference 2");

#     // Show the plot in a pop-up window
#     plot.show();

#     // Create a Plot object
#     sciplot::Plot plot1;

#     // Set the x and y labels
#     plot1.xlabel("time (s)");
#     plot1.ylabel("velocity (m/s)");

#     // Set the legend to be on the bottom along the horizontal
#     plot1.legend()
#         .atOutsideBottom()
#         .displayHorizontal()
#         .displayExpandWidthBy(2);

#     // Plot curves.
#     plot1.drawCurve(t, vel_coords).label("vel");
#     plot1.drawCurve(t, vel_ref1_coords).label("vel reference 1");
#     plot1.drawCurve(t, vel_ref2_coords).label("vel reference 2");

#     // Show the plot in a pop-up window
#     plot1.show();

#     // Create a Plot object
#     sciplot::Plot plot2;

#     // Set the x and y labels
#     plot2.xlabel("time (s)");
#     plot2.ylabel("orientation (rad)");

#     // Set the legend to be on the bottom along the horizontal
#     plot2.legend()
#         .atOutsideBottom()
#         .displayHorizontal()
#         .displayExpandWidthBy(2);

#     // Plot curves.
#     plot2.drawCurve(t, theta_coords).label("orientation");
#     plot2.drawCurve(t, theta_ref1_coords).label("orientation reference 1");
#     plot2.drawCurve(t, theta_ref2_coords).label("orientation reference 2");

#     // Show the plot in a pop-up window
#     plot2.show();
# }

# // Plots controls on a graph.
# // TODO: Add plots relative to the second reference trajectory.
# void plot_controls(const std::vector<std::vector<Vector>> &us, const std::vector<Vector>& refCtrls1, const std::vector<Vector> refCtrls2, double timestep)
# {
#     const auto num_players = us.size();
#     if (num_players != 2)
#     {
#         throw std::runtime_error("plot_controls supports plotting two players' control inputs!");
#     }

#     const size_t horizon = us.front().size();

#     if (us[0].size() != us[1].size())
#     {
#         throw std::runtime_error("inconsistent horizons for plot_controls.");
#     }

#     // Create a time vector.
#     std::vector<double> t(horizon);
#     for(std::size_t i = 0; i < horizon; ++i)
#     {
#         t[i] = timestep*i;
#     }

#     // Construct vectors to plot.
#     std::vector<double> p1_acceleration(horizon - 1);
#     std::vector<double> p1_rotation(horizon - 1);
#     std::vector<double> p2_acceleration(horizon - 1);
#     std::vector<double> p2_rotation(horizon - 1);
#     std::vector<double> u_ref1_acceleration(horizon - 1);
#     std::vector<double> u_ref1_rotation(horizon - 1);
#     std::vector<double> u_ref2_acceleration(horizon - 1);
#     std::vector<double> u_ref2_rotation(horizon - 1);
#     for (auto t = 0ul; t + 1 < horizon; ++t)
#     {
#         p1_rotation[t] = us[0][t][0];
#         p1_acceleration[t] = us[0][t][1];

#         p2_rotation[t] = us[1][t][0];
#         p2_acceleration[t] = us[1][t][1];

#         u_ref1_rotation[t] = refCtrls1[t](0);
#         u_ref1_acceleration[t] = refCtrls1[t](1);

#         u_ref2_rotation[t] = refCtrls2[t](0);
#         u_ref2_acceleration[t] = refCtrls2[t](1);

#     }

#     // Create a Plot object
#     sciplot::Plot plot;

#     // Set the x and y labels
#     plot.xlabel("time (s)");
#     plot.ylabel("acceleration (m/s^2)");

#     // Set the legend to be on the bottom along the horizontal
#     plot.legend()
#         .atOutsideBottom()
#         .displayHorizontal()
#         .displayExpandWidthBy(2);

#     // Plot curves.
#     plot.drawCurve(t, p1_acceleration).label("P1 acceleration controls");
#     plot.drawCurve(t, p2_acceleration).label("P2 acceleration controls");
#     plot.drawCurve(t, u_ref1_acceleration).label("acceleration reference 1");
#     plot.drawCurve(t, u_ref2_acceleration).label("acceleration reference 2");

#     // Show the plot in a pop-up window
#     plot.show();

#     // Create a Plot object
#     sciplot::Plot plot1;

#     // Set the x and y labels
#     plot1.xlabel("time (s)");
#     plot1.ylabel("angular rate (rad/s)");

#     // Set the legend to be on the bottom along the horizontal
#     plot1.legend()
#         .atOutsideBottom()
#         .displayHorizontal()
#         .displayExpandWidthBy(2);

#     // Plot curves.
#     plot1.drawCurve(t, p1_rotation).label("P1 rotation controls");
#     plot1.drawCurve(t, p2_rotation).label("P2 rotation controls");
#     plot1.drawCurve(t, u_ref1_rotation).label("rotation reference 1");
#     plot1.drawCurve(t, u_ref2_rotation).label("rotation reference 2");

#     // Show the plot in a pop-up window
#     plot1.show();
# }