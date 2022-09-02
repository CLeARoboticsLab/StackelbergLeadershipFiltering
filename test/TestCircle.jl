// A test that runs the circle example from the fall project.

using CircularHypothesisFilteringUtils: dyn_ode, get_circular_reference_trajectory, rk4_dynamics, construct_big_A, construct_big_B
using LinearAlgebra
using StackelbergControlHypothesesFiltering
using Test: @test, @testset
using Random: seed!

seed!(0)

// #include "ukf.hpp"
// #include "lq_solvers.hpp"
// #include "utils.hpp"
// #include "math.hpp"

// #include <iostream>
// #include <random>

// #include "sciplot/sciplot.hpp"

# P1 is the leader.
LEADER_IDX = 0;

SHOULD_ADD_NOISE = false;
POSITION_STD_DEV = 0.005; # half centimeter
ANGLE_STD_DEV = 0.01; # one-hundredth of a radian
VELOCITY_STD_DEV = 0.005; # half centimeter/sec



# AMM filter with a bank of SR-UKFs.
std::vector<Vector> estimate_model_probabilities(
    double timestep,
    const std::vector<Vector> &xs,
    const std::vector<std::vector<Vector>> &us,
    const std::vector<std::vector<Vector>> &reference_trajectories,
    const std::vector<std::vector<Vector>> &reference_ctrls,
    const size_t player_idx)
{

    const std::size_t other_idx = !player_idx;
    static constexpr size_t numPlayers = 2;

    static const Matrix cholInitialCov = Vector{{0.1, 0.1, 0.1, 0.1}}.
        asDiagonal();
    static const Matrix cholProcNoiseCov = Vector{{1e-2, 1e-3}}.asDiagonal();
    static const Matrix cholMeasNoiseCov = Vector::Constant(2, 1e-3).
        asDiagonal();

    static const MeasFunction genMeas([](double, const Vector& state){ return
        state.head(2).eval(); }, 2, 4);
    static const ResidFunction calcResiduals([](const Vector& a, const Vector&
        b){ return (a - b).eval(); }, 2);

    std::vector<NewStateFunction> genNewState;
    for(const auto& r : reference_ctrls)
    {
        genNewState.emplace_back([timestep, us, r](double startTime, double
            endTime, const Vector& state, const Vector& noise)
        {
            const std::size_t t = std::round(startTime/timestep);
            Vector ctrls = r[t] + noise;
            for(const auto& n : us)
            {
                ctrls += n[t];
            }
            return rk4_dynamics(endTime - startTime, state, ctrls);
        }, 4, 2);
    }

    auto horizon = xs.size();

    std::vector<Vector> true_xs;
    true_xs.resize(horizon);
    for (size_t t = 0; t < horizon; ++t)
    {
        true_xs[t] = xs[t].segment(4*player_idx, 4) + reference_trajectories[
            player_idx][t];
    }

    std::vector<Vector> likelihoods(horizon, Vector(numPlayers));

    // Generate initial PDFs.
    std::vector<Vector> means(numPlayers, true_xs[0]);
    std::vector<Matrix> cholCovs(numPlayers, cholInitialCov);//Vector{{INIT_FILTER_STD_DEV_M, INIT_FILTER_STD_DEV_M, INIT_FILTER_STD_DEV_RAD, INIT_FILTER_STD_DEV_M_S}}.asDiagonal();

    // First measurement update.
    for(std::size_t i = 0; i < numPlayers; ++i)
    {
        sqrt_ukf_update(genMeas, calcResiduals, 0., means[i], cholCovs[i],
            true_xs[i].head(2), cholMeasNoiseCov, means[i], cholCovs[i],
            likelihoods[0](i));
    }

    // Run filters.
    for (size_t t = 0; t + 1 < horizon; ++t)
    {
        // Predict using each model.
        sqrt_ukf_predict(genNewState[player_idx], t*timestep, (t + 1)*timestep,
            means[0], cholCovs[0], cholProcNoiseCov, means[0], cholCovs[0]);
        sqrt_ukf_predict(genNewState[other_idx], t*timestep, (t + 1)*timestep,
            means[1], cholCovs[1], cholProcNoiseCov, means[1], cholCovs[1]);

        // Measurement update.
        for(std::size_t i = 0; i < numPlayers; ++i)
        {
            sqrt_ukf_update(genMeas, calcResiduals, (t + 1)*timestep, means[i],
                cholCovs[i], true_xs[i].head(2), cholMeasNoiseCov, means[i],
                cholCovs[i], likelihoods[t + 1](i));
        }
    }

    // Calculate model probabilities over time.
    std::vector<Vector> modelProbs(horizon);
    modelProbs[0] = Vector::Constant(numPlayers, 1./numPlayers);
    for(std::size_t t = 0; t + 1 < horizon; ++t)
    {
        modelProbs[t + 1] = modelProbs[t].cwiseProduct(likelihoods[t]);
        modelProbs[t + 1] /= modelProbs[t + 1].sum();
    }
    return modelProbs;
}

# static void plot_model_probabilities(double timestep, const std::vector<Vector>&
#     modelProbs)
# {
#     const std::size_t horizon = modelProbs.size();
#     const std::size_t numModels = modelProbs[0].size();
#     std::vector<std::vector<double>> modelProbsSep(numModels);
#     for(std::size_t i = 0; i < numModels; ++i)
#     {
#         modelProbsSep[i].resize(horizon);
#         for(std::size_t t = 0; t < horizon; ++t)
#         {
#             modelProbsSep[i][t] = modelProbs[t](i);
#         }
#     }

#     // Create a time vector.
#     std::vector<double> t(horizon);
#     for(std::size_t i = 0; i < horizon; ++i)
#     {
#         t[i] = timestep*i;
#     }

#     // Create a Plot object.
#     sciplot::Plot plot;

#     // Set the x and y labels.
#     plot.xlabel("time (s)");
#     plot.ylabel("model probabilities");

#     // Set the legend to be on the bottom along the horizontal.
#     plot.legend().atOutsideBottom().displayHorizontal().displayExpandWidthBy(2);

#     // Plot curves.
#     for(std::size_t i = 0; i < numModels; ++i)
#     {
#         plot.drawCurve(t, modelProbsSep[i]).label("model " + std::to_string(i +
#             1));
#     }

#     // Show the plot in a pop-up window.
#     plot.show();
# }

int main()
{
    # if(false) {
    #     coupling_example();
    # }
    # else {

    radius1_m = 10.0
    radius2_m = 1.1 * radius1_m
    horizon = 200
    time_step = 0.1

    reference_traj1, reference_ctrl1 = get_circular_reference_trajectory(horizon, time_step, radius1_m)
    reference_traj2, reference_ctrl2 = get_circular_reference_trajectory(horizon, time_step, radius2_m)

    if (reference_traj1.empty() || reference_traj1.size() != reference_traj2.size())
    {
        throw std::runtime_error("Reference trajectories must be non-empty and same size.");
    }

    // Common dynamics, costs, and initial condition.
    std::vector<Dynamics> dynamics;
    dynamics.reserve(reference_traj1.size());
    for (size_t i = 0; i < reference_traj1.size(); ++i)
    {
        const Vector& state_small = reference_traj1[i];
        const Vector& state_big = reference_traj2[i];
        auto A = construct_big_A(state_small, state_big, time_step);
        auto B1 = construct_big_B(state_small, time_step);
        auto B2 = construct_big_B(state_small, time_step);
        dynamics.emplace_back(Dynamics(A, {B1, B2}));
    }


    // Assumes state costs for P1 only for position - no control cost.
    Matrix Q1 = Matrix::Zero(dynamics[0].stateMat().rows(), dynamics[0].stateMat().cols());
    Q1(0, 0) = 1;
    Q1(1, 1) = 1;
    Q1(4, 4) = 1;
    Q1(5, 5) = 1;
    // TODO: Add cross costs for error position states so that the two error spaces are consistent?
    // Ex. Q1(0, 4) = 100; Q1(1, 5) = 100;
    Cost c1 = (Q1);
    c1.addControlCost(0, Vector::Constant(2, 1).asDiagonal());

    // All zeros for now - assumes no state cost, but P2 does have control cost.
    Matrix Q2 = Matrix::Zero(dynamics[0].stateMat().rows(), dynamics[0].stateMat().cols());
    Q2(0, 0) = 1;
    Q2(1, 1) = 1;
    Q2(4, 4) = 1;
    Q2(5, 5) = 1;
    // TODO: Add cross costs for error position states so that the two error spaces are consistent?
    // Ex. Q2(0, 4) = 100; Q2(1, 5) = 100;
    Cost c2(Q2);
    c2.addControlCost(1, Vector::Constant(2, 1).asDiagonal());

    const std::vector<Cost> costs = {c1, c2};

    # x0 is halfway between the two start points of the circular reference trajectories.
    initialState = [0.5*(radius1_m + radius2_m), 0, 0.5*M_PI, 0, 0.5*(radius1_m + radius2_m), 0, 0.5*M_PI, 0];

    // Ensure that the feedback solution satisfies Nash conditions of optimality
    // for each player, holding others' strategies fixed.
    {
        const std::vector<std::vector<Matrix>> Ps = solve_lq_stackelberg_feedback(dynamics, costs, LEADER_IDX);

        const NonlinDynamics trueDyn = [reference_ctrl1, reference_ctrl2](std::
            size_t t, const std::vector<Vector>& xs, const std::vector<std::
            vector<Vector>>& us)
        {
            Vector ctrls = (Vector(4) << reference_ctrl1[t], reference_ctrl2[
                t]).finished();
            for(const auto& n : us)
            {
                ctrls.head(2) += n[t];
                ctrls.tail(2) += n[t];
            }
            Vector x = xs[t];
            x.head(4) = rk4_dynamics(time_step, x.head(4), ctrls);
            x.tail(4) = rk4_dynamics(time_step, x.tail(4), ctrls);
            return x;
        };

        std::vector<Vector> xRefs(horizon);
        for(std::size_t t = 0; t < horizon; ++t)
        {
            xRefs[t].resize(reference_traj1[t].size() + reference_traj2[t].
                size());
            xRefs[t].head(reference_traj1[t].size()) = reference_traj1[t];
            xRefs[t].tail(reference_traj2[t].size()) = reference_traj2[t];
        }

        std::vector<Vector> xs;
        std::vector<std::vector<Vector>> us;
        unroll_feedback(trueDyn, Ps, initialState.replicate(2, 1), xs, us, xRefs);

        # plot_trajectory(xs, reference_traj1, radius1_m, reference_traj2, radius2_m);
        # plot_states(xs, reference_traj1, reference_traj2, time_step);
        # plot_controls(us, reference_ctrl1, reference_ctrl2, time_step);

        const auto modelProbs = estimate_model_probabilities(time_step, xs, us,
            {reference_traj1, reference_traj2}, {reference_ctrl1,
            reference_ctrl2}, 0);
        plot_model_probabilities(time_step, modelProbs);

        // Test that the equilibrium is Nash.
        test_stackelberg_game(dynamics, costs, xs, us, xRefs, Ps, trueDyn, initialState, 1.0);
    }
    }
}
