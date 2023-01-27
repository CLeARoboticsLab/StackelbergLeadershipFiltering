##
## TODO(documentation)
##

# A control strategy for multiple players.
struct FeedbackGainControlStrategy <: MultiplayerControlStrategy
    num_players::Int                                # number of players
    horizon::Int                                    # horizon
    Ps::AbstractVector{<:AbstractArray{Float64, 3}} # feedback gains
end
FeedbackGainControlStrategy(Ps::AbstractVector{<:AbstractArray{Float64, 3}}) = FeedbackGainControlStrategy(length(Ps), size(Ps[1], 3), Ps)

# This function accepts a feedback gain control strategy and applies it to a state at a given time (i.e. index).
function apply_control_strategy(tt::Int, strategy::FeedbackGainControlStrategy, x::AbstractArray{Float64})
    return [-strategy.Ps[ii][:, :, tt] * x for ii in 1:strategy.num_players]
end

# Export a commonly used control strategy for convenience and its required method.
export FeedbackGainControlStrategy, apply_control_strategy
