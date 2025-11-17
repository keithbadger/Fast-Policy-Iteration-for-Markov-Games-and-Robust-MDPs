include("game_algorithm_definition.jl")
using DataFrames, CSV, Gurobi, Distributions, Random, MDPs, ProgressBars


function make_markov_game(num_states::Int64,num_actions_x::Vector{Int64},num_actions_y::Vector{Int64},r_lower::Float64,r_upper::Float64, η::Float64)
    num_next = Int(round(η*num_states))
    P = [zeros(num_actions_y[s],num_actions_x[s],num_states) for s ∈ 1:num_states]
    R = [rand(Uniform(r_lower,r_upper),num_actions_y[s],num_actions_x[s]) for s ∈ 1:num_states]
    for s ∈ 1:num_states
        for a ∈ 1:num_actions_y[s]
            for b ∈ 1:num_actions_x[s]  
                P[s][a,b,shuffle(1:num_states)[1:num_next]] = normalize(rand(Exponential(1),num_next),1)
            end
        end
    end
    return (transition = P, rewards = R)
end


function time_algorithm_games(alg,P,R,γ,ϵ,env,time_limit)
    if alg.name == "VI"
        return VI(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "PAI"
        return PAI(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "HK"
        return HoffKarp(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "FT"
        return Filar(P,R,γ,ϵ,env,time_limit,alg.η,alg.β)
    elseif alg.name == "MP"
        return Mareks(P,R,γ,ϵ,env,time_limit,alg.β)
    elseif alg.name == "KB"
        return Keiths(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "RCPI"
        return RCPI(P,R,γ,ϵ,env,time_limit)
    elseif alg.name == "WIN"
        return Winnicki(P,R,γ,ϵ,env,time_limit,alg.H,alg.m)
    elseif alg.name == "PPI"
        return PPI(P,R,γ,ϵ,env,time_limit,alg.ϵ₂,alg.β)
    else error("algorithm name must be one of: \n
                VI, PAI, HK, PPI, FT, MP, KB, RCPI, WIN")
    end
end

function benchmark_run_games(state_nums::Vector{Int64}, Γ::Vector{Float64}, algs, action_nums::Vector{Int64}, r_lower::Float64, r_upper::Float64, η::Number, ϵ::Number, maxtime::Float64)
    results = DataFrame(runtime = Vector{Float64}(undef,0),game_id = Vector{Int64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0),
                        γ = Vector{Float64}(undef,0), times = Vector{Vector{Float64}}(undef,0), errors = Vector{Vector{Float64}}(undef,0))
    G_ENV = Gurobi.Env()
    thread_lock = ReentrantLock()
    pbar = ProgressBar(total=length(state_nums)*length(Γ)*length(algs))
    Threads.@threads for id ∈ eachindex(state_nums)
        nₛ = state_nums[id]
        G = make_markov_game(nₛ,rand(action_nums,nₛ),rand(action_nums,nₛ),r_lower,r_upper,η)
        for disc ∈ Γ
            for alg ∈ algs
                out = time_algorithm_games(alg,G.transition,G.rewards,disc,ϵ,G_ENV,maxtime)
                @lock thread_lock (push!(results, [out.times[end],id,nₛ,alg.name,disc,out.times,out.errors]); update(pbar)) 
            end
        end
    end
    return results
end


#= algorithms = [(name = "VI",), (name = "PAI",), (name = "HK",), (name = "FT", η = 1e-3, β = .5),
              (name = "KB",), (name = "RCPI",), (name = "WIN", H = 2, m = 4),
              (name = "PPI", ϵ₂ = .1, β = .5)]  =#

#= algorithms = [(name = "PAI",), (name = "KB",), (name = "RCPI",), (name = "VI",)]  =#