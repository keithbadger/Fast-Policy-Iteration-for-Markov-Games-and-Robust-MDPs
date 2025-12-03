include("rmdp_algorithm_definition.jl")

using DataFrames, CSV, Gurobi, Distributions, Random, MDPs, ProgressBars



function make_random_inventory(state_number,sale_price_cap,order_capacity_cap)
    state_number ≥ 2 || error("There must be atleast 2 states")

    inventory_size = rand(1:state_number-1)
    backlog_size = state_number - inventory_size - 1
    maximum_order = rand(1:min(order_capacity_cap,inventory_size))

    sale_price = rand()*sale_price_cap
    item_cost = rand()*sale_price
    item_storage_cost = rand()*sale_price
    item_backlog_cost = rand()*sale_price
    delivery_cost = rand()*(maximum_order*(sale_price-item_cost))

    expected_demand = rand()*(state_number-1)
    demand_values = 0:(state_number-1)
    demand_dist = Poisson(expected_demand)
    demand_probabilities = map(x->pdf(demand_dist,x), demand_values)
    demand_probabilities[end] += max(1-sum(demand_probabilities),0)

    demand = Domains.Inventory.Demand(demand_values,demand_probabilities)
    costs = Domains.Inventory.Costs(item_cost,delivery_cost,item_storage_cost,item_backlog_cost)
    limits = Domains.Inventory.Limits(inventory_size,backlog_size,maximum_order)

    problem_params = Domains.Inventory.Parameters(demand,costs,sale_price,limits)
    return Domains.Inventory.Model(problem_params)
end

function make_random_grid(side_length)
    reward_s = zeros(side_length*side_length)
    reward_s[rand(1:side_length*side_length)] = 1
    pars = Domains.GridWorld.Parameters(reward_s,side_length,rand())
    return Domains.GridWorld.Model(pars)
end

function make_random_ruin(state_number)
    return Domains.Gambler.Ruin(rand(),state_number-1)
end


function time_algorithm(alg,model,γ,ξ,W,ϵ,env,max_time)
    if alg.name == "VI"
        return VI(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "PAI"
        return PAI(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "HK"
        return HoffKarp(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "FT"
        return Filar(model,γ,ξ,W,ϵ,env,max_time,alg.η,alg.β)
    elseif alg.name == "MP"
        return Mareks(model,γ,ξ,W,ϵ,env,max_time,alg.β)
    elseif alg.name == "RCPI∞"
        return Keiths(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "RCPI₀"
        return RCPI(model,γ,ξ,W,ϵ,env,max_time)
    elseif alg.name == "WS"
        return Winnicki(model,γ,ξ,W,ϵ,env,max_time,alg.H,alg.m)
    elseif alg.name == "PPI"
        return PPI(model,γ,ξ,W,ϵ,env,max_time,alg.β,alg.ϵ₂)
    else error("algorithm name must be one of: \n
                VI, PAI, HK, PPI, FT, M1, KB, RCPI, WIN")
    end
end

function benchmark_run_inventory(algs ,state_nums::Vector{Int64}, Γ::Vector{Float64}, ξ, ϵ::Number, maxtime::Float64)
    results = DataFrame(runtime = Vector{Float64}(undef,0),inv_id = Vector{Int64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0),
                        γ = Vector{Float64}(undef,0), times = Vector{Vector{Float64}}(undef,0), errors = Vector{Vector{Float64}}(undef,0))
    G_ENV = Gurobi.Env()
    thread_lock = ReentrantLock()
    pbar = ProgressBar(total=length(state_nums)*length(Γ)*length(algs))
    Threads.@threads for id ∈ eachindex(state_nums)
        nₛ = state_nums[id]
        inv_prob = make_random_inventory(nₛ,100,50)
        W = [ones(action_count(inv_prob,s),state_count(inv_prob)) for s ∈ 1:state_count(inv_prob)]
        for disc ∈ Γ
            for alg ∈ algs
                out = time_algorithm(alg,inv_prob,disc,ξ,W,ϵ,G_ENV, maxtime)
                @lock thread_lock (push!(results, [out.times[end],id,nₛ,alg.name,disc,out.times,out.errors]); update(pbar)) 
            end
        end
    end
    return results
end

function benchmark_run_gridworld(algs ,state_nums::Vector{Int64}, Γ::Vector{Float64}, ξ, ϵ::Number, maxtime::Float64)
    results = DataFrame(runtime = Vector{Float64}(undef,0),grid_id = Vector{Int64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0),
                        γ = Vector{Float64}(undef,0), times = Vector{Vector{Float64}}(undef,0), errors = Vector{Vector{Float64}}(undef,0))
    G_ENV = Gurobi.Env()
    thread_lock = ReentrantLock()
    pbar = ProgressBar(total=length(state_nums)*length(Γ)*length(algs))
    Threads.@threads for id ∈ eachindex(state_nums)
        nₛ = state_nums[id]
        grid_prob = make_random_grid(nₛ)
        W = [ones(action_count(grid_prob,s),state_count(grid_prob)) for s ∈ 1:state_count(grid_prob)]
        for disc ∈ Γ
            for alg ∈ algs
                out = time_algorithm(alg,grid_prob,disc,ξ,W,ϵ,G_ENV, maxtime)
                @lock thread_lock (push!(results, [out.times[end],id,nₛ,alg.name,disc,out.times,out.errors]); update(pbar)) 
            end
        end
    end
    return results
end

function benchmark_run_ruin(algs ,state_nums::Vector{Int64}, Γ::Vector{Float64}, ξ, ϵ::Number, maxtime::Float64)
    results = DataFrame(runtime = Vector{Float64}(undef,0),ruin_id = Vector{Int64}(undef,0), state_number = Vector{Int64}(undef,0), algorithm = Vector{String}(undef,0),
                        γ = Vector{Float64}(undef,0), times = Vector{Vector{Float64}}(undef,0), errors = Vector{Vector{Float64}}(undef,0))
    G_ENV = Gurobi.Env()
    thread_lock = ReentrantLock()
    pbar = ProgressBar(total=length(state_nums)*length(Γ)*length(algs))
    Threads.@threads for id ∈ eachindex(state_nums)
        nₛ = state_nums[id]
        ruin_prob = make_random_ruin(nₛ)
        W = [ones(action_count(ruin_prob,s),state_count(ruin_prob)) for s ∈ 1:state_count(ruin_prob)]
        for disc ∈ Γ
            for alg ∈ algs
                out = time_algorithm(alg,ruin_prob,disc,ξ,W,ϵ,G_ENV, maxtime)
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

#[ones(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]