using JuMP, Gurobi, MDPs, LinearAlgebra

function robust_bellman_solve(P̄ₛ,W,Z,κ, env)
    lpm = Model(()->(Gurobi.Optimizer(env)))
    set_silent(lpm)
    set_attribute(lpm, "Threads", 1)
    set_attribute(lpm, "BarConvTol", 1e-9)
    set_attribute(lpm, "OptimalityTol", 1e-9)
    n = size(P̄ₛ)[2]
    m = size(P̄ₛ)[1]
    @variable(lpm, d[1:m])
    @variable(lpm, x[1:m])
    @variable(lpm, λ)
    @variable(lpm, yᵖ[1:n,1:m])
    @variable(lpm, yⁿ[1:n,1:m])
    @objective(lpm,Max,sum([x[a] + P̄ₛ[a,:]'*(yⁿ[:,a] .- yᵖ[:,a]) for a ∈ 1:m]) - κ*λ)
    @constraint(lpm, sum(d) == 1)
    @constraint(lpm, d .≥ 0)
    s = @constraint(lpm, -yᵖ .+ yⁿ .+ ones(n)*x' .≤ Z'*diagm(d))
    @constraint(lpm, yᵖ .+ yⁿ .- λ*W' .≤ 0)
    @constraint(lpm,yᵖ .≥ 0)
    @constraint(lpm,yⁿ .≥ 0)
    @constraint(lpm,λ ≥ 0)
    optimize!(lpm)
    return (πₛ = value.(d), pₛ = -dual.(s)')
end

function worst_prob_solve(P̄ₛ,W,Z,κ,d,env)
    lpm = Model(()->(Gurobi.Optimizer(env)))
    set_silent(lpm)
    set_optimizer_attribute(lpm, "Threads", 1)
    n = size(P̄ₛ)[2]
    m = size(P̄ₛ)[1]
    @variable(lpm, P[1:m,1:n])
    @variable(lpm, Θ[1:m,1:n])
    @objective(lpm,Min,sum([d[a]*P[a,:]'*Z[a,:] for a ∈ 1:m]))
    @constraint(lpm, P*ones(n) .== 1)
    @constraint(lpm, P .≥ 0)
    @constraint(lpm, Θ .≥ 0)
    @constraint(lpm, P .- P̄ₛ .≥ -Θ)
    @constraint(lpm, P̄ₛ .- P .≥ -Θ)
    @constraint(lpm, -sum([W[a,:]'*Θ[a,:] for a ∈ 1:m]) ≥ -κ)
    optimize!(lpm)
    return (πₛ = d, pₛ = value.(P))
end


function Bv!(vᵏ⁺¹,vᵏ,P̄,R,W,γ,ξ,env)
    for s ∈ eachindex(R)
        Z = R[s] + γ*ones(size(R[s])[1])*vᵏ'
        out = robust_bellman_solve(P̄[s],W[s],Z,ξ,env)
        vᵏ⁺¹[s] = [out.pₛ[a,:]'*Z[a,:] for a ∈ 1:size(R[s])[1]]'*out.πₛ
    end
end

function B!(vᵏ⁺¹,πᵏ,Pᵏ,vᵏ,P̄,R,W,γ,ξ,env)
    for s ∈ eachindex(R)
        Z = R[s] + γ*ones(size(R[s])[1])*vᵏ'
        out = robust_bellman_solve(P̄[s],W[s],Z,ξ,env)
        vᵏ⁺¹[s] = [out.pₛ[a,:]'*Z[a,:] for a ∈ 1:size(R[s])[1]]'*out.πₛ
        πᵏ[s] = out.πₛ
        Pᵏ[s] = out.pₛ
    end 
end

function Bμ!(vᵏ⁺¹,πᵏ,Pᵏ,vᵏ,P̄,R,W,γ,ξ,env)
    for s ∈ eachindex(R)
        Z = R[s] + γ*ones(size(R[s])[1])*vᵏ'
        out = worst_prob_solve(P̄[s],W[s],Z,ξ,πᵏ[s],env)
        vᵏ⁺¹[s] = [out.pₛ[a,:]'*Z[a,:] for a ∈ 1:size(R[s])[1]]'*out.πₛ
        Pᵏ[s] = out.pₛ
    end
end

function P_π!(P_π,πᵏ,Pᵏ)
    for s ∈ eachindex(Pᵏ)
        P_π[s,:] = Pᵏ[s]'*πᵏ[s]
    end
end

function R_π!(R_π,πᵏ,Pᵏ,R)
    for s ∈ eachindex(R)
        r = R[s]
        R_π[s] = [r[a,:]'*Pᵏ[s][a,:] for a ∈ 1:size(r)[1]]'*πᵏ[s]
    end
end



function VI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    Bv!(u,v,P̄,R,W,γ,ξ,env)
    err = norm(u-v,Inf)
    while err > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,err)
        v .= u
        Bv!(u,v,P̄,R,W,γ,ξ,env)
        err = norm(u-v,Inf)
    end
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    append!(times,time()-start)
    append!(errors,err)
    return (value = u, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end


function PAI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    err = norm(u-v,Inf)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    while err > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,err)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        v .= (I - γ*P_π) \ R_π
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        err = norm(u-v,Inf)
    end
    append!(times,time()-start)
    append!(errors,err)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end

function Ψ!(z,πᵏ,Pᵏ,v,P̄,R,W,ξ,γ,env)
    B!(z,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    sum((z-v).^2)
end


function Filar(model,γ,ξ,W,ϵ,env,time_limit,η,β,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    err = norm(u-v,Inf)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    s = zeros(length(R))
    while err > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,err)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        s .= (I - γ*P_π) \ R_π - v
        α = 1.
        δ = ((γ*P_π - I)'*(u-v))'*s
        J = sum((u-v).^2)
        while Ψ!(u,πᵏ,Pᵏ,v+α*s,P̄,R,W,ξ,γ,env) - J > η*α*δ && time()-start < time_limit
            α *= β
        end
        v .= v + α*s
        err = norm(u-v,Inf)
    end
    append!(times,time()-start)
    append!(errors,err)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end

function Keiths(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    d = norm(u-v,Inf)
    while d > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,d)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        v .= (I - γ*P_π) \ R_π
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        while norm(u-v, Inf) > γ*d && time()-start < time_limit
            v .= u
            B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        end
        d = norm(u-v,Inf)
    end
    append!(times,time()-start)
    append!(errors,d)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end


function RCPI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    w = Vector{Float64}(undef,length(v₀))
    z = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    d = norm(u-v, Inf)
    while d > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,d)
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        w .= (I - γ*P_π) \ R_π
        B!(z,πᵏ,Pᵏ,w,P̄,R,W,γ,ξ,env)
        if norm(z-w,Inf) > γ*d
            v .= u
            B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        else
            v .= w
            u .= z
        end
        d = norm(u-v, Inf)
    end
    append!(times,time()-start)
    append!(errors,d)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end

"""

Optimizes over values where `B u .≥ u`
"""
function Mareks(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,β,v₀=zeros(state_count(model)))
    # TODO: check if v₀ satisfies the LE condition?
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end
    
    v = copy(v₀)
    u = copy(v₀)
    for s ∈ 1:state_count(model)
        v[s] = -abs((1/(1-γ))*minimum(R[s]))
    end
    u .= v

    # Bellman operator
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)

    # For policy evaluation
    P_π = zeros(state_count(model),state_count(model))
    R_π = zeros(state_count(model))
    s = zeros(state_count(model))

    # A temporary variable to check monotonicity
    Bu = zeros(state_count(model))
    
    while norm(u-v, Inf) > ϵ && time()-start < time_limit
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)

        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)

        s .= (I - γ*P_π) \ R_π - v
        α = 1
        # TODO: should this be a u or v?
        Bu .= u
        B!(Bu,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        while any(u .> Bu) && time()-start < time_limit
            α *= β
            u .= v + α*s
            Bu .= u
            B!(Bu,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        end
    end
    return (value = v, policy = πᵏ, worst_transition = Pᵏ)
end

function Winnicki(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,H,m,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    err = norm(u-v,Inf)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    while err > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,err)
        for i = 2:H
            v .= u
            B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        end
        P_π!(P_π,πᵏ,Pᵏ)
        R_π!(R_π,πᵏ,Pᵏ,R)
        for i = 1:m
            u .= R_π + γ*P_π*u
        end
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        err = norm(u-v,Inf)
    end
    append!(times,time()-start)
    append!(errors,err)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end

function HoffKarp(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    w = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    err = norm(u-v,Inf)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    while err > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,err)
        Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        while norm(w-u,Inf) > 1e-6 && time()-start < time_limit
            P_π!(P_π,πᵏ,Pᵏ)
            R_π!(R_π,πᵏ,Pᵏ,R)
            u .= (I - γ*P_π) \ R_π
            Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        end
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        err = norm(u-v,Inf) 
    end
    append!(times,time()-start)
    append!(errors,err)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end

function PPI(model::TabMDP,γ,ξ,W,ϵ,env,time_limit,β,ϵ₂,v₀=zeros(state_count(model)))
    times = Vector{Float64}(undef,0)
    errors = Vector{Float64}(undef,0)
    start = time()
    πᵏ = [zeros(action_count(model,s)) for s ∈ 1:state_count(model)]
    Pᵏ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    P̄ = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    R = [zeros(action_count(model,s),state_count(model)) for s ∈ 1:state_count(model)]
    for s ∈ 1:state_count(model)
        for a ∈ 1:action_count(model,s)
            next = transition(model,s,a)
            for (sⁿ,p,r) ∈ next
                P̄[s][a,sⁿ] += p
                R[s][a,sⁿ] += r
            end
        end
    end

    v = copy(v₀)
    u = Vector{Float64}(undef,length(v₀))
    w = Vector{Float64}(undef,length(v₀))
    B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
    err = norm(u-v,Inf)
    P_π = Matrix{Float64}(undef, (state_count(model),state_count(model)))
    R_π = Vector{Float64}(undef, state_count(model))
    while err > ϵ && time()-start < time_limit
        append!(times,time()-start)
        append!(errors,err)
        Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        while norm(w-u,Inf) > ϵ₂ + 1e-6 && time()-start < time_limit
            P_π!(P_π,πᵏ,Pᵏ)
            R_π!(R_π,πᵏ,Pᵏ,R)
            u .= (I - γ*P_π) \ R_π
            Bμ!(w,πᵏ,Pᵏ,u,P̄,R,W,γ,ξ,env)
        end
        ϵ₂ *= β
        v .= u
        B!(u,πᵏ,Pᵏ,v,P̄,R,W,γ,ξ,env)
        err = norm(u-v,Inf)
    end
    append!(times,time()-start)
    append!(errors,err)
    return (value = v, policy = πᵏ, worst_transition = Pᵏ, times = times, errors = errors)
end