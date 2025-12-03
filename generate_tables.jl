using Plots, CSV, DataFrames, Statistics, Distributions, Arrow, LaTeXStrings,Latexify


small_fnames = ["games_all.arrow", "inventory_all.arrow", "ruin_all.arrow", "gridworld_all.arrow"]
large_fnames = ["games_large.arrow", "inventory_large.arrow", "ruin_large.arrow", "gridworld_large.arrow"]

results_arr = []

for (small,large) ∈ zip(small_fnames,large_fnames)

    results1 = copy(DataFrame(Arrow.Table("data2/"*small)))
    results2 = copy(DataFrame(Arrow.Table("data2/"*large))) 


    temp1 = combine(groupby(results1, :algorithm), :runtime => median => :median_small)
    temp2 = combine(groupby(results2, :algorithm), :runtime => median => :median_large)

    push!(results_arr,leftjoin(temp1,temp2,on=:algorithm))
end

results = reduce((left,right) -> leftjoin(left,right,on=:algorithm,makeunique=true),results_arr)
show(results,allrows = true)
println()
latexify(results; env = :tabular, booktabs = true, fmt = "%.6f", latex = false)
