using Plots, CSV, DataFrames, Statistics, Distributions, Arrow, LaTeXStrings, Latexify

filenames = ["mg_all.arrow","games_large.arrow", "inv_all.arrow", "inventory_large.arrow", "ruin_all.arrow", "ruin_large.arrow", "grid_all.arrow", "gridworld_large.arrow"]
filenames2 = ["games_large.arrow", "inventory_large.arrow", "ruin_large.arrow", "gridworld_large.arrow"]
id_names = [:game_id, :inv_id, :ruin_id, :grid_id]
column_titles = ["game_small", "game_large", "inventory_small", "inventory_large", "ruin_small", "ruin_large", "grid_small", "grid_large"]
column_titles2 = ["Markov Games", "Inventory", "Gambler's Ruin", "Gridworld"]
# Initialize an empty list to store the DataFrames
temp_dfs = []

#= for (ind, filename) ∈ enumerate(filenames)
    results = copy(DataFrame(Arrow.Table("Paper/data/" * filename)))
    temp = combine(groupby(combine(groupby(results, [id_names[Int(ceil(ind/2))], :γ, :state_number]), [:runtime, :algorithm] => (t,a) -> t, :algorithm => a->a), :algorithm_function), :runtime_algorithm_function => median => column_titles[ind])
    push!(temp_dfs, temp)
end =#

for (ind, filename) ∈ enumerate(filenames2)
    results = copy(DataFrame(Arrow.Table("data/" * filename)))
    temp = combine(groupby(combine(groupby(results, [id_names[ind], :γ, :state_number]), [:runtime, :algorithm] => (t,a) -> t, :algorithm => a->a), :algorithm_function), :runtime_algorithm_function => maximum => column_titles2[ind])
    push!(temp_dfs, temp)
end

# Start with the first DataFrame and left-join the rest
final_df = reduce((left, right) -> outerjoin(left, right, on=:algorithm_function), temp_dfs)

display(final_df)

# save the table to the clipboard as latex code for easy inclusion in LaTeX documents
latexify(final_df; env = :tabular, booktabs = true, fmt = "%.1f", latex = false)



