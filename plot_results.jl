using Plots, CSV, DataFrames, Statistics, Distributions, Arrow, LaTeXStrings,Latexify

@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; number = 10, maxlen = 1000)
    n = sum(x .≤ maxlen)
    step = Int(ceil(n/number))
    sx, sy = x[[1:step:n;n]], y[[1:step:n;n]]
    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
end

tolerance = 1e-3
algorithms = ["RCPI∞", "RCPI₀", "VI", "PAI", "FT", "WS", "HK", "PPI"]
alg_marks = [:circle, :rect, :diamond, :hexagon, :cross, :dtriangle, :star5, :utriangle]
alg_colors = Plots.palette(:auto)[1:8]

filenames = ["games_all.arrow", "inventory_all.arrow", "ruin_all.arrow", "gridworld_all.arrow","games_large.arrow", "inventory_large.arrow", "ruin_large.arrow", "gridworld_large.arrow"]
xmaxs = [2.,20,100,50,5,7000,Inf,800]

for (filename, xmax) ∈ zip(filenames,xmaxs)
    results = DataFrame(Arrow.Table("data2/"*filename))
    max_runs = combine(groupby(results, :algorithm), [:runtime,:times] => ((a,b) -> b[argmax(a)]) => :times, [:runtime,:errors] => ((a,b) -> b[argmax(a)]) => :errors )
    p = plot(yscale = :log, xlim = (0,xmax), xlabel = "Time (s)", ylabel = L"\psi_\infty(v)", size = (600,400))
    for (i,alg) ∈ enumerate(algorithms)
        run = subset(max_runs, :algorithm => (a -> a .== alg))
        if size(run)[1] > 0
            if run.errors[end] < tolerance
                z = (run.errors[end]/run.errors[end-1])^(1/(run.times[end]-run.times[end-1]))
                run.times[end] = log(z,tolerance/run.errors[end-1]) + run.times[end-1]
                run.errors[end] = tolerance
            end
            plot!(run.times,run.errors,label = alg, seriescolor = alg_colors[i], markershape = alg_marks[i],markerstrokewidth = .5, legend = :topright, seriestype = :samplemarkers, number = 5, maxlen = xmax)
        end
    end
    hline!([1e-3], linestyle = :dash, color = :red, label = "Tolerance")
    display(p)
end






