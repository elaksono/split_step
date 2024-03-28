module PlotPulse

using LaTeXStrings, Plots
export plotCompare


"""
plotCompare(arr_x, num_pulse, anl_pulse, plot_var;\n
        dom = "t", xtext = "Time", comp = real, imax = nothing, ylabel = "Re[A(t)]",
        legend=:bottomleft, margins=5Plots.mm)

Plot to compare the pulse profile obtained from numerical simulation and analytical solution
arr_x can be an array of time or frequency points
comp can be real, imag, or abs
"""
function plotCompare(arr_x, num_pulse, anl_pulse, plot_var; 
        domain = "t", xtext = "Time", comp = real, imax = nothing, ylabel = "Re[A(t)]", color = nothing,
        palette = palette(:tab10), legend=:bottomleft, margins=5Plots.mm)
    if imax == nothing
        imax = size(num_pulse)[2]
    end
    
    sorted_id = sortperm(arr_x)
    sorted_x = arr_x[sorted_id]
    
    for i in collect(1:imax)
        if color == nothing
            color = palette[i]
        end
        
        scatter!(plot_var, arr_x, comp.(num_pulse[:,i]), label = "Numeric " * L"A_{%$i}(%$domain)", color = color,
            xlabel = xtext * " " * domain, ylabel = ylabel, legend=legend, left_margin=5margins, bottom_margin=5margins)
        plot!(plot_var, sorted_x, comp.(anl_pulse[sorted_id,i]), label = "Analytic " * L"A_{%$i}(%$domain)", color = color)
    end
end;
end