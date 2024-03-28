module FileIO

using JSON, OrderedCollections
export generate_input_files

function generate_input_files(master_file::String; filedir = "./input/")
    master = JSON.parse(open("$(filedir)$(master_file).json"));
    
    filename = master["filename"];
    vars = master["variation"];
    
    # Template JSON
    temp = JSON.parse(open("$(filedir)$(filename).json"), dicttype=OrderedDict);
    orig_save_label = temp["save_label"]
    
    # Prepare the output directory
    outdir = replace(filedir, "./input/" => "./output/")
    if !isdir(outdir)
        mkdir(outdir)
        println("New output directory created: $(outdir)")
    end
    temp["save_folder"] = outdir
    
    figdir = replace(filedir, "./input/" => "./figures/")
    if !isdir(figdir)
        mkdir(figdir)
        println("New figure directory created: $(figdir)")
    end
    
    
    parameters = []
    for var in vars
        val = master[var]
        pars = collect(val[1]:val[2]:val[3])
        push!(parameters, pars)
    end
    parameters = collect(Iterators.product(parameters...))

    for (n, par) in enumerate(parameters)
        # The varying variables (e.g., loss, pm_depth)
        for (i, var) in enumerate(vars)
            temp[var] = par[i]
        end
        
        temp["save_label"] = orig_save_label * "_$(n)"
        json_string = JSON.json(temp, 4)

        # To write the json file
        open("$(filedir)$(filename)_$(n).json", "w") do f
            write(f, json_string)
        end
    end
    return filedir, filename, length(parameters)
end;

end