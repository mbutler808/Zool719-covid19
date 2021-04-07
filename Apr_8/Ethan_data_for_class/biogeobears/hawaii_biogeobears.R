require(BioGeoBEARS)
require(ape)

trfn <- read.tree("combo_contree.txt")
trfn <- drop.tip(trfn, "NC_045512_2")
write.tree(trfn, "temp.txt")

trfn = "temp.txt"
geogfn = "sars2_geoloc_hawaii.txt" #needs to be in phylip format
BioGeoBEARS_run_object = define_BioGeoBEARS_run()
BioGeoBEARS_run_object$trfn = trfn
BioGeoBEARS_run_object$geogfn = geogfn

#Defines the model inputs/parameters
BioGeoBEARS_run_object$max_range_size = 2
BioGeoBEARS_run_object$min_branchlength = 0.000001
BioGeoBEARS_run_object$include_null_range = TRUE
BioGeoBEARS_run_object$num_cores_to_use = 8
BioGeoBEARS_run_object$force_sparse = FALSE
BioGeoBEARS_run_object = readfiles_BioGeoBEARS_run(BioGeoBEARS_run_object)
BioGeoBEARS_run_object$return_condlikes_table = TRUE
BioGeoBEARS_run_object$calc_TTL_loglike_from_condlikes_table = TRUE
BioGeoBEARS_run_object$calc_ancprobs = TRUE


#Runs the model
results_DEC_free = bears_optim_run(BioGeoBEARS_run_object)

#prints the annotated phylogeny
pdf(file = "sars2_hawaii_DEC.pdf", height = 100, width = 15)
plot_BioGeoBEARS_results(results_DEC_free, plotwhat = "pie", splitcex=.3, statecex=.3, plotlegend = T)
plot_BioGeoBEARS_results(results_DEC_free, splitcex=.3, statecex=.3, plotlegend = T)
dev.off()


trfn = "temp.txt"
geogfn = "sars2_geoloc_hawaii.txt" #needs to be in phylip format
BioGeoBEARS_run_object = define_BioGeoBEARS_run()
BioGeoBEARS_run_object$trfn = trfn
BioGeoBEARS_run_object$geogfn = geogfn

#Defines the model inputs/parameters
BioGeoBEARS_run_object$max_range_size = 2
BioGeoBEARS_run_object$min_branchlength = 0.000001
BioGeoBEARS_run_object$include_null_range = TRUE
BioGeoBEARS_run_object$num_cores_to_use = 8
BioGeoBEARS_run_object$force_sparse = FALSE
BioGeoBEARS_run_object = readfiles_BioGeoBEARS_run(BioGeoBEARS_run_object)
BioGeoBEARS_run_object$return_condlikes_table = TRUE
BioGeoBEARS_run_object$calc_TTL_loglike_from_condlikes_table = TRUE
BioGeoBEARS_run_object$calc_ancprobs = TRUE
BioGeoBEARS_run_object$BioGeoBEARS_model_object@params_table["j","type"] = "free"
BioGeoBEARS_run_object$BioGeoBEARS_model_object@params_table["j","init"] = 0.01

check_BioGeoBEARS_run(BioGeoBEARS_run_object)

results_DEC_j = bears_optim_run(BioGeoBEARS_run_object)

pdf(file = "sars2_hawaii_DECJ.pdf", height = 100, width = 15)
plot_BioGeoBEARS_results(results_DEC_j, plotwhat = "pie", splitcex=.3, statecex=.3, plotlegend = T)
plot_BioGeoBEARS_results(results_DEC_j, splitcex=.3, statecex=.3, plotlegend = T)
dev.off()

LnL_dec <- get_LnL_from_BioGeoBEARS_results_object(results_DEC_free)
LnL_decj <- get_LnL_from_BioGeoBEARS_results_object(results_DEC_j)

numparams3 = 2
numparams4 = 3

stats = AICstats_2models(LnL_dec, LnL_decj, numparams3, numparams4)
stats$AIC1
stats$AIC2