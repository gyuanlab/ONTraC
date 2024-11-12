#! /usr/bin/Rscript

# ---------- Load modules ----------
library(STdeconvolve)

# ---------- parameters ----------
args <- commandArgs(trailingOnly = TRUE)

counts_path <- args[1]
num_cell_type <- as.numeric(args[2])
save_directory <- args[3]
output_file_name <- args[4]

counts_df <- read.csv(counts_path, row.names = 1, header = TRUE)
counts_df[is.na(counts_df)] <- 0
counts_matrix <- as.matrix(counts_df)
sparse_counts <- as(counts_matrix, "dgCMatrix")

if (nrow(sparse_counts) == 0 || ncol(sparse_counts) == 0) {
    stop("The input matrix is empty after filtering.")
}

counts <- cleanCounts(sparse_counts, min.lib.size = 50, min.reads = 10)
corpus <- restrictCorpus(
    counts,
    removeAbove = 1.0,
    removeBelow = 0.05,
    nTopOD = 1000
)
ldas <- fitLDA(t(as.matrix(corpus)), Ks = c(num_cell_type))
optLDA <- optimalModel(models = ldas, opt = "min") # nolint
results <- getBetaTheta(optLDA, perc.filt = 0.05, betaScale = 1000)

write.table(results$theta,
            file = gzfile(paste0(save_directory, output_file_name)),
            sep = ",",
            row.names = FALSE,
            col.names = FALSE,
            quote = FALSE)
