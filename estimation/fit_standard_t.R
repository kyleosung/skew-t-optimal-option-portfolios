args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript fit_standard_t.R <input.csv> <output.json>", call. = FALSE)
}
infile  <- args[1]
outfile <- args[2]

# @Manual{Azzalini2026_SN_R_package,
#     title = {The {R} package \texttt{sn}: The skew-normal and related distributions such as the skew-$t$ and the {SUN} (version 2.1.3).},
#     author = {Adelchi Azzalini},
#     address = {Universit\`a degli Studi di Padova, Italia},
#     year = {2026},
#     note = {Home page: \url{http://azzalini.stat.unipd.it/SN/}},
#     url = {https://cran.r-project.org/package=sn},
#   }

library(sn)
library(jsonlite)

Y <- as.matrix(read.csv(infile, check.names = FALSE))
storage.mode(Y) <- "double"

s <- 1 / sd(as.vector(Y))
Y_scaled <- Y * s

p <- ncol(Y)

# Intercept-only symmetric multivariate t fit (alpha fixed to zero)
fit_s <- mst.mple(
  x = matrix(1, nrow(Y), 1),
  y = Y_scaled,
  symmetr = TRUE,
  opt.method = "nlminb"
)

dp <- fit_s$dp

# Back-transform direct parameters
dp$beta  <- dp$beta / s
dp$Omega <- dp$Omega / s^2
# dp$nu unchanged

mu    <- as.vector(dp$beta)
Sigma <- dp$Omega
nu    <- dp$nu

# Correlation matrix extracted from Sigma
sigma_std <- sqrt(diag(Sigma))
Corr <- Sigma / outer(sigma_std, sigma_std)
diag(Corr) <- 1.0

# Adjust log-likelihood back to original Y scale
n <- nrow(Y)
logL <- fit_s$logL + n * p * log(s)

out <- list(
  mu    = mu,
  Sigma = Sigma,
  Corr  = Corr,
  nu    = nu,
  logL  = logL
)

write_json(out, outfile, auto_unbox = TRUE, digits = NA)