args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript fit_ac_skewt.R <input.csv> <output.json>", call. = FALSE)
}
infile <- args[1]
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

# Intercept-only multivariate skew-t fit
fit_s <- mst.mple(
  x = matrix(1, nrow(Y), 1),
  y = Y_scaled,
  opt.method = "nlminb"
)

dp <- fit_s$dp

# Back-transform direct parameters
dp$beta  <- dp$beta / s
dp$Omega <- dp$Omega / s^2
# dp$alpha unchanged
# dp$nu unchanged

# For intercept-only model, beta is the location xi
dp_dist <- list(
  xi    = as.vector(dp$beta),
  Omega = dp$Omega,
  alpha = dp$alpha,
  nu    = dp$nu
)

cp <- if (ncol(Y) > 1) dp2cp(dp_dist, family = "ST") else NULL

# Adjust log-likelihood back to original Y scale
n <- nrow(Y)
p <- ncol(Y)
logL <- fit_s$logL + n * p * log(s)

# Project notation:
# mu    = location vector
# Sigma = scale matrix
# alpha = shape vector (R's dp$alpha); not the tail-risk level
# omega = skewness vector, omega = diag(Sigma)^(-1/2) alpha
# nu    = degrees of freedom
out <- list(
  dp = list(
    mu    = as.vector(dp$beta),
    Sigma = as.matrix(dp$Omega),
    alpha = dp$alpha,
    omega = dp$alpha / sqrt(diag(as.matrix(dp$Omega))),
    nu    = dp$nu
  ),
  cp = cp,
  logL = logL
)

write_json(out, outfile, auto_unbox = TRUE, digits = NA)
