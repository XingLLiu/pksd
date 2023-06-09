############################ 
# Example: Sensor location problem
# adapted from Tak et al. 2016 https://www.tandfonline.com/doi/full/10.1080/10618600.2017.1415911
############################
library("parallel")

######## Data
model <- "modified" # "original"

if (model == "original") {
  loc_true = t(matrix(c(
    0.125, 0.81,
    0.225, 0.475,
    0.35, 0.1,
    0.45, 0.22,
    0.55, 0.73,
    0.57, 0.93,
    0.85, 0.05,
    0.85, 0.8,
    0.3, 0.7,
    0.5, 0.3, 
    0.7, 0.7), nrow=2))

  dist_true <- t(matrix(c(
    0.        , 0.38052192, 0.7524714 , 0.65599793, 0.40797034,
      0.44127336, 1.0521014 , 0.7210038 , 0.19553706, 0.61861753,
      0.57290727,
    0.38052192, 0.        , 0.38855532, 0.34372503, 0.43527004,
    0.59660107, 0.7553813 , 0.6980564 , 0.24463728, 0.33101594,
    0.5384702 ,
    0.7524714 , 0.38855532, 0.        , 0.18410246, 0.6854816 ,
    0.8598348 , 0.49265116, 0.843833  , 0.59837455, 0.24214445,
    0.6814517 ,
    0.65599793, 0.34372503, 0.18410246, 0.        , 0.5165532 ,
    0.71293557, 0.43061745, 0.7368188 , 0.516485  , 0.09596623,
    0.56796384,
    0.40797034, 0.43527004, 0.6854816 , 0.5165532 , 0.        ,
    0.18353213, 0.74661744, 0.32896733, 0.27020597, 0.42551932,
    0.15500511,
    0.44127336, 0.59660107, 0.8598348 , 0.71293557, 0.18353197,
    0.        , 0.9402288 , 0.32038736, 0.37822977, 0.63199854,
    0.24966446,
    1.0521014 , 0.7553813 , 0.49265116, 0.43061745, 0.74661744,
    0.9402288 , 0.        , 0.7449668 , 0.8354213 , 0.4267671 ,
    0.6346882 ,
    0.7210038 , 0.6980564 , 0.843833  , 0.7368188 , 0.32896724,
    0.32038736, 0.7449668 , 0.        , 0.56324923, 0.6024839 ,
    0.19176492,
    0.19553706, 0.24463728, 0.59837455, 0.516485  , 0.27020597,
    0.37822968, 0.8354213 , 0.5632493 , 0.        , 0.42596278,
    0.43248585,
    0.61861753, 0.33101594, 0.24214445, 0.09596623, 0.4255194 ,
    0.63199854, 0.42676708, 0.6024839 , 0.42596278, 0.        ,
    0.4076002 ,
    0.57290727, 0.5384702 , 0.6814517 , 0.56796384, 0.15500511,
    0.2496646 , 0.6346882 , 0.19176508, 0.43248585, 0.40760016,
    0.        ),
    nrow=11
  ))

  O_true <- t(matrix(c(
    0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
    1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
    0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
    0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
    0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1.,
    0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
    0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
    0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1.,
    1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
    0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
    0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.),
    nrow=11
  ))

  # Observation indicators from the fifth sensor (1st column) to the first four sensors
  # and those from the sixth sensor (2nd column) to the first four sensors.
  Ob <- O_true[1:8, 9:11] # 8 x 3

  # Observation indicators among the first four sensors. 
  Os <- O_true[1:8, 1:8] # 8 x 8

  # Each row indicates the location of the known sensors (9th to 11th).
  Xb <- loc_true[9:11,]

  # Each row indicates the location of the unknown sensors (1st to 8th).
  Xs <- loc_true[1:8,]

  # The observed distances from the observed sensors (each col) to the unobserved sensors.
  Yb <- (dist_true * O_true)[1:8, 9:11] # 8 x 3

  # Observed distances among the first 8 sensors.
  Ys <- (dist_true * O_true)[1:8, 1:8] # 8 x 8

  # Path to save result
  path <- "res/sensors/original_ram"

} else if (model == "modified") {
    # Observation indicators from the fifth sensor (1st column) to the first four sensors
  # and those from the sixth sensor (2nd column) to the first four sensors.
  Ob <- matrix(c(1, 0, 1, 0, 1, 0, 1, 0), ncol = 2)

  # Observation indicators among the first four sensors. 
  Os <- matrix(c(0, 0, 0, 1,
                0, 0, 1, 1,
                0, 1, 0, 0,
                1, 1, 0, 0), ncol = 4)

  # Each row indicates the location of the known sensors (5th and 6th).
  Xb <- matrix(c(0.5, 0.3, 0.3, 0.7), ncol = 2)

  # Each row indicates the location of the unknown sensors (1st, 2nd, 3rd, and 4th).
  Xs <- matrix(c(0.5748, 0.0991, 0.2578, 0.8546, 
                0.9069, 0.3651, 0.1350, 0.0392), ncol = 2)

  # The observed distances from the fifth sensor (1st column) to the first four sensors
  # and those from the sixth sensor (2nd column) to the first four sensors.
  Yb <- matrix(c(0.6103, 0, 0.2995, 0, 
                0.3631, 0, 0.5656, 0), ncol = 2)

  # Observed distances among the first four sensors.
  Ys <- matrix(c(0, 0, 0, 0.9266,
                0, 0, 0.2970, 0.8524,
                0, 0.2970, 0, 0,
                0.9266, 0.8524, 0, 0), ncol = 4)

  # Path to save result
  path <- "res/sensors/modified_ram"
}




######## Target joint posterior density
ns <- dim(Ob)[1]
nb <- dim(Ob)[2]

norm2 <- function(loca, locb) {
  sqrt(sum((loca - locb)^2))
}

loglkhd <- function(loc, R = 0.3, sigma = 0.02, Ob, Os, Xb, Xs, Yb, Ys) {
  
  First.term <- NULL
  for (i in 1 : nb) {
    TEMP <- sapply(1 : ns, function(j) {
      norm_sq <- norm2(Xb[i, ], loc[(2 * j -1) : (2 * j)])
      (-norm_sq^2 / 2 / R^2 * Ob[j, i]) +
        log(- expm1(-norm_sq^2 / 2 / R^2))*(1 - Ob[j, i]) 
    })
    First.term <- c(First.term, TEMP)
  }
  
  Second.term <- NULL
  for (i in 1 : (ns - 1)) {
    TEMP <- sapply((i + 1) : ns, function(j) {
      norm_sq <- norm2(loc[(2 * i -1) : (2 * i)], 
                       loc[(2 * j -1) : (2 * j)])
      (-norm_sq^2 / 2 / R^2 * Os[i, j]) +
        log(- expm1(-norm_sq^2 / 2 / R^2))*(1 - Os[i, j]) 
    })
    Second.term <- c(Second.term, TEMP)
  }
  
  First.obs.term <- NULL
  for (i in 1 : nb) {
    TEMP <- sapply(1 : ns, function(j) {
      dnorm(Yb[j, i], mean = norm2(Xb[i, ], loc[(2 * j -1) : (2 * j)]), 
            sd = sigma, log=TRUE)*Ob[j, i]
    })
    First.obs.term <- c(First.obs.term, TEMP)
  }
  
  Second.obs.term <- NULL
  for (i in 1 : (ns - 1)) {
    TEMP <- sapply((i + 1) : ns, function(j) {
      dnorm(Ys[i, j], mean = norm2(loc[(2 * i -1) : (2 * i)], 
                                   loc[(2 * j -1) : (2 * j)]), 
            sd = sigma, log=TRUE)*Os[i, j]
    })
    Second.obs.term <- c(Second.obs.term, TEMP)
  }

  log.lik <- sum(c(First.term, Second.term, First.obs.term, Second.obs.term))
  post <- log.lik
  post
  
}

if (model == "original") {
  l.target <- function(loc, R = 0.3, sigma = 0.02, Ob, Os, Xb, Xs, Yb, Ys) {
    loglkhd(loc, R, sigma, Ob, Os, Xb, Xs, Yb, Ys)
  }
} else if (model == "modified") {
  l.target <- function(loc, R = 0.3, sigma = 0.02, Ob, Os, Xb, Xs, Yb, Ys) {
    log.lik <- loglkhd(loc, R, sigma, Ob, Os, Xb, Xs, Yb, Ys)
    post <- log.lik + sum(dnorm(loc, mean = rep(0, 8), sd = rep(10, 8), log = TRUE))
    post
  }
}

if (model == "original") {
  loc <- c(0.16513085, 0.9014813 , 0.6309742 , 0.4345461 , 0.29193902,
          0.64250207, 0.9757855 , 0.43509948, 0.6601019 , 0.60489583,
          0.6366315 , 0.6144488 , 0.8893349 , 0.6277617 , 0.53197503,
          0.02597821)
  loc <- c(2.02875680, 1.17621448, 0.53307339, 0.98817213, 0.82887315,
          0.55902586, 0.57532534, 0.52816820, 0.22400111, 0.09576452,
          0.95891744, 0.05777278, 0.64247474, 0.16569425, 0.74568182,
          0.20768322)
  l.target(loc, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)

} else if (model == "modified") {
  loc <- c(2.02875680, 1.17621448, 0.53307339, 0.98817213, 0.82887315,
          0.55902586, 0.57532534, 0.52816820)
  l.target(loc, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
}

######## RAM

ram.kernel <- function(current.location, current.aux, loc.number, scale) {
  
  eps <- 10^(-308)
  accept <- 0
  x.c <- current.location 
  log.x.c.den <- l.target(x.c, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
  x.c.den <- exp(log.x.c.den)
  z.c <- current.aux
  log.z.c.den <- l.target(z.c, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
  z.c.den <- exp(log.z.c.den)

  # downhill
  x.p1 <- x.c
  x.p1[(2 * loc.number - 1) : (2 * loc.number)] <- x.p1[(2 * loc.number - 1) : (2 * loc.number)] + 
    rnorm(2, 0, scale)
  log.x.p1.den <- l.target(x.p1, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
  x.p1.den <- exp(log.x.p1.den)
  N.d <- 1
  while (-rexp(1) > log(x.c.den + eps) - log(x.p1.den + eps)) {
    x.p1 <- x.c
    x.p1[(2 * loc.number - 1) : (2 * loc.number)] <- x.p1[(2 * loc.number - 1) : (2 * loc.number)] + 
      rnorm(2, 0, scale)
    log.x.p1.den <- l.target(x.p1, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
    x.p1.den <- exp(log.x.p1.den)
    N.d <- N.d + 1
  }
  
  # uphill
  x.p2 <- x.p1
  x.p2[(2 * loc.number - 1) : (2 * loc.number)] <- x.p2[(2 * loc.number - 1) : (2 * loc.number)] + 
    rnorm(2, 0, scale)
  log.x.p2.den <- l.target(x.p2, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
  x.p2.den <- exp(log.x.p2.den)
  N.u <- 1
  while (-rexp(1) > log(x.p2.den + eps) - log(x.p1.den + eps)) {
    x.p2 <- x.p1
    x.p2[(2 * loc.number - 1) : (2 * loc.number)] <- x.p2[(2 * loc.number - 1) : (2 * loc.number)] + 
      rnorm(2, 0, scale)
    log.x.p2.den <- l.target(x.p2, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
    x.p2.den <- exp(log.x.p2.den)
    N.u <- N.u + 1
  }
  
  # downhill for N.d
  N.dz <- 1     # number of total downhill trials for estimate
  z <- x.p2
  z[(2 * loc.number - 1) : (2 * loc.number)] <- z[(2 * loc.number - 1) : (2 * loc.number)] + 
    rnorm(2, 0, scale)
  log.z.den <- l.target(z, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
  z.den <- exp(log.z.den)
  while (-rexp(1) > log(x.p2.den + eps) - log(z.den + eps)) {
    z <- x.p2
    z[(2 * loc.number - 1) : (2 * loc.number)] <- z[(2 * loc.number - 1) : (2 * loc.number)] + 
      rnorm(2, 0, scale)
    log.z.den <- l.target(z, 0.3, 0.02, Ob, Os, Xb, Xs, Yb, Ys)
    z.den <- exp(log.z.den)
    N.dz <- N.dz + 1
  }
  
  # accept or reject the proposal
  min.nu <- min(1, (x.c.den + eps) / (z.c.den + eps))
  min.de <- min(1, (x.p2.den + eps) / (z.den + eps))
  l.mh <- log.x.p2.den - log.x.c.den + log(min.nu) - log(min.de)
  
  if (l.mh > -rexp(1)) {
    x.c <- x.p2
    z.c <- z
    accept <- 1
  }
  
  c(x.c, z.c, N.d, N.u, N.dz, accept)
}

MHwG.RAM <- function(initial.loc, initial.aux, jump.scale, 
                     Ob, Os, Xb, Xs, Yb, Ys, n.sample = 10, n.burn = 10) {
  
  print(Sys.time())
  n.total <- n.sample + n.burn
  accept <- matrix(0, nrow = n.total, ncol = ns)
  out <- matrix(NA, nrow = n.total, ncol = 2*ns)
  loc.t <- initial.loc
  aux.t <- initial.aux
  Nd <- matrix(NA, nrow = n.total, ncol = ns)
  Nu <- matrix(NA, nrow = n.total, ncol = ns)
  Nz <- matrix(NA, nrow = n.total, ncol = ns)
  
  for (i in 1 : n.total) {
    if (i %% 1000 == 0) {
      cat("Iter ", i, "of", n.total, "\n")
    }
    for (j in 1 : ns) {
      TEMP <- ram.kernel(loc.t, aux.t, j, jump.scale[j])
      loc.t <- TEMP[1 : (2*ns)]
      aux.t <- TEMP[(2*ns+1) : (4*ns)]
      Nd[i, j] <- TEMP[(4*ns+1)]
      Nu[i, j] <- TEMP[(4*ns+2)]
      Nz[i, j] <- TEMP[(4*ns+3)]
      accept[i, j] <- TEMP[(4*ns+4)]
    }
    out[i, ] <- loc.t
  }
  print(Sys.time())
  list(x = out[-c(1 : n.burn), ], 
       accept = accept[-c(1 : n.burn), ],
       N.d = Nd[-c(1 : n.burn), ],
       N.u = Nu[-c(1 : n.burn), ],
       N.z = Nz[-c(1 : n.burn), ])
  
}

## run with different seeds for the same jump scale
args = commandArgs(trailingOnly=TRUE)

j.scale.val <- as.double(args[[1]])
seed.list <- 1:10

res <- mclapply(1 : length(seed.list), function(k) {
  set.seed(k)
  j.scale <- rep(j.scale.val, ns)
  res.ram <- MHwG.RAM(runif(2*ns), runif(2*ns), jump.scale = j.scale, 
                                Ob, Os, Xb, Xs, Yb, Ys, 
                                n.sample = 400000, n.burn = 40000)
  
  write.csv(res.ram$x, paste0(path, j.scale.val, "/", "seed", k, ".csv"), row.names=FALSE)
}, mc.cores=length(seed.list))
