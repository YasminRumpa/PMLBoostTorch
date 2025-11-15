pkgs <- c("glmnet", "randomForest", "e1071", "pROC", "data.table", "torch")
to_install <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(to_install)) install.packages(to_install, repos = "http://cran.us.r-project.org")
lapply(pkgs, require, character.only = TRUE)

generate_data <- function(scenario, n = 1000, p = 10, q = 10, seed = 1) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), ncol = p)
  Z <- matrix(rnorm(n * q), ncol = q)
  colnames(X) <- paste0("X", 1:p)
  colnames(Z) <- paste0("Z", 1:q)
  
  tau <- rep(NA, n)
  tau[X[,1] < 0 & X[,2] < 0] <- 1
  tau[X[,1] >= 0 & X[,2] < 0] <- 2
  tau[X[,1] < 0 & X[,2] >= 0] <- 3
  tau[X[,1] >= 0 & X[,2] >= 0] <- 4
  
  group_cols <- split(1:ncol(Z), rep(1:4, length.out = ncol(Z)))
  Z1 <- Z2 <- rep(NA, n)
  for (g in 1:4) {
    idx <- which(tau == g)
    if (length(group_cols[[g]]) >= 2) {
      c1 <- group_cols[[g]][1]; c2 <- group_cols[[g]][2]
    } else if (length(group_cols[[g]]) == 1) {
      c1 <- group_cols[[g]][1]; c2 <- group_cols[[g]][1]
    } else {
      c1 <- 1; c2 <- 1
    }
    if (length(idx) > 0) {
      Z1[idx] <- Z[idx, c1]
      Z2[idx] <- Z[idx, c2]
    }
  }
  
  if (scenario <= 3) {
    if (scenario == 1) Y <- 2 * Z1 - Z2 + rnorm(n)
    else if (scenario == 2) Y <- 3 * Z1^2 + 2 * sin(Z2) + rnorm(n)
    else Y <- 5 * sin(Z1) + 0.5 * X[,2]^3 - 2 * Z1 * Z2 + rnorm(n)
  } else {
    if (scenario == 4) prob <- plogis(2 * Z1 - Z2)
    else if (scenario == 5) prob <- plogis(3 * Z1^2 + 2 * sin(Z2))
    else prob <- plogis(5 * sin(Z1) + 0.5 * X[,2]^3 - 2 * Z1 * Z2)
    Y <- rbinom(n, 1, prob)
  }
  list(y = Y, x = X, z = Z)
}

MLP_torch <- nn_module(
  "MLP_torch",
  initialize = function(input_dim, hidden_layers = c(32, 16), output_dim = 1, is_binary = FALSE) {
    self$is_binary <- is_binary
    layers <- list()
    prev <- input_dim
    for (h in hidden_layers) {
      layers <- append(layers, nn_linear(prev, h))
      layers <- append(layers, nn_relu())
      prev <- h
    }
    layers <- append(layers, nn_linear(prev, output_dim))
    self$net <- nn_sequential(!!!layers)
  },
  forward = function(x) {
    out <- self$net(x)
    if (self$is_binary) torch_sigmoid(out) else out
  }
)

train_torch_nn <- function(z_train, y_train,
                           z_val = NULL, y_val = NULL,
                           hidden = c(32, 16), epochs = 40, lr = 0.01,
                           batch_size = 32, is_binary = FALSE,
                           device = torch_device(if(cuda_is_available()) "cuda" else "cpu")) {
  
  xtr <- torch_tensor(as.matrix(z_train), dtype = torch_float(), device = device)
  ytr <- torch_tensor(matrix(y_train, ncol = 1), dtype = torch_float(), device = device)
  
  input_dim <- ncol(z_train)
  model <- MLP_torch(input_dim, hidden, output_dim = 1, is_binary = is_binary)
  model$to(device = device)
  
  optimizer <- optim_adam(model$parameters, lr = lr)
  loss_fn   <- if (is_binary) nn_bce_loss() else nn_mse_loss()
  
  n <- nrow(z_train)
  for (epoch in seq_len(epochs)) {
    model$train()
    idx <- sample(seq_len(n), n)
    for (i in seq(1, n, by = batch_size)) {
      batch <- idx[i:min(i + batch_size - 1, n)]
      xb <- xtr[batch, ]; yb <- ytr[batch, ]
      optimizer$zero_grad()
      out <- model(xb)
      loss <- loss_fn(out, yb)
      loss$backward()
      optimizer$step()
    }
  }
  
  predfun <- function(newz) {
    if (is.null(newz) || nrow(as.matrix(newz)) == 0) return(numeric(0))
    xt <- torch_tensor(as.matrix(newz), dtype = torch_float(), device = device)
    model$eval()
    with_no_grad({
      out <- model(xt)$squeeze()
      as.numeric(out$to(device = "cpu")$numpy())
    })
  }
  
  if (!is.null(z_val) && !is.null(y_val)) {
    val_pred <- predfun(z_val)
    metric <- if (is_binary)
      as.numeric(pROC::auc(pROC::roc(y_val, val_pred)))
    else
      mean((y_val - val_pred)^2)
  } else metric <- NA_real_
  
  list(predfun = predfun, metric = metric)
}

train_and_eval_candidates <- function(y_train, z_train, y_val, z_val, is_binary) {
  out <- list()
  df_tr <- as.data.frame(z_train); df_val <- as.data.frame(z_val)
  
  # 1) Linear
  if (is_binary) {
    fit_lin <- tryCatch(glm(y_train ~ ., data = cbind(y_train = y_train, df_tr), family = binomial), error = function(e) NULL)
    pred_lin <- if (!is.null(fit_lin)) predict(fit_lin, newdata = df_val, type = "response") else rep(mean(y_train), nrow(df_val))
  } else {
    fit_lin <- tryCatch(lm(y_train ~ ., data = cbind(y_train = y_train, df_tr)), error = function(e) NULL)
    pred_lin <- if (!is.null(fit_lin)) predict(fit_lin, newdata = df_val) else rep(mean(y_train), nrow(df_val))
  }
  metric_lin <- if (is_binary) tryCatch(as.numeric(pROC::auc(pROC::roc(y_val, pred_lin))), error = function(e) NA) else mean((y_val - pred_lin)^2)
  out$Linear <- list(predfun = function(newz) {
    if (is.null(fit_lin)) rep(mean(y_train), nrow(as.data.frame(newz))) else predict(fit_lin, newdata = as.data.frame(newz), type = ifelse(is_binary, "response", "response"))
  }, metric = metric_lin)
  
  # 2) Lasso
  fit_lasso <- tryCatch(cv.glmnet(as.matrix(z_train), y_train, family = ifelse(is_binary, "binomial", "gaussian")), error = function(e) NULL)
  pred_lasso <- if (!is.null(fit_lasso)) as.numeric(predict(fit_lasso, newx = as.matrix(z_val), s = "lambda.min", type = ifelse(is_binary, "response", "link"))) else rep(mean(y_train), nrow(z_val))
  metric_lasso <- if (is_binary) tryCatch(as.numeric(pROC::auc(pROC::roc(y_val, pred_lasso))), error = function(e) NA) else mean((y_val - pred_lasso)^2)
  out$Lasso <- list(predfun = function(newz) {
    if (is.null(fit_lasso)) rep(mean(y_train), nrow(as.data.frame(newz))) else as.numeric(predict(fit_lasso, newx = as.matrix(newz), s = "lambda.min", type = ifelse(is_binary, "response", "link")))
  }, metric = metric_lasso)
  
   # 3) Random Forest
  y_rf <- if (is_binary) as.factor(y_train) else y_train
  fit_rf <- tryCatch(randomForest(x = as.matrix(z_train), y = y_rf), error = function(e) NULL)
  pred_rf <- if (!is.null(fit_rf)) {
    pr <- predict(fit_rf, as.data.frame(z_val), type = if (is_binary) "prob" else "response")
    if (is_binary) pr[,2] else as.numeric(pr)
  } else rep(mean(y_train), nrow(z_val))
  metric_rf <- if (is_binary) tryCatch(as.numeric(pROC::auc(pROC::roc(y_val, pred_rf))), error = function(e) NA) else mean((y_val - pred_rf)^2)
  out$RandomForest <- list(predfun = function(newz) {
    if (is.null(fit_rf)) rep(mean(y_train), nrow(as.data.frame(newz))) else {
      pr <- predict(fit_rf, as.data.frame(newz), type = if (is_binary) "prob" else "response")
      if (is_binary) pr[,2] else as.numeric(pr)
    }
  }, metric = metric_rf)

  # 4) SVM
  y_svm <- if (is_binary) as.factor(y_train) else y_train
  fit_svm <- tryCatch(svm(x = as.matrix(z_train), y = y_svm, probability = is_binary), error = function(e) NULL)
  pred_svm <- if (!is.null(fit_svm)) {
    pr <- predict(fit_svm, as.data.frame(z_val), probability = is_binary)
    if (is_binary) attr(pr, "probabilities")[,2] else as.numeric(pr)
  } else rep(mean(y_train), nrow(z_val))
  metric_svm <- if (is_binary) tryCatch(as.numeric(pROC::auc(pROC::roc(y_val, pred_svm))), error = function(e) NA) else mean((y_val - pred_svm)^2)
  out$SVM <- list(predfun = function(newz) {
    if (is.null(fit_svm)) rep(mean(y_train), nrow(as.data.frame(newz))) else {
      pr <- predict(fit_svm, as.data.frame(newz), probability = is_binary)
      if (is_binary) attr(pr, "probabilities")[,2] else as.numeric(pr)
    }
  }, metric = metric_svm)
  
  
  # 5) Torch NN (multi-layer)
  nn_try <- tryCatch({
    train_torch_nn(z_train, matrix(y_train, ncol = 1),
                   z_val, matrix(y_val, ncol = 1),
                   hidden = c(32, 16), epochs = 30, lr = 0.01,
                   batch_size = min(32, nrow(z_train)), is_binary = is_binary)
  }, error = function(e) NULL)
  
  pred_nn <- if (!is.null(nn_try)) nn_try$predfun(z_val) else rep(mean(y_train), nrow(z_val))
  metric_nn <- if (is_binary) tryCatch(as.numeric(pROC::auc(pROC::roc(y_val, pred_nn))), error = function(e) NA) else mean((y_val - pred_nn)^2)
  out$TorchNN <- list(predfun = if (!is.null(nn_try)) nn_try$predfun else function(newz) rep(mean(y_train), nrow(as.data.frame(newz))),
                      metric = metric_nn)
  
  return(out)
}

tgml_weak_learner_single_split <- function(y, x, z, min_samples = 10, max_candidates = 30, is_binary = FALSE, seed = 1) {
  set.seed(seed)
  n <- length(y)
  if (n < 2 * min_samples) stop("Not enough samples for weak learner")
  best_score <- if (is_binary) -Inf else Inf
  best <- NULL
  
  for (j in seq_len(ncol(x))) {
    vals <- unique(x[, j])
    if (length(vals) <= 1) next
    cand <- unique(quantile(vals, probs = seq(0.05, 0.95, length.out = max_candidates)))
    for (v in cand) {
      left_idx <- which(x[, j] <= v)
      right_idx <- which(x[, j] > v)
      if (length(left_idx) < min_samples || length(right_idx) < min_samples) next
      
      Ltrain_idx <- sample(left_idx, size = max(2, floor(0.7 * length(left_idx))))
      Lval_idx <- setdiff(left_idx, Ltrain_idx)
      if (length(Lval_idx) < 1) Lval_idx <- left_idx
      
      Rtrain_idx <- sample(right_idx, size = max(2, floor(0.7 * length(right_idx))))
      Rval_idx <- setdiff(right_idx, Rtrain_idx)
      if (length(Rval_idx) < 1) Rval_idx <- right_idx
      
      left_cand <- train_and_eval_candidates(y[Ltrain_idx], z[Ltrain_idx,,drop=FALSE], y[Lval_idx], z[Lval_idx,,drop=FALSE], is_binary)
      right_cand <- train_and_eval_candidates(y[Rtrain_idx], z[Rtrain_idx,,drop=FALSE], y[Rval_idx], z[Rval_idx,,drop=FALSE], is_binary)
      
      if (is_binary) {
        l_best_name <- names(left_cand)[which.max(sapply(left_cand, function(x) x$metric))]
        r_best_name <- names(right_cand)[which.max(sapply(right_cand, function(x) x$metric))]
        l_best <- left_cand[[l_best_name]]; r_best <- right_cand[[r_best_name]]
        val_idx_all <- c(Lval_idx, Rval_idx)
        preds_all <- numeric(length(val_idx_all))
        preds_all[1:length(Lval_idx)] <- l_best$predfun(z[Lval_idx,,drop=FALSE])
        preds_all[(length(Lval_idx)+1):length(val_idx_all)] <- r_best$predfun(z[Rval_idx,,drop=FALSE])
        combined_metric <- tryCatch(as.numeric(pROC::auc(pROC::roc(y[val_idx_all], preds_all))), error = function(e) 0.5)
        score_here <- combined_metric
        better <- (score_here > best_score)
      } else {
        l_best_name <- names(left_cand)[which.min(sapply(left_cand, function(x) x$metric))]
        r_best_name <- names(right_cand)[which.min(sapply(right_cand, function(x) x$metric))]
        l_best <- left_cand[[l_best_name]]; r_best <- right_cand[[r_best_name]]
        val_idx_all <- c(Lval_idx, Rval_idx)
        preds_all <- numeric(length(val_idx_all))
        preds_all[1:length(Lval_idx)] <- l_best$predfun(z[Lval_idx,,drop=FALSE])
        preds_all[(length(Lval_idx)+1):length(val_idx_all)] <- r_best$predfun(z[Rval_idx,,drop=FALSE])
        combined_metric <- mean((y[val_idx_all] - preds_all)^2)
        score_here <- combined_metric
        better <- (score_here < best_score)
      }
      
      if (is.null(best) || better) {
        best_score <- score_here
        best <- list(split_var = colnames(x)[j], split_val = v,
                     left = l_best, right = r_best,
                     left_idx = left_idx, right_idx = right_idx,
                     internal_val_idx = val_idx_all, combined_metric = combined_metric)
      }
    }
  }
  
  if (is.null(best)) {
    base_val <- mean(y)
    predfun <- function(newz) rep(base_val, nrow(as.data.frame(newz)))
    return(list(is_stump = TRUE, predfun = predfun))
  }
  class(best) <- "TGMLWeak"
  return(best)
}

predict_tgml_weak <- function(weak, x_new, z_new) {
  if (is.null(weak) || (!is.null(weak$is_stump) && weak$is_stump)) {
    return(weak$predfun(z_new))
  }
  x_new <- as.data.frame(x_new); z_new <- as.data.frame(z_new)
  preds <- numeric(nrow(x_new))
  var <- weak$split_var; val <- weak$split_val
  left_pos <- which(x_new[[var]] <= val)
  right_pos <- which(x_new[[var]] > val)
  if (length(left_pos) > 0) preds[left_pos] <- weak$left$predfun(z_new[left_pos,,drop=FALSE])
  if (length(right_pos) > 0) preds[right_pos] <- weak$right$predfun(z_new[right_pos,,drop=FALSE])
  preds
}

tgml_boosting_regression <- function(y, x, z, M = 50, min_samples = 10, max_candidates = 30, seed = 1) {
  n <- length(y)
  w <- rep(1/n, n)
  learners <- vector("list", M)
  alphas <- numeric(M)
  
  for (m in seq_len(M)) {
    set.seed(seed + m)
    idx <- sample(seq_len(n), size = n, replace = TRUE, prob = w)
    
    wl <- tgml_weak_learner_single_split(y[idx], x[idx,,drop=FALSE], z[idx,,drop=FALSE],
                                         min_samples = min_samples, max_candidates = max_candidates,
                                         is_binary = FALSE, seed = seed + m)
    
    pred_full <- predict_tgml_weak(wl, x, z)
    err_vec <- abs(y - pred_full)
    err_m <- sum(w * err_vec) / sum(w)
    
    if (err_m >= 0.5) break
    
    beta <- err_m / (1 - err_m)
    alpha <- log(1 / beta)
    
    w <- w * (beta ^ (1 - err_vec / max(err_vec)))
    w <- w / sum(w)
    
    learners[[m]] <- wl
    alphas[m] <- alpha
  }
  
  keep <- alphas > 0
  list(learners = learners[keep], alphas = alphas[keep])
}

predict_tgml_boost_reg <- function(boost, x_new, z_new) {
  if (length(boost$learners) == 0) return(rep(NA, nrow(x_new)))
  preds <- sapply(boost$learners, function(wl) predict_tgml_weak(wl, x_new, z_new))
  rowSums(t(t(preds) * boost$alphas)) / sum(boost$alphas)
}


                  
tgml_boosting_classification <- function(y, x, z, M = 50, min_samples = 10, max_candidates = 30, seed = 1) {
  n <- length(y)
  w <- rep(1/n, n)                    
  learners <- vector("list", M)
  alphas   <- numeric(M)

  for (m in seq_len(M)) {
    set.seed(seed + m)

  
    idx <- sample(seq_len(n), size = n, replace = TRUE, prob = w)

    
    wl <- tgml_weak_learner_single_split(
      y = y[idx],
      x = x[idx, , drop = FALSE],
      z = z[idx, , drop = FALSE],
      min_samples = min_samples,
      max_candidates = max_candidates,
      is_binary = TRUE,
      seed = seed + m
    )

   
    if (is.null(wl) || (!is.null(wl$is_stump) && wl$is_stump)) {
      break
    }

    
    p <- predict_tgml_weak(wl, x, z)
    p <- pmax(pmin(p, 1 - 1e-8), 1e-8)  

    
    err_vec <- ifelse(y == 1, 1 - p, p)
    err_m   <- sum(w * err_vec) / sum(w)

    
    if (err_m >= 0.5 || is.na(err_m)) break

    
    beta  <- err_m / (1 - err_m)
    alpha <- log(1 / beta)

    
    w <- w * exp(alpha * (2 * (y != (p > 0.5)) - 1))  
    w <- w / sum(w)                                   
    
    learners[[m]] <- wl
    alphas[m]     <- alpha
  }

  
  keep <- alphas > 0
  list(learners = learners[keep], alphas = alphas[keep])
}

predict_tgml_boost_bin <- function(boost, x_new, z_new) {
  if (length(boost$learners) == 0) return(rep(0.5, nrow(as.data.frame(x_new))))

  logit <- rowSums(sapply(seq_along(boost$learners), function(i) {
    p <- predict_tgml_weak(boost$learners[[i]], x_new, z_new)
    p <- pmax(pmin(p, 1 - 1e-8), 1e-8)
    boost$alphas[i] * log(p / (1 - p))
  }))

  plogis(logit)  
}


                  

run_pml_boosting <- function(n = 1000, p = 10, q = 10, nsim = 20, n_iter = 30, min_samples = 25) {
  results <- data.table::data.table(Scenario = integer(), Seed = integer(), Model = character(), Metric = numeric(), Metric_Type = character(), CPU = numeric())
  
  for (sc in 1:6) {
    is_binary <- sc > 3
    for (seed in 1:nsim) {
      cat(sprintf("Scenario %d Seed %d\n", sc, seed))
      dat <- generate_data(scenario = sc, n = n, p = p, q = q, seed = seed)
      y <- dat$y; x <- dat$x; z <- dat$z
      
      set.seed(seed + 1000)
      train_idx <- sample(seq_len(n), size = floor(0.7 * n))
      test_idx <- setdiff(seq_len(n), train_idx)
      y_train <- y[train_idx]; x_train <- x[train_idx,,drop=FALSE]; z_train <- z[train_idx,,drop=FALSE]
      y_test <- y[test_idx]; x_test <- x[test_idx,,drop=FALSE]; z_test <- z[test_idx,,drop=FALSE]
      
      # Baseline models
      t_base <- proc.time()
      bas <- baseline_models_eval(y_train, z_train, y_test, z_test)
      t_base_elapsed <- as.numeric((proc.time() - t_base)["elapsed"])
      for (mname in names(bas)) {
        results <- rbind(results, list(sc, seed, mname, bas[[mname]], ifelse(is_binary, "AUC", "MSE"), t_base_elapsed))
      }
      
      # Boosting
      t_boost <- proc.time()
      if (!is_binary) {
        boost_model <- tgml_boosting_regression(y_train, x_train, z_train, M = n_iter, min_samples = min_samples, seed = seed)
        preds_test <- predict_tgml_boost_reg(boost_model, x_test, z_test)
        metric_boost <- mean((y_test - preds_test)^2)
      } else {
        boost_model <- tgml_boosting_classification(y_train, x_train, z_train, M = n_iter, min_samples = min_samples, seed = seed)
        preds_test <- predict_tgml_boost_bin(boost_model, x_test, z_test)
        metric_boost <- tryCatch(as.numeric(pROC::auc(pROC::roc(y_test, preds_test))), error = function(e) NA)
      }
      t_boost_elapsed <- as.numeric((proc.time() - t_boost)["elapsed"])
      results <- rbind(results, list(sc, seed, ifelse(is_binary, "PML_Boosting_Binary", "PML_Boosting_Regression"),
                                     metric_boost, ifelse(is_binary, "AUC", "MSE"), t_boost_elapsed))
    }
  }
  return(results)
}

baseline_models_eval <- function(y_train, z_train, y_test, z_test) {
  is_binary <- length(unique(y_train)) == 2
  df_train <- as.data.frame(z_train); df_test <- as.data.frame(z_test)
  res <- list()
  
  # Linear
  if (is_binary) {
    fit <- tryCatch(glm(y_train ~ ., data = df_train, family = binomial), error = function(e) NULL)
    pred <- if (!is.null(fit)) predict(fit, newdata = df_test, type = "response") else rep(0.5, nrow(df_test))
    res$Linear <- tryCatch(as.numeric(pROC::auc(pROC::roc(y_test, pred))), error = function(e) NA)
  } else {
    fit <- tryCatch(lm(y_train ~ ., data = df_train), error = function(e) NULL)
    pred <- if (!is.null(fit)) predict(fit, newdata = df_test) else rep(mean(y_train), nrow(df_test))
    res$Linear <- mean((y_test - pred)^2)
  }
  
  # Lasso
  fit_lasso <- tryCatch(cv.glmnet(as.matrix(z_train), y_train, family = ifelse(is_binary, "binomial", "gaussian")), error = function(e) NULL)
  pred <- if (!is.null(fit_lasso)) as.numeric(predict(fit_lasso, newx = as.matrix(z_test), s = "lambda.min", type = ifelse(is_binary, "response", "link"))) else rep(mean(y_train), nrow(z_test))
  res$Lasso <- if (is_binary) tryCatch(as.numeric(pROC::auc(pROC::roc(y_test, pred))), error = function(e) NA) else mean((y_test - pred)^2)
  
  res
}

set.seed(42)
res_demo <- run_pml_boosting(n = 500, p = 8, q = 8, nsim = 3, n_iter = 10, min_samples = 20)
summary_dt <- res_demo[, .(Avg_Metric = mean(Metric, na.rm = TRUE), SD_Metric = sd(Metric, na.rm = TRUE)), by = .(Scenario, Model, Metric_Type)]
print(summary_dt)



