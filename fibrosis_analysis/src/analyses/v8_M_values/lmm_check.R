# lme4 reproduction of the primary single-predictor models
# (run from the project root:  Rscript ./results/v8_M_values/lmm_check.R)
library(lme4)
dat <- read.csv("./results/v8_M_values/lmm_egfr_long.csv")
dat <- dat[complete.cases(dat[, c("recipient_age_tx", "recipient_sex_male", "m_density", "m_mean_iqr", "m_collagen_mean_mean", "collagen_ratio", "collagen_iqr", "f_mean_intensity_iqr", "collagen_ratio_otsu", "ce_density_otsu", "f_mean_intensity_iqr_otsu")]), ]
ctrl <- lmerControl(optimizer = "bobyqa", check.conv.singular = "warning")
base <- lmer(egfr ~ (time + time2) * (recipient_age_tx + recipient_sex_male) + (1 + time | id),
             data = dat, REML = FALSE, control = ctrl)
print(summary(base)$coefficients)
for (p in c("m_density", "m_mean_iqr", "m_collagen_mean_mean", "collagen_ratio", "collagen_iqr", "f_mean_intensity_iqr", "collagen_ratio_otsu", "ce_density_otsu", "f_mean_intensity_iqr_otsu")) {
  f <- as.formula(paste0("egfr ~ (time + time2) * (recipient_age_tx + recipient_sex_male + ", p, ") + (1 + time | id)"))
  m <- lmer(f, data = dat, REML = FALSE, control = ctrl)
  a <- anova(base, m)
  cat(sprintf("%-26s LR = %6.3f  df = %d  p = %.4f\n", p, a$Chisq[2], a$Df[2], a$`Pr(>Chisq)`[2]))
}
