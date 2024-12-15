library(survival)
library(dplyr)
library(tidyr)
library(ggplot2)


plot_price_demand <- function(
    fit, income, xlabel = "WTP", ylabel = "Demand", legend_title = "Income", ...
) {
  pct <- 1:99/100  ##定义死亡率
  # income <- dat_used$income %>% median
  ptime <- predict(fit, newdata = data.frame(income=income), type='quantile', p=pct, se=TRUE)
  pred <- data.frame(
    time = as.vector(t(ptime$fit)),
    income = rep(income, each = 99) %>% as.factor,
    p = rep(1 - pct, length(income))
  )
  # ymin=ptime$fit -1.96*ptime$se.fit,
  # ymax=ptime$fit +1.96*ptime$se.fit,
  # p=1-pct)

  if (length(income) > 1) {
    pred %>% ggplot() +
      geom_line(aes(x=time, y=p, colour = income), ...) +
      labs(x = xlabel, y = ylabel, colour = legend_title) +
      theme_bw()
  } else {
    pred %>% ggplot() +
      geom_line(aes(x=time, y=p), ...) +
      labs(x = xlabel, y = ylabel) +
      theme_bw()
  }

  return (pred)

}

get_demand <- function(fit, price, income) {
  shape <- 1 / fit$scale
  pred <- predict(fit, newdata = data.frame(income = income), type = "linear")
  1 - pweibull(price, shape = shape, scale = exp(pred))
}

get_price <- function(fit, demand, income) {
  pct <- 1 - demand
  ptime <- predict(fit, newdata = data.frame(income=income), type='quantile', p=pct)
  ptime
}


dat <- haven::read_dta("../data/合并_787.dta")
dat_used <- dat %>%
  select(income = 家庭月平均收入元, WTP = 国产二价HPV疫苗愿意承受的最高价格是全程) %>%
  mutate(status = 1)

# mean(dat_used$income)

# 收入分布
dat_used %>% ggplot() +
  geom_histogram(aes(x = income)) +
  # geom_density(aes(x = income), colour = "red") +
  theme_bw()
# 异常值处理（盖帽法），单侧
income_thre <- quantile(dat_used$income, 0.95)
dat_used <- dat_used %>%
  mutate(income_wo_outlier = if_else(income > income_thre, income_thre, income))
# 重新看一下分布
dat_used %>% pivot_longer(c(income, income_wo_outlier), names_to = "name", values_to = "income_value") %>%
  ggplot() +
  # geom_histogram(aes(x = income_value, fill = name), position = "identity", alpha = 0.7) +
  geom_density(aes(x = income_value, colour = name)) +
  theme_bw()

# WTP为0的样本数量
(dat_used$WTP == 0) %>% sum
# [1] 4

# 拟合模型
fit1 <- survreg(
  Surv(WTP, status) ~ income,
  data=dat_used %>% filter(WTP > 0) %>% mutate(income = income_wo_outlier),
  dist = "weibull"
)
summary(fit1)

# 家庭人均月可支配收入（2019）
income_2019 <- 30733 / 12 * 2  # median(dat_used$income_wo_outlier)
print(income_2019)


# 1. 单条曲线
pred_df_1 <- plot_price_demand(
  fit1, income = income_2019,
  size = 1.0, xlabel = "疫苗价格(元)", ylabel = "疫苗需求"
)
ggsave("./results/价格需求函数1.png",
       width = 5, height = 5, dpi = 300, limitsize = FALSE)
write.csv(pred_df_1, file = "./results/price_demand_function1.csv")

# 2. 不同收入
incomes <- quantile(dat_used$income_wo_outlier, c(0, 0.07, 0.15, 0.25, 0.5, 0.75, 1.0))
pred_df_2 <- plot_price_demand(
  fit1, income = incomes,
  size = 1.0, xlabel = "疫苗价格(元)", ylabel = "疫苗需求",
  legend_title = "收入(元)"
)
ggsave("./results/价格需求函数2.png",
       width = 5, height = 5, dpi = 300, limitsize = FALSE)
write.csv(pred_df_2, file = "./results/price_demand_function2.csv")

# 3. 得到指定疫苗价格下的需求
get_demand(fit1, price = c(500, 670, 1000, 2000), income = income_2019)
# (median)[1] 0.83384580 0.48284920 0.05408996
# (家庭月平均收入) 0.80108810 0.67131357 0.41121832 0.02842436

# 4. 得到指定需求下的价格
get_price(fit1, demand = 0.6782, income = income_2019)
# 1
# 661.3756