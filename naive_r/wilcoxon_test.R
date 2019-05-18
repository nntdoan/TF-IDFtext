library(dplyr)
library(stats)
library(ggpubr)
library(psych)

# LOAD THE DATA
raw_long <- read.csv("exp_result_verbs_long.csv")
View(raw_long)
# The data looks like this
#       categories        similarity_score
# 1     same_genre        0.02
# 2     same_genre        0.02
# 3     same_genre        0.08
# ....
# 2451  different_genres  0.02
# 2452  different_genres  0.03
# 2453  different_genres  -0.02

raw_wide <- read.csv("exp_result_verbs_wide.csv")
View(raw_wide)
# The data looks like this
#   measurement same_genre different_genres
# 1 similarity  0.02        0.02
# 2 similarity  0.02        0.03
# 3 similarity  0.08        -0.02
# ...

# The two dataframes are fundamentally the same, just different format

# ------------------------------------------------------
# SUMMARY OF DATA
group_by(raw_long, categories) %>%
  summarise(
    count = n(),
    median = median(similarity_score, na.rm = TRUE),
    IQR = IQR(similarity_score, na.rm = TRUE)
  )
# output:
# A tibble: 2 x 4
#   categories       count median   IQR
#   <fct>            <int>  <dbl> <dbl>
# 1 different_genres  2500   0.05  0.1 
# 2 same_genre        2450   0.06  0.10

describeBy(raw_long, "categories")
# output:
#  Descriptive statistics by group 
# group: different_genres
# vars    n mean   sd median trimmed  mad   min max range skew kurtosis se
# categories*         1 2500 1.00 0.00   1.00    1.00 0.00  1.00 1.0  0.00  NaN      NaN  0
# similarity_score    2 2500 0.07 0.08   0.05    0.06 0.06 -0.15 0.8  0.95 1.49     4.41  0
# --------------------------------------------------------------------------------- 
# group: same_genre
# vars    n mean   sd median trimmed  mad   min  max range skew kurtosis se
# categories*         1 2450 2.00 0.00   2.00    2.00 0.00  2.00 2.00  0.00  NaN      NaN  0
# similarity_score    2 2450 0.07 0.09   0.06    0.06 0.07 -0.14 0.74  0.88  1.4     3.85  0

# ------------------------------------------------------
# CREATE BOX-PLOTS
ggboxplot(raw_long, x = "categories", y = "similarity_score", color = "categories", palette = c("#808080", "#000000"),
          ylab = "Similarity scored by general LSA space", xlab = "Categories of verbs") + 
scale_x_discrete(labels = c('Different genres','Same genre'))


# ------------------------------------------------------
# Perform Unpaired Two-Samples Wilcoxon Test
test_verb <- wilcox.test(raw_wide$same_genre, raw_wide$different_genres, na.rm = TRUE, alternative = "two.sided")
test_verb
# output:
# 	Wilcoxon rank sum test with continuity correction
# data:  raw_wide$same_genre and raw_wide$different_genres
# W = 3140100, p-value = 0.1223
# alternative hypothesis: true location shift is not equal to 0

# References:
# http://www.sthda.com/english/wiki/unpaired-two-samples-wilcoxon-test-in-r
# https://www.stat.auckland.ac.nz/~wild/ChanceEnc/Ch10.wilcoxon.pdf 
# https://stats.idre.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/
# http://www.sthda.com/english/wiki/paired-samples-t-test-in-r
