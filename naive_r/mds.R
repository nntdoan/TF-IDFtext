library(vegan)

romance.raw <- read.csv("lsa_x100terms_v1.csv")
row.names(romance.raw) <- colnames(romance.raw)

mds.cor.vegan <- (1 - romance.raw)

group <- factor(rep(c("Romance", "Non-romance"), each = 50))
cols <- c("black", "red")

set.seed(2)
data.nmds <- metaMDS(mds.cor.vegan, trace=FALSE, k=2)

plot(data.nmds, type = 'n', xlab="Dimension 1", ylab="Dimension 2")
text(data.nmds, display = "sites", col = cols[group])
legend("topright", legend = tools::toTitleCase(levels(group)), 
       fill = cols, bty = "n")

stressplot(data.nmds)

# Adapted from: 
# https://stackoverflow.com/questions/54533100/how-to-plot-sites-in-different-colours-in-a-nmds-plot-in-r-vegan-package
# http://www.flutterbys.com.au/stats/tut/tut15.1.html
# https://jonlefcheck.net/2012/10/24/nmds-tutorial-in-r/ 
# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/122-multidimensional-scaling-essentials-algorithms-and-r-code/
# https://jonlefcheck.net/2012/10/24/nmds-tutorial-in-r/
