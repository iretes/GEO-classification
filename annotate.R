args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Wrong command line arguments")
}

dataset=args[1]

if (!require("GEOquery")) {
  if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
  BiocManager::install("GEOquery")
}

library(GEOquery)
data<-getGEO(dataset, GSEMatrix = TRUE)
fenoData=fData(data[[1]])

pfeature=args[2]
molecule_ID = "GB_ACC"

d=exprs(data[[1]])

d <- rbind(pData(data[[1]])[[pfeature]], d)
rownames(d)=c("TARGET",fData(data[[1]])[[molecule_ID]])

write.table(d,args[3],sep=",")

