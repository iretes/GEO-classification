dataset="GSE13355" # put here the gene series number

library(GEOquery)
data<-getGEO(dataset, GSEMatrix = TRUE)
fenoData=fData(data[[1]])

pfeature="characteristics_ch1" # put here the target field
molecule_ID = "GB_ACC"

d=exprs(data[[1]])

hist(d) # visualise distribution

d <- rbind(pData(data[[1]])[[pfeature]], d)
rownames(d)=c("TARGET",fData(data[[1]])[[molecule_ID]])

write.table(d,paste(dataset,"annot.csv",sep=''),sep=",")

