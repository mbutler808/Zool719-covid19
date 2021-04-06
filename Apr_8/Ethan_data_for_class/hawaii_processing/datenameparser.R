require(ape)
require(stringr)

dat <- read.FASTA("hawaii_all.fasta")
name <- names(dat)

locations <- str_extract(name, "[^_]+$")
dates <- str_extract_all(name, "[0-9]{4}_[0-9]{2}_[0-9]{2}")
dates <- str_replace_all(dates, "_", "/")
dates <- format(as.Date(dates, format = "%Y/%m/%d"), "%m/%d/%Y")

id <- str_remove_all(name, "_[0-9]{4}_[0-9]{2}_[0-9]{2}")

acc <- str_extract(name, "EPI_ISL_.*$")
acc <- str_remove_all(acc, "_[0-9]{4}_[0-9]{2}_[0-9]{2}")

names(dat) <- id

write.FASTA(dat, "hawaii_all_cleaned.fasta")

namedatgeo <- cbind(acc, dates)
write.table(namedatgeo, file="HI_collection_dates.txt", sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE, quote=F)


acc <- str_extract(name, "EPI_ISL_.*$")
acc <- str_remove_all(acc, "_[0-9]{4}_[0-9]{2}_[0-9]{2}")
write.table(acc, file="HI_ids.txt", sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE, quote=F)
