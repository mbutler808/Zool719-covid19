require(ape)
require(stringr)

####Reads and Cleans international seq###
dat <- read.FASTA("inter_intro.fasta")
name <- names(dat)

locations <- str_extract(name, "[^_]+$")
dates <- str_extract_all(name, "[0-9]{4}_[0-9]{2}_[0-9]{2}")
dates <- str_replace_all(dates, "_", "/")
dates <- format(as.Date(dates, format = "%Y/%m/%d"), "%m/%d/%Y")
dates[6] <- "04/15/2020"
dates[18] <- "11/15/2020"

id <- str_remove(name, "_[0-9]{4}_[0-9]{2}_[0-9]{2}")
id <- str_remove(id, "_[^_]+$")
id <- str_remove_all(id, "hCoV_19_")
id[6] <- str_remove(id[6], "_[0-9]{4}_[0-9]{2}")
id[18] <- str_remove(id[18], "_[0-9]{4}_[0-9]{2}")

names(dat) <- id

write.FASTA(dat, "inter_intro_cleaned.fasta")

namedatgeo <- cbind(id, dates)
write.table(namedatgeo, file="inter_intro_collection_dates.txt", sep = "\t", dec = ".", row.names = FALSE, col.names = FALSE, quote=F)

####Reads and Cleans domestic seq###
dat <- read.FASTA("us_intro.fasta")
name <- names(dat)

locations <- str_extract(name, "[^_]+$")
dates <- str_extract_all(name, "[0-9]{4}_[0-9]{2}_[0-9]{2}")
dates <- str_replace_all(dates, "_", "/")
dates <- format(as.Date(dates, format = "%Y/%m/%d"), "%m/%d/%Y")
dates[6] <- "04/15/2020"
dates[18] <- "11/15/2020"

id <- str_remove(name, "_[0-9]{4}_[0-9]{2}_[0-9]{2}")
id <- str_remove(id, "_[^_]+$")
id <- str_remove_all(id, "hCoV_19_")

names(dat) <- id

write.FASTA(dat, "us_intro_cleaned.fasta")

namedatgeo <- cbind(id, dates)
write.table(namedatgeo, file="us_intro_collection_dates.txt", sep = "\t", dec = ".", row.names = FALSE, col.names = FALSE, quote=F)

