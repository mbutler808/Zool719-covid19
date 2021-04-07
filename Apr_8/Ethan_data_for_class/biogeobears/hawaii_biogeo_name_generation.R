require(ape)
require(phytools)

# Creating biogeo matrix for Hawaii analysis

# Reads in the fasta file
fasta <- read.FASTA(file = "hawaii_and_friends2_aligned.fasta")

# Extracts the fasta names 
allnames <- names(fasta)

# Parses the names based on their location and assigns them to either 1 or 0
oahu <- as.numeric(grepl("Oahu", allnames))
maui <- as.numeric(grepl("Maui", allnames))
kauai <- as.numeric(grepl("Kauai", allnames))
hawaii <- as.numeric(grepl("Hawaii", allnames))
nothi_usa <- as.numeric(grepl("USA_", allnames) & grepl("..", allnames) & !grepl("HI", allnames))
nothi_inter <- as.numeric(!grepl("USA", allnames))
nothi_inter <- as.numeric(grepl("Australia", allnames) | grepl("Netherlands", allnames) | grepl("India", allnames) | grepl("England", allnames) | grepl("Switzerland", allnames) | grepl("Canada", allnames) | grepl("Wales", allnames) | grepl("Japan", allnames) | grepl("France", allnames) | grepl("Sweden", allnames) | grepl("NC_045512_2", allnames))

# Renames the columns to single letters
location <- data.frame(allnames, oahu, maui, kauai, hawaii, nothi_usa, nothi_inter)
names(location)[names(location) == 'oahu'] <- 'o'
names(location)[names(location) == 'maui'] <- 'm'
names(location)[names(location) == 'kauai'] <- 'k'
names(location)[names(location) == 'hawaii'] <- 'h'
names(location)[names(location) == 'nothi_usa'] <- 'u'
names(location)[names(location) == 'nothi_inter'] <- 'i'
names(location)[names(location) == 'allnames'] <- ''

# writes .tsv
write.table(location, file="sars2_geoloc_hawaii.txt", sep = "\t", dec = ".", row.names = FALSE, col.names = TRUE, quote=F)
