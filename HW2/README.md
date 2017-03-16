The .R files leveraged the TSMining package. Once the motif discovery or SAX conversions were done, the data were written to text files. Python was used for the rest of the processing and classificaiton.
The file descriptions are as follows:
motif-discovery.R was used for experiment (2), which is subsequence motif discovery
SAXwords.R was used for convering time series to SAX
bop-classification.py was used for experiment (3), bag of patterns given multiple SAX words
motif-classification.py was used for classying the subsequence motifs in experiment (2)
pair-motif-classification.py is for classifying time series pair motifs in experiment (1)
plit-data.py is a preprocessing data script