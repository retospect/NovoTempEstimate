This dataset contains protein sequences used to train, validate, and test binary classifiers that form TemStaPro program, which is applied for protein thermostability prediction with respect to nine temperature thresholds from 40 to 80 degrees Celsius using a step of five degrees.

The data is given in files of FASTA format. Each protein sequence has a header made of three values separated by vertical bar symbols: organism's, to which the protein belongs, UniParc taxonomy identifier; UniProtKB/TrEMBL identifier of the protein sequence; organism's growth temperature taken from the dataset of growth temperatures of over 21 thousand organisms (Engqvist, 2018).

TemStaPro-Major-30 set is composed of 12 files:

one training
one validation
one imbalanced testing
nine balanced samples of 2000 sequences from each of the balanced testing set
TemStaPro-Minor-30 set is composed of cross-validation and testing files all balanced for 65 degrees Celsius temperature threshold.

SupplementaryFileC2EPsPredictions.tsv file contains thermostability predictions using the default mode of TemStaPro program to check the thermostability of different C2EP groups.

The detailed description is given in the revised version of the corresponding paper (https://doi.org/10.1093/bioinformatics/btae157).

If you use the data from this dataset, please cite both the paper and the DOI of the dataset.
Other
This project has received funding from European Regional Development Fund (project No 13.1.1-LMT-K-718-05-0021) under grant agreement with the Research Council of Lithuania (LMTLT). Funded as European Union's measure in response to COVID-19 pandemic.

