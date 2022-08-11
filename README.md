# Transformer
# Language: Python
# Input: TXT
# Output: PNG
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin that runs Transformer model (Vaswani et al, 2017)

The plugin expectes as input a tab-delimited file of keyword-value pairs:
inputfile: Dataset
divide: Row where dataset starts
pct: Percent of rows to use for training
lr: Learning rate
epochs: Number of epochs
column: Column to plot
