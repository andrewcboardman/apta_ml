Contains tools for processing FASTA files of sequences to be ready for learning

Workflow 1: single FASTA to single csv (for Ising)
python encode.py -i in.fasta -o out.csv

Workflow 2: two FASTAs to one csv with labels
python label_fasta.py -i in1.fasta -o out1.fasta -l 1
python label_fasta.py -i in2.fasta -o out2.fasta -l 2
cat out1.fasta out2.fasta | perl seq-shuf.pl > out.fasta
python encode.py -i out.fasta -o out.csv

Workflow 3: Labelled csv to HDF5 archive with train & test datasets
python to_hdf5.py -i in.csv -o out.hdf5
