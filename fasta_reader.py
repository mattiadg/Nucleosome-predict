from Bio import SeqIO
import numpy as np


def get_dataset(species):
    assert species in ["elegans", "melanogaster", "sapiens"]
    prefix = "res/fasta_files/nucleosomes_vs_linkers_"
    handle = open(prefix + species + ".fas", "rU")
    seqs = []
    labels = []
    for record in SeqIO.parse(handle, "fasta"):
        seqs.append(record.seq)
        labels.append(record.id)
    handle.close()
    return np.array(seqs), np.array(labels)

if __name__ == '__main__':
    print(get_dataset('elegans'))