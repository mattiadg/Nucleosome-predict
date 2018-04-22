#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  SeqLib.py
#  
#  Copyright 2016 riccardo rizzo <riccardo@yorick>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
from Bio import SeqIO
from itertools import *






def convLetter(nucleotide):
	"""
	Converte la lettera in un vettore di 4 elementi
	"""
	code={	"A":[0], 
			"C":[1], 
			"G":[2], 
			"T":[3],
			"a":[0],
			"c":[1],
			"g":[2],
			"t":[3]}
	
	out=np.zeros(4)
	for p in code[nucleotide]:
		out[p]=1
	
	return out
	
def convKmer(kmer):
	"""
	Converte la stringa di lunghezza L in una matrice 4 x L
	"""
	ll=[]
	for l in kmer:
		ll.append( convLetter(l) )
	
	out=np.vstack(ll).transpose()
	return out
	
	
def convSequenza(sequenza, k):
	"""
	Converte la sequenza in una lista di array di lunghezza 4 x k
	"""
	out=[]
	for i in range(0, len(sequenza)-k+1):
		ss=sequenza[i:i+k]
		out.append(convKmer(ss))

	return out


def readFasta(listaStringhe):
	"""
	Legge una lista di stringhe che contiene una seq in formato fasta 
	e restituisce una coppia [nome, sequenza]
	"""
	nome=listaStringhe.pop(0)
	out="".join(listaStringhe)
	return [nome, out]
	

def readFastaFile(nomeFile):
	"""
	Legge un intero fasta file e mette tutto in un dictionary
	"""
	out={}
	fasta_sequences = SeqIO.parse(open(nomeFile),'fasta')

	for fasta in fasta_sequences:
		
		name, sequence = fasta.id, fasta.seq.tostring()

		out[name]= sequence
	return out

	

def conv4X(seq):
	"""
	Converte la sequenza in una rappresentazione 4X
	La matrice che rappresenta la sequenza e' generata come matrice numpy
	La matrice ha dimensione 4xlen(seq)
	"""
	temp=[]
	for b in seq:
		aa=convLetter(b)
		temp.append(aa)
	st=np.column_stack(temp)
	
	return st




def seqFrom4X(mat):
	"""
	Converte la matrice in ingresso di dimensione 
	4Xlen(seq) in una sequenza
	Stampa un errore se la matrice ha un numero
	di righe diverso da 4
	
	L'INPUT PUO' ESSERE L'OUTPUT DI conv4x()
	E' IMPORTANTE CHE TUTTI GLI ELEMENTI DA SCARTARE SIANO
	EGUALI A ZERO
	
	"""
	code=("A", "C", "G", "T")
	# verifica il numero di righe
	R,C = np.shape(mat)

	if R != 4:
		print "numero di righe non corretto"
		return ""
	else:
		# fa la trasposta di mat
		mat=mat.T
		# mat adesso e' una matrice len(seq) X 4
		ll=mat.tolist()
		out=[]
		for l in ll:
			#print l
			a=np.nonzero(l)
			#print code[a[0][0]]
			out.append(code[a[0][0]])
		
		return "".join(out)



def genKmers(k):
	""" restiruisce una lista di tutti i kmers di ordine k """
	alphabet = 'ACGT'
	kmers = [''.join(i) for i in product(alphabet, repeat = k)]
	return kmers



def repKLambda(sequenza, k, off, k1):
	""" 
	genera la rappresentazione a kmers aggiungendo un salto
	pari a off.
	IL kmero e' considerato cosi'
	k=4, off=5, k1=2
	sequenza 		ACGTGCCACGTTTGCCCAT....
	kmero in pos 2	  ||.....||
					  GT     GT
						\   /
	kmero				GTGT
	
	Si lascia uno spazio di "off" dopo k1 caratteri del kmero
	(lambda=off per ragioni "storiche")
	"""

	seq_k = list(repeat(0, 4**k))
	k2=k-k1
	kmers=genKmers(k)
	for i in range(len(sequenza)-(k-1)-off):
		kmer = sequenza[i:i+k1]
		kmer += sequenza[i+k1+off:i+k1+off+k2]
		if (kmer in kmers):
			seq_k[kmers.index(kmer)] +=1
	return seq_k


def repQuadLambda(seq, k, k1):
	"""
	La rappresentazione crea un quadrato impilando dei vettori 
	di frequenza dei kmeri con lambda (gap) crescenti. I valori di 
	lambda vanno da 0 (ordinaria rappresentazione a k-mer) 
	a lambda=4**k-1. 
	
	L'output e' una matrice numpy. 
	
	il valore di k1 e' la posizione iniziale del gap segnato da lambda
	
	"""
	dim=4**k
	out=np.zeros( (dim,dim) )
	
	for l in range(dim):
		v=np.asarray(repKLambda(seq, k, l, k1))
		out[l]=v
	
	return out
		 
	
def isIUPAC(stringa):
	""" 
	controlla che la stringa (o lista di stringhe) non contenga caratteri IUPAC
	OUTPUT:
		True se la stringa contiene un carattere IUPAC
		False se la stringa e' puramente acgt
		
	"""
	test_characters = frozenset(r"MRWSYKVHDBXNmrwsykvhdbxn")
	# se l'input e' una lista di stringhe lo unisce in una sola'
	ss="".join(stringa)
	return bool(set(ss) & test_characters)




IUPAC_conversione= {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "M": "A",
    "R": "A",
    "W": "A",
    "S": "C",
    "Y": "C",
    "K": "G",
    "V": "A",
    "H": "A",
    "D": "A",
    "B": "C",
    "X": "G",
    "N": "G"   
}



def main(args):
	
	ii=""">Nucleosome sequence 1
GGAACCGGTACGGACTCAGGGAATCCGACTGTCTAATTAAAACAGAGGTGACAGATGGTCCT
TGCGGACGTTGACTGTCACTGATTTCTGCCCAGTGCTCTGAATGTTAAATCGTAGTAATTCG
AGTAAGCGCGGGTAAACGGCGGG"""
	
	input1= ii.split("\n")
	
	
	
	label, seq=readFasta(input1)
	print seq
	
	# collaudo conv4X seqFrom4X
	
	mat=conv4X(seq)
	print seqFrom4X(mat)
	
	##print seq.keys[0], seq[seq.keys[0]]
	
	##seq="acgccgtgaaaaaa"
	##rif="01234567890123"
	##print seq
	##print rif
	#kfreq= repKLambda(seq, 2, 1, 1)
	#i=0
	#for k in genKmers(2):
		#print k, kfreq[i]
		#i +=1
	
	#print isIUPAC(input1)

	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
	
	#print genKmers(4)
	

	
