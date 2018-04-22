
# carico la libreria per generare le rappresentazioni
import sys
import numpy as np

import SeqLib as seqlib



# coding: utf-8
# struttura rapporto in output 

# insieme delle lunghezze delle sequenze
lunghezze=[]
# numero di sequenze con caratteri IUPAC
contaStrani=0
# numero di sequenze in totale
contaTotale = 0




directory="./"

listaFile=[	"Mouse_Promoter.fas",
			"Mouse_chr.fas",
			"Mouse_5UTRExon.fas"]

for nomeFile in listaFile:

	Nome = nomeFile.split(".")[0]

	print Nome
	seqNome=directory+Nome +".fas"
	from Bio import SeqIO
	import pandas as pd


	# In[5]:

	# carico le sequenze usando Biopython
	seqDict={}
	handle = open(seqNome, "rU")
	i=0
	for record in SeqIO.parse(handle, "fasta"):
		#estrae il tipo della sequenza dalla stringa ">Drosophila 5UTRExon link"
		tipo= record.description.split()[2]
		
		nomeRecord = record.id +"."+str(i)+"."+ tipo
		seqDict[nomeRecord]=str(record.seq)
		i += 1
	handle.close()


	# controllo la lunghezza dele sequenze
	ll=[]
	for x in seqDict.keys():
		seq=seqDict[x]
		ll.append(len(seq))
	lunghezze=set(ll)
	print lunghezze




	# crea la rappresentazione in un dict
	rapp=[]
	contaStrani=0
	contaTotale=0
	for x in seqDict.keys():
		seq=seqDict[x]
		contaTotale=contaTotale + 1
		if seqlib.isIUPAC(seq) or len(seq) != 151:
			contaStrani = contaStrani+1
		else:
			if "nuc" in x:
				label=1
			else:
				label=0
			m=seqlib.conv4X(seq)
			a=np.reshape(m, (1,-1), order='F') # strasforma in una matrice con una sola riga 
			aa=map(str,a.tolist()[0])  #trasforma in una lista di stringhe
			ss=str(x)+";"+ ", ".join(aa)+";"+str(label)
			rapp.append(ss)
	print "finito"

	print "numero sequenze difettose: ", str(contaStrani), "Numero totale di sequenze", str( contaTotale )


	# scrive il rapporto sul file fasta
	fout=open(directory+Nome+"_rapporto.txt", "w")
	fout.write("Numero totale di sequenze : "+ str(contaTotale) + "\n")
	fout.write("Numero di sequenze con caratteri IUPAC : " +str(contaStrani) +"\n")
	fout.write("Lunghezze delle sequenze :")
	llss=[str(x) for x in lunghezze]
	fout.write(",".join(llss)+"\n")
	fout.close()


	# Salva la rappresentazione delle sequenze

	fout=open(directory+Nome+"_rapp.csv", "w")
	for r in rapp:
		fout.write(r +"\n")
	fout.close()





