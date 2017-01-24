import csv
import sys

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

stop_words = stopwords.words('english')
#retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')

def main(infile, outfile):
	with open(infile, 'r') as notes:
		with open(outfile, 'w') as out:
			reader = csv.reader(infile)
			i = 0
			for line in reader:
				if i % 10000 == 0:
					print(i)
				#probably okay to normalize to lowercase. medical terms written in lowercase.
				tokens = [t.lower() for t in tokenizer.tokenize(line[1]) if not t.isnumeric() and t.lower() not in stop_words]
				text = ' '.join(tokens)
				outfile.write(','.join([line[0], text]) + '\n')

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("usage: python tokenize_notes.py infile_name outfile_name")
		sys.exit(0)
	infile = sys.argv[1]
	outfile = sys.argv[2]
	main(infile, outfile)