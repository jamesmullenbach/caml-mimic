"""
	Put data in a better format: labels not as separate columns (and therefore variable # cols)
	but in one column, separated by ;

	shouldn't need this anymore as of 2/9
"""
import csv
import os
import sys

def main(filename):
	re_delimit(filename)

def re_delimit(filename):
	with open(filename, 'r') as infile:
		outname = filename.replace('.csv', '_fixed.csv')
		with open(outname, 'w') as outfile:
			outfile.write(",".join(['SUBJECT_ID', 'TEXT', 'LABELS']) + "\n")
			reader = csv.reader(infile)
			next(reader)
			for line in reader:
				label_str = ';'.join(line[2:])
				outfile.write(",".join([str(line[0]), line[1], label_str]) + "\n")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("usage: python " + str(os.path.basename(__file__) + " [filename]"))
		sys.exit(0)
	main(sys.argv[1])
