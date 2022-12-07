import sys, fileinput
from sacremoses import MosesTokenizer

t = MosesTokenizer(lang='en')

if __name__ == "__main__":
    for line in fileinput.input():
        if line.strip() != "":
            tokens = t.tokenize(line.strip(), escape=False)

            sys.stdout.write(" ".join(tokens) + "\n")
        else:
            sys.stdout.write('\n')
