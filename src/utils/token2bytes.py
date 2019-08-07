import sys
import re

def not_empty(s):
    return s and s.strip()

def run(input,output):
    with open(input, "r", encoding='utf-8') as f, open(
            output, "w+", encoding='utf-8') as d:
        for line in f:
            line = line.strip().split(" ",1)
            key, seq = line[0],line[1]
            #seq_=''
            #for char in seq:
            #    seq_ += ' ' + char
            seq_=list(filter(not_empty, re.split(r"\\x| ",str(seq.encode("utf-8"))[2:-1])))
            out=[]
            for x in seq_:
                if len(x)==1:
                    #out += str(ord(x))
                    out.append(ord(x))
                else:
                    #out += str(int(x,16))
                    out.append(int(x,16))
            #out.append('\n')
            #print(' '.join(str(i) for i in out))
            d.write("{} {}\n".format(key," ".join(str(i) for i in out)))


if __name__ == '__main__':
    input = sys.argv[1]
    output = sys.argv[2]
    run(input,output)
