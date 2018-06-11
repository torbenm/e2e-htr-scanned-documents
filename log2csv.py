import sys

if len(sys.argv) < 3:
    print "Please give a logfile path and an output path"
    exit()

logfile = sys.argv[1]
csvfile = sys.argv[2]


titles = []
rows = []
with open(logfile, 'r') as f:
    titlesset = False
    for line in f.readlines():
        val_part = line.split('\r')[-1]
        if val_part.find('loss') > -1:
            segments = val_part.strip().split('|')
            row = []
            for segment in segments:
                pts = segment.strip().replace(' = ', ' ').split(' ')
                if not titlesset:
                    titles.append(pts[0])
                row.append(pts[1])
            titlesset = True
            rows.append(row)

with open(csvfile, 'w') as f:
    f.write(';'.join(titles) + '\n')
    for row in rows:
        f.write(';'.join(row) + '\n')
