import iam


iamdb = iam.IamDataset(True, 300, 30)

iamdb._loaddata()

for x, y in iamdb.generateBatch(1):
    print "len", len(y)
    print "len", x
