import re


s = "(44, 150)"

# a = s.split(",")
a =  re.split(r"\(|,|\)", s)
b = [int(i.strip()) for i in a if i != ""]
print b
line2 = "(word;Word,emp? hahaha; whole, cai"
print re.split(r";|,|\(|\?\s|;\s|,\s", line2) 

for i in xrange(1, 10):
    print i
