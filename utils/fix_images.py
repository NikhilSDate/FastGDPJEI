from PIL import Image
from math import log10, floor
for i in range(66):
    file_name = ''
    if(i==0):
        leading_zeroes = 2
    else:
        leading_zeroes = 2 - floor(log10(i))
    for j in range(leading_zeroes):
        print(i,leading_zeroes)
        file_name=file_name+'0'
    file_name=file_name+str(i)+'.png'
    img=Image.open('../aaai/'+file_name)
    img.save('../aaai/'+file_name)



