import json
texts=dict()
for i in range(66):
    if i<10:
        filename='aaai/00'+str(i)+'.json'
    else:
        filename='aaai/0'+str(i)+'.json'
    with open(filename) as infile:
        json_text=json.loads(infile.read())

        texts[i]={'text':json_text['text']}
outfile=open('../aaai/problem_texts.json', 'w')
json_out=json.dump(texts,outfile,indent=4)
