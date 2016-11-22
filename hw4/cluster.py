check_chacter = ['!', ':', '(', ')', '[', ']', '{', '}'
                    , '=', '-', '>', '<', '*', '@', ';', '/', '|']

train_data = open('docs.txt', "r")
for line in train_data.readlines():
    word = line.split('\n')[0].split(' ')[0]
    if any(i in word for i in check_chacter) or word == '':
        continue
    print word
