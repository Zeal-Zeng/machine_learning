# encoding='utf-8'
import random
with open('data/test_liner_data.txt','w') as file:
    for i in range(50):
        sum = 9+int(random.random()*200)/100.0
        x1 = int(random.random()*1000)/100.0
        x2 = int((sum - x1)*1000)/1000.0
        file.writelines(str(x1)+','+str(x2)+'\n')