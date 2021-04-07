import pickle

f = open('tuyen1.pkl','rb')
tuyen2 = pickle.load(f)
f.close()

print(tuyen2) # ได้ ตู้เย็นยี่ห้อ hitachi ใส่ผักดองกับนมสดกับเนื้อบด