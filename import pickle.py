from tuyen import Tuyen
import pickle

tuyen1 = Tuyen('hitachi',['ผักดอง','นมสด','เนื้อบด'])

f = open('tuyen1.pkl','wb')
pickle.dump(tuyen1,f)
f.close()