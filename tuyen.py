#tuyen.py
class Tuyen:
    def __init__(self,yiho,khong):
        self.yiho = yiho
        self.khong = khong
    
    def __str__(self):
     return 'ตู้เย็นยี่ห้อ %s ใส่%s'%(self.yiho,'กับ'(self.khong[0]))
