# -*-coding:utf-8-*- 
import os
def ab(path):#遍历指定文件夹中所有文件，检查图像大小，长高小于300的删除,不是图像的文件也删除
    for root,dirs,files in os.walk(path):
        for name in files:
            print(os.path.join(root,name))
            aa1=os.path.join(root,name)
            if aa1.split('.')[-1]=='pyc':
                print('del')
                os.remove(aa1)#删除文件
            
def main():
    path="./"
    ab(path)
if __name__=="__main__":
    main()