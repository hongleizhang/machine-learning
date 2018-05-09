#encoding:utf-8

'''
@author:zhanghonglei
@email:hongleizhang@bjtu.edu.cn
@date:2017-06-18
'''

#encoding:utf-8
%matplotlib inline

import matplotlib.pylab as plt
class Perceptron(object):
	"""docstring for Perceptron"""
	def __init__(self, X,y):
		super(Perceptron, self).__init__()
		self.X = X
		self.y=y

	def plot_initial(self):
		mk=[]
		cs=[]
		for l in y:
			if l>0:
				mk.append('o')
				cs.append('red')
			else:
				mk.append('x')
				cs.append('blue')
		# * 用来传递任意个无名字参数，这些参数会以一个Tuple的形式访问。
		#**用来处理传递任意个有名字的参数，这些参数用dict来访问。
		x1,x2=zip(*X)
		#print x1,x2
		for _mk,_cs,_x1,_x2 in zip(mk,cs,x1,x2):
			plt.scatter(_x1,_x2,marker=_mk,c=_cs,s=100)
		plt.text(6.8,6.8,r'+1')
		plt.text(0.2,0.2,r'-1')
		plt.axis([0,8,0,8])
		plt.plot([0,8],[3,4],'k-',linewidth=2.0,color='g')
		plt.show()

	def train(self,w1=0,w2=0,b=0):
		
		x1,x2=zip(*self.X)
		for _ in range(5):
			mk=[]
			cs=[]
			for i in range(10):
				print(i)
				s=w1*x1[i]+w2*x2[i]+b
                		#wi += lr*(y-\hat(y))*xi
                		#如果本来是1，预测成了0，那么说明w与x的夹角大于90度，则需将两个向量相加
                		#如果本来是0，预测成了1，那么说明w与x的夹角小于90度，以至于是个大于0的数给判成了1，则需将两个向量夹角变大，因此向量相减即可
				if (s>=0 and y[i]<=0) or (s<=0 and y[i]>0):  
					w1+=y[i]*x1[i]
					w2+=y[i]*x2[i]
					b+=y[i]
			for i in range(10):
				s=w1*x1[i]+w2*x2[i]+b
				if s>0:
					cs.append('red')
					mk.append('o')
				else:
					cs.append('blue')
					mk.append('x')
			for _mk,_cs,_x1,_x2 in zip(mk,cs,x1,x2):
				plt.scatter(_x1,_x2,marker=_mk,c=_cs,s=100)
			s1=(0,-b/w2)
			s2=(8,(-b-8*w1)/w2)
			plt.plot([s1[0],s2[0]],[s1[1],s2[1]],'k-',linewidth=2.0,color='g')
			print('w1:%s,w2:%s,b:%s'%(w1,w2,b))
			plt.show()


if __name__ == '__main__':
	
	X=[(1,1),(0.9,1.8),(2.3,1),(2.8,1.5),(1.3,4),(6,4.6),(4.2,5.1),(5.1,6.6),(6.2,5.6),(4.6,5.2)]
	y=[-1,-1,-1,-1,1,1,1,1,1,1]
	perceptron = Perceptron(X,y)
	perceptron.plot_initial()
	perceptron.train()
