# -
天池大赛新人赛挖掘幸福感  
参考了天池论坛中Mr_yang大佬的思路  
6个主成分  
两层  
6*16*1  
Relu  
1200epoch，每批量20  
损失函数mean_squared_error  
优化函数 Adam  
最低loss 1.1  
波动特别大，感觉不是很靠谱。  
![image](images/1.jpg)
下面增加epoch大小试试
6个主成分  
两层  
6*16*1  
Relu  
1600epoch，每批量40  
损失函数mean_squared_error  
优化函数 Adam  
最低loss 10.73  
发现还是不行
![image](images/2.jpg)
