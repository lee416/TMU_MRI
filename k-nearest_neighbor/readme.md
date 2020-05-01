# **K-nearest neighbors KNN**


## **Reference**
https://www.youtube.com/watch?v=8xABazRelfI  
https://ithelp.ithome.com.tw/articles/10224036  
https://www.youtube.com/watch?v=8xABazRelfI&t=309s  
https://www.youtube.com/watch?v=SaxJrekR2M8  
https://www.youtube.com/watch?v=HVXime0nQeI  
https://www.youtube.com/watch?v=UqYde-LULfs  

recommend video:  
[Youtube KNN Algorithm 0:00~11:26](https://www.youtube.com/watch?v=6kZ-OPLNcgE)


## **KNN**
-------
### **什麼是KNN？**

有一新個數值，周圍被很多其他數值圍繞，離他最近的K個數值決定它的種類就是KNN方法


用圖片來說最快:

![](https://imgur.com/vdjAFuC.png)

來源: https://ithelp.ithome.com.tw/articles/10224036

K = 3 >> 綠  
K = 7 >> 紅 

#### **K值怎麼決定?**

K值可以很容易的直接決定分類的結果，大致上來說K越小越容易符合常見的分類結果，所以通常不會太小，但如果K超大， K值就對分類越模糊，容易分類不明確。

#### 距離怎麼算?

常見方法是用歐幾里得距離，即($\Delta$x,$\Delta$y)相減開根號  

$d(x,y)= \sqrt{(x_1,y_1)^2-(x_2,y_2)^2}$  

![](https://imgur.com/RVwuhe9.png)

#### 為甚麼不KNN?
![](https://imgur.com/IzQWwDf.png)  
來源:https://www.deeplearningitalia.com/manifold-based-tools-isomap-algorithm/  

1. 以這個知名的圖為例，藍點以KNN計算應該應該屬於x1，但依傳統趨勢分析來說藍點應該跟x2趨勢上比較接近。
2. KNN是沒法考慮空間向量的，簡單來說就是，計算量會隨著參數變多，舉例來說：如果參數有100筆，K值又是100，單一個數值的計算量就是100*100個



## **Code Practice**
-------
[Python Practice](http://xperimentallearning.blogspot.com/2017/04/scikit-learn-sklearn-library-machine.html)  
[Datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)  



