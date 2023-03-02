# 이항분류 실습

## 독일 신용 데이터

### 데이터 정보

* GC : German Credit Screening dataset
* 각 고객에 재정 상태에 대한 각종 정보와 함께, 대출 후 대출금 체납여부가 포함되어 있다.
* 1000명의 고객의 재정상태에 관한 데이터를 20가지 항목으로 수집하였다.
* 대출 기간(duration)과 같은 양적(numerical) 정보 7가지와 대출 목적(purpose)과 같은 범주형(categorical)정보 13가지로 구성
* 해당 데이터는 R 패키지 RSADBE에 내장되어 있다.


```{r}
library(RSADBE)
data(GC)
```

<br/>

### 데이터 전처리

* numerical data와 good/bad를 사용하기 위한 전처리

```{r}
GC.num<-GC[,c(2,5,8,11,13,16,18,21)]
```

<br/>

* 전체 데이터의 70%를 train set으로, 나머지를 test set으로 활용
```{r}
set.seed(1)
train<-sample(1:nrow(GC),ceiling(0.7*nrow(GC)),F)
GC.train<-GC.num[train,]
GC.test<-GC.num[-train,-8]
GC.label<-GC.num[-train,8]
head(GC.num)
head(GC.train)
head(GC.test)
```
 
 <br/>
 
### EDA

EDA(Exploratory Data Analysis, 탐색적 데이터 분석)

* 본격적으로 판별분석에 들어가기 앞서, GC 데이터를 정량적, 가시적으로 요약하여 보자.
* 본 실습에서는 20개 정보중 7개의 양적정보만을 사용한다.
* cor 함수는 good_bad를 제외한 7개의 양적 정보간의 상관계수를 계산한 행렬을 반환한다.

```{r}
dim(GC)
dim(GC.num)
```

<br/>

* 대출 체납 여부가 없는 사람들의 비율

```{r}
sum(GC.num$good_bad=="good")/length(GC$good_bad)
names(GC.num)
summary(GC.num)
pairs(GC.num)
cor(GC.num[,-8])
```

<br/>

### 판별 분석

#### **KNN(K-Nearest Neighbor)**

* k=1,3 일때의 결과가 아래와 같다.
```{r}
library(class)
set.seed(2)
knn.pred<-knn(train=GC.train[,-8],test=GC.test,cl=GC.train$good_bad,k=1)
table(GC.label, knn.pred)

knn.pred.3<-knn(train=GC.train[,-8],test=GC.test,cl=GC.train$good_bad,k=3)
table(GC.label, knn.pred.3)
```

<br/>

#### **LDA**

```{r}
library(MASS)
GC.fit.LDA<-lda(good_bad ~ ., data=GC.num, subset=train)
GC.fit.LDA$prior
GC.fit.LDA$scaling
```



* LDA의 prior : training set에서 $\hat{\pi}_{good} =0.702, \hat{\pi}_{bad} = 0.297$임을 의미한다.
* LDA의 scaling : -0.054 X duration + $\cdots$ + -0.239 X depends 이 클수록 good, 작을수록 bad로 분류한다.

<br/>

* 이를 plot을 이용해 그리면 training set에서의 -0.054 X duration + $\cdots$ + -0.239 X depends 의 값의 히스토그램이 얻어진다.


```{r}
plot(GC.fit.LDA)
lda.pred<-predict(GC.fit.LDA,GC.test)
table(GC.label,lda.pred$class)
```


<br/>

#### **QDA**

* QDA는 판별함수, 즉 $\delta_j(x)$가 1차함수가 아니므로, LDA에서의 값이나 plot은 차용할 수 없다.

```{r}
GC.fit.QDA <- qda(good_bad ~ ., data=GC.num, subset=train)
qda.pred<-predict(GC.fit.QDA, GC.test)
table(GC.label, qda.pred$class)
```

<br/>

#### **로지스틱 회귀모형**

* 전진선택으로 모형선택 후 예측

```{r}
GC.fit.logit<-glm(good_bad ~ ., data=GC.train, family=binomial)
GC.fit.logit.null<-glm(good_bad ~ 1, data=GC.train, family=binomial)
GC.fit.logit.FS=step(GC.fit.logit.null, scope=list(lower=GC.fit.logit.null, upper=GC.fit.logit), direction="forward")
GC.fit.logit.FS.pred<-predict(GC.fit.logit.FS, newdata=GC.test, type="response")
logit.pred=as.factor(ifelse(GC.fit.logit.FS.pred >0.7,"good","bad")) # 1:good
table(GC.label, logit.pred)
```

```{r}
mean(knn.pred != GC.label); mean(knn.pred.3 != GC.label)
mean(lda.pred$class != GC.label)
mean(qda.pred$class != GC.label)
mean(logit.pred !=GC.label)
```

<br/>

### 모형 평가

#### **모형 평가 - cost 고려**

* 전체 데이터에서 "good" 상태가 70%이므로 test error rate이 0.3 남짓이라면 좋은 모형이라고 할 수 없다.
* 다만 test error rate는 모든 오분류에 같은 가중치를 주는 모형 평가방법이고, 이는 의료, 금융, 공정등에서는 알맞지 않은 면이 있다.
* 데이터 제공 기관에서 제안하는 cost 함수, $c(good | bad)=5, c(bad|good)=1$을 사용하여 평가할 수 있다.

```{r}
cost <- matrix(c(0,1,5,0), nrow = 2, ncol = 2)
sum(table( GC.label,knn.pred.3) * cost) / (max(cost) * length(GC.label))
sum(table( GC.label,lda.pred$class) * cost) / (max(cost) * length(GC.label))
sum(table( GC.label,qda.pred$class) * cost) / (max(cost) * length(GC.label))
sum(table( GC.label,logit.pred) * cost) / (max(cost) * length(GC.label))
```

<br/>

#### **특이도, 민감도**

```{r}
knn.cm=table(GC.label,knn.pred.3); lda.cm=table(GC.label, lda.pred$class)
qda.cm=table(GC.label, qda.pred$class); logit.cm=table(GC.label,logit.pred)
```

<br/>

* 민감도, TPR %1:good
```{r}
knn.cm[2,2]/sum(knn.cm[2,]);lda.cm[2,2]/sum(lda.cm[2,]);qda.cm[2,2]/sum(qda.cm[2,]);logit.cm[2,2]/sum(qda.cm[2,]) 
```

<br/>

* 특이도, 1-FPR
```{r}
knn.cm[1,1]/sum(knn.cm[1,]);lda.cm[1,1]/sum(lda.cm[1,]);qda.cm[1,1]/sum(qda.cm[1,]);logit.cm[1,1]/sum(logit.cm[1,])
```

<br/>

* caret package를 이용하여 구할 수도 있다.

```{r}
library(caret)
specificity(qda.pred$class, factor(GC.label), "bad")
sensitivity(qda.pred$class, factor(GC.label), "good")
posPredValue(qda.pred$class, factor(GC.label), "good")
negPredValue(qda.pred$class, factor(GC.label), "bad")
```

<br/>

#### **ROC curve**

```{r}
library(PRROC)
PRROC_obj1 <- roc.curve(scores.class0 = lda.pred$posterior[,1], 
                        weights.class0=1*(GC.label=="bad"),curve=TRUE)
plot(PRROC_obj1)
PRROC_obj2 <- roc.curve(scores.class0 = qda.pred$posterior[,1], 
                        weights.class0=1*(GC.label=="bad"),curve=TRUE)
plot(PRROC_obj2)

PRROC_obj3 <- roc.curve(scores.class0 = 1-GC.fit.logit.FS.pred, 
                        weights.class0=1*(GC.label=="bad"),curve=TRUE)
plot(PRROC_obj3)
```

<br/>

## 반도체 공정 데이터

### 데이터 정보

* UCI Machine Learning Repository 사이트의 반도체 제조공정으로부터 나온 데이터(https://archive.ics.uci.edu/ml/datasets/SECOM)
* secom.data : 1567개의 example X 591개의 feature
* secom_labels.data : pass(+1)/fail(-1), time step

```{r}
secom00<-read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data")
secom11<-read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data")
secom11$V1[secom11$V1 ==-1] <- 0
secom <- cbind(secom11$V1, secom00)
names(secom)[1] <-"pass"
```


<br/>

### 전처리

* 모든 example에 동일한 값을 가지고 있는 feature들과, NaN이 20%이상인 feature들을 제거한 후, 남은 feature들 중에 NaN은 해당 feature의 중위수로 대처하는 전처리 진행

```{r}
## Find columns that do not have useful info
secom.red <- apply(secom, 2, function(x) max(na.omit(x)) == min(na.omit(x)))
## Find columns with too much NaN values, 20%
secom.nan.col <- apply(secom, 2, function(x){sum(is.na(x))/length(x) >=0.2})
## Remove redundant or NaN (more than 20%) columns
secom2 <- secom[,!(secom.red | secom.nan.col)]
## Imputation of missing values by the median of the corresponding variable
md=apply(secom2,2,function(x){median(x, na.rm=TRUE)})
for(i in 2:ncol(secom2)){
  secom2[is.na(secom2[,i]),i] <- md[i]
}
```

<br/> 

### 직접 해 보기

1. Test set(30%)과 Training set(70%)으로 나누고

2. KNN, LDA, QDA, 로지스틱 모형을 Training set을 이용하여 적합시킨다.

3. 각각 test error rate(오분류율), specificity(특이도), sensitivity(민감도)를 구하여라.

4. ROC 곡선과 AUC를 구하여라.

<br/>

# 불균형자료분석 실습

## 독일 신용 데이터
* 이항분류 실습에서 다룬 GC.num에서 good_bad값을 기준으로 Over Sampling(Up Sampling)과 Under Sampling(Down Sampling)을 실시해 본다.

```{r}
library(caret)
```

<br/>

* Down-sampling
```{r}
set.seed(1234)
GC.train.ds<-as.data.frame(downSample(x=subset(GC.train, select=-good_bad), 
                        y=GC.train$good_bad, yname="good_bad"))
```

<br/>

* Up-sampling
```{r}
set.seed(12345)
GC.train.us<-as.data.frame(upSample(x=subset(GC.train, select=-good_bad), 
                      y=GC.train$good_bad, yname="good_bad"))

```

<br/>

### Downsampled-dataset

#### **로지스틱모형 추정 및 예측**


* **Fitting logistic model on training set**

```{r}
GC.fit.ds<-glm(good_bad~. ,data=GC.train.ds, family=binomial)
```

<br/>

* **Prediction using test set**


```{r}
GC.ds.pred.pr <- predict(GC.fit.ds, newdata=GC.test, type="response")
GC.ds.pred<-as.factor(ifelse(GC.ds.pred.pr  > 0.5,"good","bad"))

GC.ds.t<-table(GC.label, GC.ds.pred)
GC.ds.t
```

<br/>

* **오분류율**

```{r}
mean(GC.label!=GC.ds.pred) 
```

<br/>

* **민감도 & 특이도**

```{r}
GC.ds.t[2,2]/sum(GC.ds.t[2,]);GC.ds.t[1,1]/sum(GC.ds.t[1,]) 
```

<br/>

* **AUC 값 및 ROC 곡선**

```{r}
roc.ds <- roc.curve(scores.class0=1-GC.ds.pred.pr,
                    weights.class0=1*(GC.label=="bad"), curve=TRUE)
plot(roc.ds)
```

<br/>

### UpSampled-dataset

#### **로지스틱모형 추정 및 예측**

* **Fitting logistic model on training set**

```{r}
GC.fit.us<-glm(good_bad~. ,data=GC.train.us, family=binomial)
```

<br/>

* **Prediction using test set**

```{r}
GC.us.pred.pr <- predict(GC.fit.us, newdata=GC.test, type="response")
GC.us.pred<-as.factor(ifelse(GC.us.pred.pr  > 0.5,"good","bad"))

GC.us.t<-table(GC.label, GC.us.pred)
GC.us.t
```

<br/>

* **오분류율**
```{r}

mean(GC.label!=GC.us.pred)
```

<br/>

* **민감도 & 특이도**

```{r}
GC.us.t[2,2]/sum(GC.us.t[2,]);GC.us.t[1,1]/sum(GC.us.t[1,])
```

<br/>

* **AUC 값 및 ROC 곡선**

```{r}
roc.us <- roc.curve(scores.class0=1-GC.us.pred.pr,
                    weights.class0=1*(GC.label=="bad"), curve=TRUE)
plot(roc.us)
```

<br/>


## 반도체 공정 데이터

### 직접 해 보기

1. Test set과 Training set을 나누어라.

2. Upsample과 Downsample을 실시하여라.

3. Upsampling과 Downsampling 각 case에 대해, KNN, LDA, QDA, 로지스틱모형을 적용하고, error rate(오분류율), 특이도, 민감도를 구하여 비교해 보자.

