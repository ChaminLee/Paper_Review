# < Fast R-CNN >

## *"R-CNN ++"*

## ==0. Abstract==
Fast R-CNN은 이전의 과정들에 비해서 training / test 속도를 증진시켰고, detection accuracy도 증가시켰다. R-CNN보다 훈련 속도가 9배 빠르고, test는 213배 빠르고, mAP또한 높다.

## ==1. Introduction==
image classification에 비해 object detection은 복잡한 방법을 요구한다는 데에서 더 도전적인 일이다. 그러한 복잡성 때문에, 많은 현재 접근 방법들이 multi-stage pipelines로 모델을 학습하기에 느리고, 세련되지 않았다.
복잡성은 detection이 객체에 대한 정확한 localization을 요구하는데서 발생하는데, 이는 두 가지 문제를 발생시킨다. 

* 수많은 Proposals 후보들이 생성되어야 한다.
* 이 후보들은 대략적인 localization만을 제공하기에, 정확한 localization을 위해서는 정제되어야한다. 

이 논문에서는 single-stage 학습 알고리즘을 통해 object proposals를 분류하는 것과 공간적인 locations를 정제하는것을 동시에 훈련시킨다. 

### ==1.1. R-CNN and SPPnet==
R-CNN은 다음과 같은 단점이 있었다. 

1. 학습이 multi-stage pipeline이다
R-CNN은 먼저 object proposals에 대한 ConvNet을 log loss를 이용하여 fine-tune한다. 그리고 ConvNet features에 대해 SVM을 학습한다. 그리고 학습의 마지막단계에서는, bounding-box regressors가 학습된다. 
 
2. 학습은 비용적으로, 공간적으로 비용이 많이 든다. 
SVM과 bounding-box regressor를 학습할 때, features들은 각각의 이미지의 object proposals에서 추출되고, disk에 기록된다. 이 때문에, 굉장히 많은 용량을 필요로한다.
3. object detection이 느리다.
test시에, features는 각 테스트 이미지의 각각의 object proposal에서 추출된다. VGG16을 이용한 detection은 이미지 하나에 대해 47초가 소요된다. (GPU기준)

R-CNN이 느린 이유는, 각 object proposal에 대한 ConvNet forward pass를 실행할때, 연산을 공유하지 않기 때문이다. SPPnet 방법의 경우, 전체 입력 이미지에 대해 convolution feature map을 연산하고, 공유된 feature map에서 추출한 feature vector를 이용하여 각 object proposal을 분류한다. 여러 출력의 크기가 pooling된 후에, spatial pyramid pooling에서와 같이 연결된다.  이러한 SPPnet 방법이 R-CNN보다 train/test시에 더 속도가 빨랐다. 

하지만 SPPnet도 단점이 있다. R-CNN처럼, 학습과정이 multi-stage(특성 추출하기, log loss이용하여 fine-tune, SVM학습, bounding-box regressor 학습)로 구성되어있다는 것이다.(특성들이 디스크에도 저장된다) 하지만 R-CNN과 달리 convolutional layers는 업데이트할 수 없다. 이러한 제약은 아주 깊은 networks의 정확도를 제한한다. 
 
### ==1.2. Contributions==
1. R-CNN,SPPnet보다 높은 detection 성능(mAP)
2. multi-task loss를 이용한, single-stage 학습
3. 학습이 모든 network layers를 업데이트할 수 있다.
4. feature 캐싱에 요구되는 disk storage가 필요없다. 


## ==2. Fast R-CNN architecture and training==
Fig.1 사진

위의 Fig.1은 Fast R-CNN의 구조를 보여준다. 
1. Fast R-CNN은 먼저 전체 이미지와 object proposals 세트를 입력으로 받는다. 
2. network는 먼저 전체의 이미지를 Conv, max-pooling layers를 이용하여 conv feature map을 생성한다.
3. 각각의 object proposals에 대해, RoI(region of interest) pooling layer는  feature map을 부터 고정된 길이의 feature vector를 추출한다. 
4. 각각의 feature vector는 최종적으로 두 개의 output layer로 분기되는 fully connected layers에 입력된다. 하나는 K object 클래스 + background 클래스에 대한 softmax 확률 추정치를 생성하는 것이다. 다른 하나는 K object 클래스에 대한 4가지 값을 출력해낸다. 이 4개의 값은 bounding-box 의 좌표를 표시한다. 
### ==2.1. The RoI pooling layer==
RoI pooling layer는 max-pooling을 사용하여 유효한 RoI 내부의 features를 $H \times W$의 고정된 범위로 작은 feature map으로 변환한다. (7X7), H와W는 특정 RoI와 독립적인 하이퍼파라미터이다.

각 RoI는 $(r,c,h,w)$의 좌표로 표현되는데, $(r,c)$는 좌측 상단이고, $(h,w)$는 각각 높이과 넓이를 의미한다. 

RoI max pooling은 feature map 위에서 $\frac{h}{H} \times \frac{w}{W}$크기 만큼 grid를 만들어 max-pooling을 하면 논문에서 설정했던 $H \times W$형태의 feature size로 변환된다. pooling은 각 feature map 채널에 대해 독립적으로 실행되었다. 
### ==2.2. Initializing from pre-trained networks==
pre-trained network가 Fast R-CNN Network를 초기화할 때, 세 가지 변환을 겪는다. 

1. 마지막 max-pooling layer가 RoI pooling layer로 대체된다.
2. networks의 마지막 fully connected layer와 softmax가 앞서 말했던 두 가지 output layer로 대체된다.(softmax + bounding-box regressor)
3. 두 가지의 데이터 입력을 받기 위해 network는 수정된다. (이미지들의 리스트 , 해당 이미지의 RoI 리스트)
 
### ==2.3. Fine-tuning for detection==

모든 network 가중치를 back-propagation으로 학습하는 것은 Fast R-CNN의 중요한 능력이다. 먼저, SPPnet가 spatial pyramid pooling layer 아래의 가중치를 업데이트할 수 없는지 설명하자. 
근본적인 문제는 SPP layer를 통한 back-propagation은 각 training sample로 다른 이미지가 들어온다면 매우 비효율적이라는 것이다. 

Fast R-CNN 학습시에는, SGD mini-batches는 계층적으로 샘플링되는데, $N$개의 이미지를 먼저 샘플링하고, 각 이미지로부터의 $R/N$ RoIs를 샘플링한다. 중요한 것은, 같은 이미지로부터의 RoIs는 forward / backward 계산시에 연산과 메모리가 공유된다. $N$을 작게 만들면 mini-batch 연산이 줄어들게 되는 것이다. 예를 들어서, $N=2$, $R=128$를 사용할 경우, 서로 다른 이미지에서 하나의 RoI를 샘플링하는 것 보다 약 64배 빠르다. 

한 가지 걱정은, 같은 이미지로 부터의 RoIs들이 상호연관되어있기 때문에, 학습 수렴속도를 느리게 할 것이라는 것이다. 하지만 $N=2, R=128$과 R-CNN보다 적은 SGD iteration을 사용하여 좋은 결과를 얻을 것을 보아 이 문제는 걱정하지 않아도 된다. 

##### ==Multi-task loss==
Fast R-CNN network는 근본이 같은(sibling) 두 가지 output layers를 가지고 있다. 첫 번째 결과는, $K+1$카테고리에 대한 $p=(p_0,...,p_K)의$이산확률분포이다. $p$는 fully connected layer의 $K+1$출력에 대해 softmax로 계산된다. (즉, 확률 / object (K) + background (1))
 두 번째 결과는 bounding-box regressor이다. $K$클래스에 대해 index $k$로 $t^k =(t_x^k,t_y^k,t_w^k,t_h^k)$로 나타낸다. 
동시에 분류 및 bounding-box regression을 훈련하기 위해 RoI 레이블이 지정된 각 RoI에 multi-task loss $L$을 사용한다. 

$$L(p,u,t^u,v) = L_{cls}(p,u) + \lambda[u≥1]L_{loc}(t^u,v)$$
ground-truth class = $u$
ground-truth bounding-box regression target = $v$

$L_{cls}(p,u)=-\log p_u$는 true class $u$에 대한 log loss이다. 두 번째 loss인 $L_{loc}$는 클래스 $u$에 대한 true bounding-box regression targets의 튜플인 $v=(v_x,v_y,v_w,v_h)$로 나타나고, 예측된 튜플은 클래스 $u$에 대해 $t^u = (t_x^u,t_y^u,t_w^u,t_h^u)$로 나타난다. 아이버슨 괄호 $[u≥1]$은 $u≥1$일 때 1이고, 다른 경우에는 0으로 나타난다.

아이버슨 괄호 정의 사진

background의 경우 $u=0$이기 때문에, background RoI에 대해서는 ground-truth bounding box와 $L_{loc}$의 개념이 무시된다. 
bounding-box regression에 대해 다음의 loss를 사용한다. 

$$L_{loc}(t^u,v) = \sum_{i \in \{x,y,w,h\}} smooth_{L_1} (t_i^u - v_i)$$

위 식은 다음의 조건을 따른다.

$$smooth_{L_1}(x)=\begin{cases} 
0.5x^2 ...... |x| <1  \\
|x|-0.5    ......otherwise
\end{cases}$$

$L_1$ loss는 $L_2$ loss에 비해 이상치에 덜 민감하다.

맨 위의 식에서 $\lambda$는 두 가지 loss사이에서 균형을 잡아준다. ground-truth regression target인 $v_i$를 평균을 0, 단위 분산으로 만들어준다.(표준화: standardization와 같다.)
##### ==Mini-batch sampling==
Fast R-CNN에서는 $R=128$의 미니배치를 사용하고, 각 이미지로부터 64개의 RoIs를 샘플링한다. Ground-truth와 IoU > 0.5 인 경우 Positive, [0.1,0.5)일 경우 Negative로 구분한다.(너무 낮은 것을 샘플로 추가하지 않는다.)
##### ==Back-propagation through RoI pooling layers==
Forward pass가 모든 이미지들을 독립적으로 처리하기 때문에 $N > 1$로의 확장이 간단하지만, 명확성을 위해 미니 배치당 하나의 이미지($N = 1$)만 가정한다.
$x_i \in \mathbb{R}$을 RoI pooling layer에 입력하는 i 번째 활성화 입력으로 하고, $y_{rj}$가 $r$번째 RoI로부터 나온 layer의 $j$번째 출력이라고 하자. Pooling layer는 $y_{rj} = x_{i*(r,j)}$를 계산하는데, 인덱스는 다음과 같다. $i*(r,j) = argmax_{i' \in \mathcal{R}(r,j)} x_{i'}$
$\mathcal{R}(r,j)$는 출력 단위 $y_{rj}$ max pools이 있는 sub-window에 있는 입력의 인덱스 집합이다. 하나의 $x_i$는 아마 여러 다른 결과인 $y_{rj}$에 할당될 것이다. 
RoI pooling layer의 backward pass는 다음 식을 따르는 각 입력 변수 $x_i$를 고려한 비용 함수의 편미분이다.

$$\frac{\partial{L}}{\partial{x_i}} = \sum_r\sum_j [i=i*(r,j)]\frac{\partial{L}}{\partial{y_{rj}}} $$

말로는, 각 미니배치 RoI $r$ 그리고 각 pooling 결과인 $y_{rj}$, 편미분 $\partial{L}/\partial{x_i}$는 $i$가 max pooling에 의해 $y_{rj}$에 대해  선택된 argmax일 때 축적된다. 
##### ==SGD hyper-parameters==
softmax classification과 bounding-box regression에 사용된 fc layers는 각각 평균이 0이고, 표준편차가 0.01, 0.001인 정규분포(가우시안)로부터 초기화된다. 
### ==2.4. Scale invariance==
* 각각의 이미지는 학습과 테스트시에, 사전에 정의된 픽셀 크기로 전처리된다. Network는 training data로부터 scale-invariant한 object detection을 직접 학습해야한다.

## ==3. Fast R-CNN detection==
Network는 이미지와 R object proposals의 리스트를 입력으로 받는다. $R$은 보통 2,000개 정도이지만, 더 큰 경우도 고려할 것이다. 
### ==3.1. Truncated SVD for faster detection==
커다란 Fc layers는 truncated SVD를 통해 압축하는 것으로 간단하게 가속화할 수 있다. 

truncated SVD 사진

$$W \approx U\Sigma_t V^T$$

$U$는 $u \times t$의 형태로 $t$가 $W$의 left-singular vectors이다. $\Sigma_t$는 $t \times t$형태인 대각행렬로 $t$는 $W$의 singular values이다. $V$는 $v \times t$의 형태로 $t$는 $W$의 right-singular vectors이다. Truncated SVD는 파라미터수를 $uv$에서 $t(u+v)$만큼 줄인다.
## ==4. Main results==
1. State-of-the-art mAP on VOC07, 2010, and 2012
2. R-CNN, SPPnet과 비교했을때, 더 빠른 training과 testing
3. VGG16을 fine-tuning함으로써 올라가는 mAP

### ==4.1. Experimental setup==
CaffeNet을 "S"(small)이라고하고, VGG는 깊이가 S랑 같지만 더 넓다. 이를 M이라고한다. 그리고 마지막 network는 매우 깊은 VGG16 모델이고, L이라고 한다. 
### ==4.2. VOC 2010 and 2012 results==
Table 사진

모든 다른 방법들은 같은 pre-trained VGG16 network로부터 초기화된다. Table.2를 보면 training set이 확대되었을때, Fast R-CNN의 mAP는 68.8%로 SegDeepM보다 뛰어나다.
### ==4.3. VOC 2007 results==
모든 방법들은 같은 pre-trained VGG16 network에서 출발하고 bounding-box regression을 이용한다. 
Table.2를 보면 difficult 샘플을 제외하면 Fast R-CNN mAP는 68.1%로 향상되었다. 다른 시도들은 difficult 샘플을 사용한다. 
### ==4.4. Training and testing time==
Fast R-CNN은 또한 수백 기가바이트의 디스크를 필요로하지 않는다. Features를 캐싱하지 않기 때문이다.

Table.4
##### ==Truncated SVD==
Truncated SVD는 조금의 mAP를 떨어뜨리고, 모델 압축이후 추가적인 fine-tuning 없이 detection 시간을 30%보다 더 줄인다.

Fig.2

SVD가 속도를 증진시킨 것을 보여준다.
### ==4.5. Which layers to fine-tune?==

결과가 매우 깊은 network에서는 유지하지 못할 것이라는 가설을 세웠다.  다음의 시도는 가설을 검증해준다 : RoI pooling layer를 통해 학습하는 것이 매우 깊은 network에 중요하다.

Table.5

모든 conv layers는 fine-tuned되어야 하는가? 아니다! 작은 networks에서 우리는 conv1이 유전적이고 과업에 대해 독립적이라는 것을 알았다. conv1을 학습하던 아니던 mAP에는 영향을 주지못한다. conv3_1, 그리고 위부터 layers를 업데이트할 필요를 알아냈다. conv2_1 이상으로 업데이트를 하면 conv3_1학습에 비해 학습이 1.3배 느려진다. (mAP도 0.3%밖에 증가하지 않음) 논문에 나와있는 Fast R-CNN의 결과는 VGG16 을 conv3_1과 그 위로 fine-tune하는 방식이다. 
## ==5. Design evaluation==

### ==5.1. Does multi-task training help?==
Multi-task training은 파이프라인 관리를 피할 수 있기에 편리하다. 하지만 task가 서로에게 영향을 주기 때문에 결과를 개선할 수 있는 잠재력이 있다. 
Table.6
표를 봤을 때, multi-task를 이용하여 학습했을 때, 분류 정확도를 향상시켜준다는 것을 볼 수 있다. multi-task training + bounding-box regressor를 이용했을 때 가장 성능이 좋았다. 
### ==5.2. Scale invariance: to brute force or finesse?==
Table.7
S,M 모델들이 scale(1,5)에 대해 학습되고 평가된것을 보여준다. 놀랍게도 대부분 single-scale detection의 성능이 multi-scale detection보다 높았다. Single-scale 과정은 특정 매우 깊은 모델에 대해 속도와 정확성과 트레이드오프 관계이기에, 이 하위 섹션 외부의 모든 시도는 s=600 픽셀의 single-scale training과 testing을 한다. 
### ==5.3. Do we need more training data?==
좋은 object detector는 많은 training data가 주어져도 성능이 향상되어야한다. 
### ==5.4. Do SVMs outperform softmax?==
Table.8

위를 보면 softmax가 SVM보다 조금 더 좋은 성능을 보이는 것을 볼 수 있다. 
### ==5.5. Are more proposals always better?==
Figure.3
Fig.3의 파란 실선을 보면 mAP가 증가하다가 proposal 수가 늘어나면서 줄어드는 것을 볼 수 있다. 위 시도를 통해 proposals를 늘리는 것은 도움이 되지 않고 오히려 정확도에 해가된다는 것을 알았다. AR(Average Recall)의 경우는 조심하여 사용해야한다 : 많은 proposals로 인한 높은 AR은 mAP 향상에 영향을 미치지 않기 때문이다. SS가 dense box로 대체되었을 때 1%만이 하락한다. (파란 삼각형)
### ==5.6. Preliminary MS COCO results==
COCO에 대해서도 살짝 실험했다.
## ==6. Conclusion==
특히, sparse한 object proposals가 detector의 성능을 향상시키는 것을 볼 수 있었다. 
