# <YOLO-v1 : You Only Look Once: Unified, Real-Time Object Detection>

## *Unified Pipeline = YOLO!*

## ==0. Abstract==

YOLO는 이전 방식과 다르게, object detection을 공간적으로 분리된 bounding boxes와 클래스 확률로 regression을 간주한다. 단일의 뉴럴 네트워크가 한 번의 평가로 전체 이미지에서 bounding boxes와 클래스 확률을 예측한다. 전체 detection 파이프라인이 단일 네트워크이기 때문에, detection 성능에 대해 직접적으로 end-to-end로 최적화될 수 있다. 

이 통합된 구조는 매우 빠르다. YOLO는 localization 에러를 더 많이 만들지만, 배경에 대한 false positive로는 덜 예측한다. 결국 YOLO는 객체의 일반적인 표현을 학습한다. 이 방식은 다른 detection 방법들 보다 성능이 뛰어나다!

## ==1. Introduction==

사람들은 이미지를 언뜻 봐도 즉시 이미지 안의 물체가 무엇인지, 어디에 있는지, 서로 무슨 관계인지 알 수 있다. 사람의 인식 시스템은 매우 빠르고 정교해서, 운전과 같은 복잡한 일을 조금의 의식적인 생각으로 수행할 수 있다. 

현재의 detection 시스템은 분류기에 detection 역할을 하도록 재설정하는 방식이다. 객체를 탐지하기 위해서, 이러한 시스템은 객체를 분류기가 받고 테스트 이미지 내에서의 다양한 위치와 크기를 평가한다. DPM같은 sliding window를 사용하여 분류기가 전체 이미지에 대해 모든 공간에서 균등하게 실행되도록 한다. 

더 최근의 접근 방식인 R-CNN은 region proposal의 방법을 사용하여 먼저 이미지내에 잠재적인 bounding boxes를 생성하고  그 제안된 boxes에 대해 분류기를 실행하는 방식이다. 분류 이후에, 후처리는 bounding boxes를 정교하게 하고, 중복되는 detections를 제거하고, 다른 객체에 기반해서 boxes를 재점수한다. 이러한 복잡한 파이프라인은 개개인의 요소가 분리되어서 학습되어야 하기 때문에 느리고 최적화하기 어렵다.

논문에서는 object detection을 단일 regression 문제로 재구성하고, 이미지 픽셀에서 bounding box 좌표와 클래스 확률까지 쭉 이어지도록했다. 그래서 객체가 무엇이고 어디있는지를 예측하기위해 이미지를 "단 한 번만 본다"(You Only Look Once)해서 YOLO라고 부른다. 

Fig.1

YOLO의 과정은 간단하고 쭉 나아가는 방식이다. 
(1) 먼저 입력이미지를 448X448로 크기를 조정한다.
(2) 이미지에 대해 단일 Convolutional network를 실행한다.
(3) NMS(Non-Max Suppresion / 중복된 detection 제거), 모델의 신뢰도에 의한 결과 detection의 임계값을 정한다. 

YOLO는 매우 간단하다. Fig.1을 봐보자. 단일의 convolutional network는 동시에 복수의 bounding boxes를 예측하고, boxes에 대한 클래스 확률을 예측한다. YOLO는 전체 이미지에 대해 학습하고 detection 성능을 직접 최적화한다. 이 통합된 모델은 전통적인 object detection 방법에 비해 여러 장점이 있다. 

#### 1 .YOLO는 매우 빠르다. 
Detection을 regression 문제로 재구성하면서 복잡한 파이프라인이 필요하지 않게 되었다. 간단하게 테스트시에, deteciton을 예측하기 위해 새로운 이미지에 NN을 실행하면 된다. Titan X GPU하에서 배치없이 45fps(frame per second), 빠른 버전은 150fps보다 빠르다. 이는 real-time으로 적용할 수 있다는 것을 의미한다. 게다가 YOLO는 다른 real-time 시스템에 비해 두 배가 넘는 mAP를 달성했다. 

#### 2. YOLO는 예측을 할 때, 이미지에 대해 전체적으로 추론한다. 
Sliding window와 region proposal 기반의 기술과 다르게, YOLO는 학습과 테스트시에 전체 이미지를 보기 때문에, 명백하게 그들의 외관과 같은 클래스에 대한 문맥상의 정보를 encodes한다. Fast R-CNN같은 상위의 탐지 방법은, 큰 문맥을 보지 못하기 때문에 객체에 대한 이미지를 배경이라고 말하는 오류를 범한다, YOLO는 Fast R-CNN과 비교해서, 절반 이하의 배경 오류를 발생시킨다.

#### 3. YOLO는 객체의 일반적인 표현을 학습한다.
자연의 이미지와 예술 이미지에 학습했을 때, YOLO는 DPM과 R-CNN과 같은 탑급 탐지 방법보다 큰 차이로 성능을 능가한다. YOLO는 매우 일반적이기 때문에 새로운 도메인이나 예기치 못한 입력에 대해서도 실패할 가능성이 적다. 

YOLO는 여전히 SOTA 탐지 시스템의 정확도에는 못미친다. 그리고 이미지내의 물체를 빠르게 식별할 수 있지만, 작은 물체를 정밀하게 localize하기 위해 노력한다. 

## ==2. Unified Detection==

YOLO는 전체 이미지로부터의 특성으로 각 bounding box를 예측한다. 이는 또한 모든 bounding boxes를 모든 클래스에 대해 동시에 예측한다. 이는 YOLO가 전체 이미지와 이미지내의 모든 객체를 전체적으로 잘 추론하고 있다는 것을 말한다. YOLO는 높은 AP(average precision)을 유지하며 end-to-end로 학습되고 real-time 속도를 가능하게 한다.

YOLO는 입력 이미지는 $S \times S$ grid(격자)로 나눈다. 만약에 객체의 중심이 grid cell에 들어가면, 그 grid cell은 그 객체를 탐지하는 역할을 한다. 

각 grid cell은 bounding boxes $B$와 boxes에 대한 신뢰 점수를 예측한다. 이 신뢰 점수는 box가 객체를 보유하고 있다고 생각하는 모델의 신뢰도와 예측하는 box의 정확도를 반영한다. 공식으로는 신뢰도를 $P_r(Object) *IOU_{pred}^{truth}$로 표현한다. 만약에 cell에 객체가 없을 때, 신뢰 점수는 0이 된다. 그렇지 않으면 신뢰 점수가 예측된 box와 ground truth(실제)사이의 IOU와 같기를 바란다. 

각 bounding box는 5개의 예측으로 구성된다 : $x,y,w,h$ 그리고 신뢰도이다. $(x,y)$는 grid cell의 경계를 기준으로 box의 중심좌표를 나타낸다. 너비와 높이는 전체 이미지에 대해 예측된다. 마지막으로, 신뢰도 예측은 예측된 box와 어느 ground truth box 사이의 IOU를 나타낸다. 

각 grid cell은 조건부 클래스 확률인 $C$를 $P_r(Class_i|Object)$로 예측한다. 이 확률은 객체를 보유하고 있는 grid cell에 대한 조건부이다. boxes의 갯수인 $B$에 상관하지 않고, 오직 grid cell당 하나의 클래스 확률을 예측한다.

테스트시에는 조건부 확률과 개개인의 신뢰도 예측을 곱한다. 

$P_r(Class_i|Object) * P_r(Object) * IOU_{pred}^{truth} = P_r(Class_i)*IOU_{pred}^{truth}$

이는 각 box에 대해 클래스별 신뢰도를 알려준다. 이 점수는 box에서 클래스가 나타나는 확률과 예측된 box가 객체와 얼마나 잘 맞는지를 담고있다.

Fig.2

Pascal VOC에 대해서는 $S=7, B=2$을 사용하고 클래스가 20개 이기에 $C=20$이어서 최종 예측은 $7\times 7 \times 30$ tensor가 된다. 

### ==2.1 Network Design==

Fully connected layers가 출력의 확률과 좌표를 예측할 때, 네트워크의 초반 convolutional layers는 이미지로부터 특성을 추출한다. YOLO의 네트워크 구조는 이미지 분류를 위한 GoogLeNet모델로 부터 영감을 얻었다. YOLO 네트워크는 24개의 conv layers와 2개의 fc layers로 이루어져있다. GoogLeNet에서 인셉션 모듈을 썼던 것 대신에, YOLO는 간단하게 1X1 reduction layers를 사용하고, 그 뒤에 3X3 conv layers를 사용한다. Fig.3에 네트워크 구조를 나타냈다. 

Fast YOLO는 더 적은 conv layers ( 24개 대신 9개 )와 layers에서 더 적은 필터를 사용한다. 네트워크 크기를 제외하고, YOLO와 Fast YOLO사이의 학습, 테스트 파라미터는 동일하다.

Fig.3

### ==2.2 Training==

Conv layers는 ImageNet 1000 클래스에 대해 pretrain 한다. Pretraining을 위해, 처음 20개의 conv layers를 사용하고, 그 뒤에 average-pooling 과 fc layers를 사용한다. 

그리고 모델을 deteciton 역할을 하도록 변환한다. 이후에 4개의 conv layers와 2개의 fc layers를 추가한다. Detection에는 종종 세밀한 시각 정보가 필요하기에 네트워크의 입력 해상도를 224X224에서 448X448로 높인다. 

마지막 layer는 클래스 확률과 bounding box 좌표를 예측한다. 이미지의 너비와 폭을 기준으로 bounding box 너비와 높이를 0과 1사이로 오도록 정규화한다. 그리고 bounding box의 좌표 $(x,y)$는 특정 grid cell 위치의 offsets 값을 사용하여 0과 1 사이에 오게 한다. 
마지막 layer에 선형 활성화 함수를 사용하고, 다른 모든 layers는 살짝 조정된 선형 활성화 함수를 사용한다. 

$$\phi(x) = \begin{cases} x,     (x > 0) \\ 0.1x, 			  (otherwise)
\end{cases}$$

모델의 출력의 SSE(sum squared error)를 최적화한다. SSE가 최적화하기 쉽기 때문에 이를 사용하지만, 우리의 목표인 AP를 최대화 하는데에는 완벽하게 맞진 않다. 이는 이상적이지 않은 분류 오류와 localization 오류에 동등하게 가중치를 부여한다. 또한, 모든 이미지의 많은 grid cells는 어떤 객체도 포함하기 있지 않다. 이는 그러한 cells의 "신뢰" 점수를 0으로 향하게 하고, 종종 객체를 포함하는 cells의 gradient를 못쓰게 만든다. 이는 모델의 불안정성을 유발하여, 학습이 초기에 분산하게 한다. 

이를 해결하려면, bounding box 좌표 예측의 loss를 증가시키고, 객체를 포함하지 않는 boxes에 대한 예측 신뢰도로 부터의 loss를 줄여야한다. 그래서 $\lambda_{coord}, \lambda_{noobj}$를 사용한다. ($\lambda_{coord}=5 / \lambda_{noobj} = 0.5$)

SSE는 또한 큰 박스거나 작은 박스의 오류에 대해 동등하게 가중치를 준다. 우리의 error metric은 큰 박스에서의 작은 편차가 작은 박스에서 작은 편차보다 덜 중요하다는 것을 반영해야한다. (모래사장에서 바늘 < 손바닥안 바늘)
이를 부분적으로 다루기 위해 기존의 너비와 높이를 직접쓰는 대신에 bounding box의 너비와 높이의 제곱근을 예측한다. 

YOLO는 grid cell당 여러 bounding boxes를 예측한다. 학습 때에, 우리는 각 객체에 대해 bounding box predictor가 책임지기를 원한다. 하나의 predictor에 객체를 예측하는 것을 책임감 있게 한다. (예측이 ground truth와 높은 IOU를 가지는 것에 기반하여) 이는 bounding box predictors간에 전문화로 이어진다. 각 predictor는 특정 크기, 종횡비, 또는 객체의 클래스를 잘 예측하여 전체적인 recall(재현율)을 개선시킨다. 

학습시에는 다음의 multi-part 비용 함수를 최적화 한다. 

비용함수 사진 

$\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^B \mathit{1}_{ij}^{obj}[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]$

Object가 존재하는 grid cell $i$의 predictor bounding box $j$에 대해 x,y의 loss를 계산한다.

$+\lambda_{coord}\sum_{i=0}^{S^2}\sum_{j=0}^B \mathit{1}_{ij}^{obj}[(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]$

Object가 존재하는 grid cell $i$의 predictor bounding box $j$에 대해 w,h의 loss를 계산한다.(앞서 말한 제곱근 부분)

$+\sum_{i=0}^{S^2}\sum_{j=0}^B \mathit{1}_{ij}^{obj}(C_i-\hat{C_i})^2$

Object가 존재하는 grid cell $i$의 predictor bounding box $j$에 대해, confidence score의 loss를 계산한다. ($C_i = 1$)

$+ \lambda_{noobj}\sum_{i=0}^{S^2}\sum_{j=0}^B \mathit{1}_{ij}^{noobj}(C_i-\hat{C_i})^2$

Object가 존재하지 않는 grid cell $i$의 predictor bounding box $j$에 대해, confidence score의 loss를 계산한다. ($C_i = 0$)

$+\sum_{i=0}^{S^2}\mathit{1}_{ij}^{obj}\sum_{c \in classes}(p_i(c) - \hat{p_i}(c))^2$

Object가 존재하는 grid cell $i$에 대해 conditional class probability의 loss 계산 ( correct class $c$ : $p_i(c)=1$ / otherwise : $p_i(c)=0$)

$\mathit{1}_i^{obj}$는 object가 존재하는 grid cell $i$를 말한다. 
$\mathit{1}_{ij}^{obj}$는 object가 존재하는 grid cell $i$의 bounding box predictor $j$를 의미한다. 
$\mathit{1}_{ij}^{noobj}$는 object가 존재하지 않는 grid cell $i$의 bounding box $j$를 말한다. 

Loss function이 객체가 grid cell 안에 있을 때, 오직 분류 에러에만 패널티를 준다는 것을 알아야한다. (앞서말한 조건부 클래스 확률) 또한 predictor가 ground truth box에 대해 책임이 있을 때,  bounding box error에 패널티를 준다. (grid cell에서 predictor의 IOU가 가장 높을 때)

135 epochs 동안 학습 데이터셋과 벨리데이션 데이터셋(VOC 2007 ,2012)에 대해 네트워크를 학습시킨다. 학습 동안에 batch = 64 / momentum = 0.9 / decay = 0.0005를 사용한다.
learning rate는 0.001에서 0.01로 에포크 마다 천천히 상승시킨다. 시작부터 높은 learning rate를 사용할 경우 모델이 불안정한 gradients 때문에 발산하기 때문이다. 그래서 75 epochs : 0.01 / 30 epochs : 0.001 / 30 epochs : 0.0001로 학습한다. 

과적합을 피하기 위해서는 dropout이나 data augmentation을 사용해야 한다. dropout rate는 0.5를 사용했다. 이는 첫 connected layer와 layer사이의 동기화를 막아준다. data augmentation으로는 임의적으로 스케일링하거나 원본 이미지 크기의 최대 20%정도에서 사용하였다. 그리고 임의적으로 이미지의 exposure 과 saturation을 조정하였다.(HSV 색 공간의 1.5배율) 

### ==2.3 Inference==

학습 때와 마찬가지로, 테스트 이미지에 대한 예측은 하나의 네트워크 평가만을 요한다. PASCAL VOC에 대해 네트워크는 이미지당 98개의 bounding boxes와 각 박스당 클래스 확률을 예측한다. YOLO는 분류 기반의 방법과는 다르게, 단일 네트워크로 평가하기 때문에 테스트 시에 매우 빠르다. 

grid 디자인은 bounding box predictions 내의 공간적 다양성을 구현한다. 어떤 물체가 어떤 grid cell에 속하는지 명확하고 네트워크는 각 물체에 대해 하나의 box만을 예측한다. 하지만, 커다란 객체나 여러 cells의 경계에 가까운 객체는 여러 cells에 의해 잘 localize 될 수 있다. NMS는 이러한 여러 detection을 수정할 수 있다. R-CNN이나 DPM에서처럼 결정적으로 작용하진 않지만, NMS는 mAP를 2~3% 향상시킨다. 

### ==2.4 Limitation of YOLO==

#### 1. grid cell이 하나의 클래스만 예측하므로 작은 object가 주변에 있으면 제대로 예측하기 힘들다.
YOLO는 각 grid cell이 두 개의  bounding boxes만 예측하고 하나의 클래스만 가질 수 있기 때문에 bounding box prediction에 강한 공간적 제약을 부과한다. 이 공간적 제약은 모델이 예측할 수 있는 근처 물체의 수를 제한한다. 모델은 그룹내에서 나타나는 작은 물체(새떼 무리)에 대해 허덕인다.
#### 2. 학습 데이터로부터 bounding box의 형태를 학습하므로, 새로운형태의 bounding box의 경우 예측하기 어렵다. 
모델은 데이터로부터 bounding boxes를 예측하는 것을 학습하기 때문에, 새로운 또는 비주류의 종횡비나 외형의 객체를 일반화하는데에 허덕인다. 또한 모델은 구조가 입력 이미지에서 여러개의 downsampling layers를 가지고 있기 때문에 bounding box를 예측하는데 비교적 좋지않은 특징을 사용한다. 
##### 3. Localization에 대한 부정확한 경우
비용 함수를 통해 학습할 때, 비용 함수는 작은 bounding boxes와 큰 bounding boxes에서의 오류를 동일하게 처리한다. 큰 box의 작은 오류는 일반적으로 약하지만, 작은 box의 작은 오류는 IOU에 큰 영향을 준다. 주된 오류의 원인은 부정확한 localization이다.

## ==3. Comparison to Other Detection Systems==

Object dection은 컴퓨터 비전에서 핵심 문제이다. Detection 파이프라인은 일반적으로 입력 이미지들로 부터 풍부한 특성 셋을 추출하는 데서 시작한다.(Haar, SIFT,HOG,convolutional features) 그리고 분류기나 localizers는 특정 공간에서 객체를 인식하는데 사용된다. 이 분류기나 localizers는 전체 이미지나 이미지의 regions 하위 세트에 대해 sliding wondow를 실행한다. 우리는 YOLO를 다른 탑급 detection 프레임워크를 유사한 것과 차이점에 대해 비교할 것 이다.

#### ==Deformable parts models==

DPM은 sliding window 접근 방식을 사용하여 object detection을 한다. DPM은 분리된 파이프라인을 사용해서 정적인 특성을 추출하고, regions를 분류하고, 높은 점수를 가진 regions에 대해 bounding boxes를 예측한다. 하지만 YOLO는 이 종류가 다른 부분들을 단일의 CNN으로 대체한다. 이 네트워크는 특성 추출, bounding box prediction, NMS, 전체 문맥상 추론의 기능도 한다. 정적인 특성을 대신에, 네트워크는 특성을 in-line으로 학습하고 detection에 최적화한다. YOLO의 통합된 구조는 DPM보다 빠르고, 더 정확하도록 만든다.  

#### ==R-CNN==

R-CNN과 그 변형들은 이미지내의 객체를 찾기 위해 sliding windows 대신에 region proposals를 사용한다. Selective Search는 잠재적 bounding boxes를 생성하고, conv network로 특성을 추출하고, SVM으로 boxes를 점수화하고, 선형 모델로 bounding boxes를 조정하고, NMS로 중복된 detections를 제거한다. 이 복잡한 파이프라인의 각 단계는 정교하게 독립적으로 조정되어야하고, 결과 시스템은 테스트시에 이미지당 40초가 넘게 걸릴 정도로 매우 느리다. 

YOLO는 R-CNN과 조금 유사성을 공유한다. 각 grid cell은 잠재적 bounding boxes를 제안하고, convolutional 특성을 이용해서 boxes를 점수 매긴다. 하지만, YOLO는 같은 물체의 다중 탐지를 완화시키는데 도움이 되는 grid cell proposals에 공간적 제약을 가한다. 또한 Selective Search가 이미지당 2,000개의 boxes를 제안하는데 비해, 우리의 시스템은 이미지당 98개만을 제안한다. 마지막으로, YOLO는 개별적인 요소들을 단일의 공동 최적화 모델로 결합한다. 

#### ==Other Fast Detectors==

Fast, Fast R-CNN은 연산을 공유하고 Selective Search 대신에 NN을 이용하여 regions를 제안하도록하여 R-CNN의 속도를 높히는데 초점을 맞췄다. [Fast R-CNN](https://leechamin.tistory.com/215?category=839075), [Faster R-CNN](https://leechamin.tistory.com/221?category=839075) 
이 두 가지 R-CNN의 모두 속도와 정확성에 대해 향상했지만, real-time 성능에는 아직 모자르다. 

YOLO는 큰 detection 파이프라인의 개별 구성 요소를 최적화하려고 하는 대신 파이프라인을 없애고 설계상 신속하게 했다. YOLO는 다양한 물체를 동시에 감지하는 법을 배우는 범용 검출기(general purpose detector)이다.

#### ==Deep MultiBox==

MultiBox는 단일 object deteciton을 신뢰도 예측을 단일 클래스 예측으로 대체함으로 가능케 한다. 하지만 MultiBox는 보편적인 object detection을 하지 못하고, 여전히 큰 파이프라인의 일부분에 불과하므로 추가적인 이미지 패치 분류가 필요하다. YOLO와 MultiBox 모두 이미지 내의 bounding box를 예측하기 위해 convolutional 네트워크를 사용하지만, YOLO는 완전한 탐지 시스템이다. 

#### ==OverFeat==

CNN을 이용하여 localization을 하고 localizer를 적응시켜 탐지를 하도록한다. OverFeat은 효율적으로 sliding window 사용하려했지만 여전히 분리된 시스템이다. OverFeat은 localization을 위해 최적화지만, 탐지로는 아니다. DPM처럼, 예측을 할 때에 localizer은 지역 정보만을 본다. OverFeat은 전체적인 맥락에서 추론하지 못하고, 일관된 detection을 만들기 위해 상당한 후처리가 필요하다. 

#### ==MultiGrasp==

MultiGrasp는 하나의 객체를 보유하고 있는 이미지의 경우 파악 가능한 단일 영역만 예측하면 된다. 크기나 위치, 객체의 경계, 클래스를 예측하지 않고 알맞는 지역만을 찾는다. YOLO는 이미지 내의 다양한 클래스의 다양한 객체에 대해 bounding boxes와 클래스 확률을 예측한다. 

## ==4. Experiments==

먼저 YOLO와 다른 real-time detection 시스템을 PASCAL VOC 2007에 대해 비교할 것이다. 최종적으로는 YOLO가 새로운 도메인에 더 일반화를 잘한다는 것을 보여줄 것이다. 

### ==4.1 Comparison to Other Real-Time Systems==

Table.1 사진

빠른 detectors의 속도와 성능을 비교하는 표이다. Fast YOLO는 가장 빠르고 2번째로 정확한 real-time detector이다. YOLO는 Fast YOLO보다 10 mAP높게 기록하고 있디. (Real Time detectors에서)

### ==4.2 VOC 2007 Error Analysis==

* Correct : correct class and IOU > 0.5
* Localization : correct class,  0.1 < IOU < 0.5
* Similar : class가 유사하다. IOU > 0.1
* Other : class가 틀리다. IOU > 0.1
* Background : IOU < 0.1 (어느 객체에 대해서)

Fig.4사진
다양한 카테고리들에 대한 top N detections내의 localization, 배경 오류비율을 보여준다. 

YOLO는 객체를 올바르게 localize하기 위해 노력한다. Fast R-CNN은 localization 오류는 작지만 배경 오류 비중은 크다. 13.6%로 물체가 없는데 물체가 있다고 판단한 False Positive의 경우이다. 이는 YOLO(4.75%)의 3배 가량된다. 

### ==4.3 Combining Fast R-CNN and YOLO==

YOLO가 배경 오류를 덜 만들기 때문에, YOLO를 사용하여 Fast R-CNN에서 탐지한 배경을 제거하여 성능을 높인다. 

Table.2

YOLO + Fast R-CNN이 75.0으로 높은 mAP를 기록하고 있다. 

YOLO는 테스트시에 다른 종류의 실수를 만들기 때문에 Fast R-CNN의 성능을 높이는데 효과적이다. 

하지만 이러한 결합은 모델을 각각 돌리고 결과도 따로 합쳐야하기 때문에 YOLO의 속도를 높여주지 못한다. 

### ==4.4 VOC 2012 Results==

Table.3 사진

YOLO는 real-time detector만을 의미한다. Fast R-CNN + YOLO는 부분부분 높은 점수를 기록하고 있다.(cat,dog,train)

### ==4.5 Generalizability : Person Detection in Artwork==

Fig.5 

(a) : Picasso Dataset에 대한 precision-recall curve이다.
(b) : VOC 2007, Picasso, People-Art Datasets에 대한 결과이다. Picasso Datasets에 대해서는 AP, Best $F_1$ 모두 평가한다.

Natural 이미지에 대해 학습을 했지만 Artwork detection에도 강한 면모를 보이며 일반화를 잘 할 수 있다고 생각할 수 있다. Artwork와 natural 이미지는 픽셀 레벨이 매우 다르지만, 객체의 크기나 모양이 유사하기에 YOLO가 여전히 좋은 bounding boxes 와 detection을 예측할 수 있는 것이다. 

Fig.6사진
아래줄에서 2 번째의 사진에서 사람을 비행기라고 판단한 것 외에는 대부분 정확하다.

## ==5. Real-Time Detection In The Wild==

Webcam에 연결하여 YOLO의 real-time 성능을 측정할 수 있다. 


## ==6. Conclusion==

분류기 기반의 접근방식과 다르게, YOLO는 탐지 성능에 직접 대응하는 비용 함수에 바로 학습하고, 전체 모델을 공동으로 훈련한다.

정리해보자면 YOLO는 이미지를 단 한 번 보고, 단 한 번의 통합된 파이프라인을 거쳐서 객체를 인식하는 방법이다!
