# < Deep Neural Networks for Object Detection >

## "객체 인식의 시작"


## ==0. Abstract==

DNN(Deep Neural Networks)은 image classification에서 뛰어난 성능을 보였다. 이 논문에서는 한 단계 더 나아가서 분류뿐만 아니라, 다양한 클래스의 객체를 localizing하는 것을, DNN을 이용해서 객체를 인지(Object Detection)할 것이다. 조금의 네트워크를 통해 비용 대비 좋은 성능을 내는 것을 Multi-scale 추론이라고 하고 이 과정을 통해 Object Detection을 할 것이다. Pascal VOC 데이터에 대해 성능을 측정할 것 이다. 

- - -
## ==1. Introduction==

이미지에 대해서 완벽히 이해하기 위해서, 더 정확하고 자세하기 객체 인식을 하는 것이 중요하게 되었다. 이러한 맥락에서 이미지를 분류하는 것 뿐만 아니라, 정확하게 클래스를 추정하고 이미지내의 객체의 위치를 추정하는 것이 Object Detection이라고 불리는 것이다. 

여기서의 문제는 매우 도전적인데, 다양한 사이즈의 매우 큰 수의 객체를 제한된 computing source를 통해 인식하는 것이다. 

이 논문에서는 주어진 이미지를 통해서 여러개의 객체를 통해 bounding boxes를 예측할 수 있다고 말한다. 더 정확하게는 객체의 bounding box의 binary mask를 예측하는 DNN기반의 regression으로 가능하다고 말한다. 추가적으로 간단한 bounding box를 이용해서 마스크로부터 인식을 추출하는 것을 추론한다. localization precision을 증가시키기 위해서 DNN mask generation을 multi-scale로 적용하는한다.( Fig2 / 조금 큰 잘린 이미지들을 통해 )

이 논문에서는 DNN-based regression이 분류 뿐만이 아니라 geometric information에 대해서도 정보를 잘 잡아낸다는 것을 증명해냈다. 

그리고 multi-scale box를 앞으로 나올 개선 스텝을 따라 예측하면 정확한 탐지를 해낼 수 있다. 이와 같이, 출력 레이어 크기에 의해 제한되는 저해상도 mask를 예측하는 DNN을 이용해서 낮은 비용으로 픽셀 단위 정밀도를 얻어내는데 사용할 수 있다. 

추가적으로, 이러한 단순성은 광범위한 classes에 적용하기 쉽다는 장점이 있고, 또한 변형이 가능한 개체뿐만 아니라 rigid한 개체와 같은 광범위한 개체에서도 더 나은 감지 성능을 보여준다.



## ==2. Related Work ==

And/Or graph는 나무처럼 모델링 되었는데, And-nodes는 다른 파트를 나타내고, Or-nodes는 같은 파트에서의 다른 모드를 나타낸다. DNN도 유사하게, And/Or graph로 구성된 multiple-layer로 구성되어 있다. low-layers는 작은 일반 원형 이미지를 나타내고, high-layers로 구성되어 있다면, 객체 부분을 나타낸다. 

또한 그 이전에 사용되었던 것으로는 Gabor filters 나 HOG filters가 있는데, 이들은 학습의 어려움과 특정한 학습 과정을 이용하는 것에 난항을 겪었다. 

하지만 NN은 위의 모델들 보다는 generic하지만, 해석하기가 어렵다. 이 논문에서의 접근방식은 다음과 같다. 전체 이미지를 input으로 사용하고, regression을 통해서 localization을 실현한다. 따라서 NN의 더욱 효율적인 적용이 가능해진다. 

## ==3. DNN-based Detection ==

논문의 핵심 접근방법은 Fig.1에서 보여지는 것 처럼  object mask에 대한 DNN-based regression을 하는 것이다. regression model에 기반해서, 우리는 그 물체의 일부뿐만 아니라 전체 물체에 대한 masks를 생성할 수 있다.
localization의 정확도를 높이기 위해서는, DNN localizer를 하위 windows의 작은 세트에 적용한다. 그 전체 과정은 Fig.2에 나타나있다. 

## ==4. Detection as DNN Regression ==

DNN은 총 7개의 layers로 이루어져있고, 처음 5개는 convolutional이고 마지막 2개는 fully connected이다. 각각의 layer는 linear units가 반영된 non-linear 변환으로 사용된다. 3개의 conv layers는 추가적인 max pooling이 있다. 

논문에서는 위의 generic architecture를 localization에 사용한다. 마지막 layer에 softmax classifier를 사용하는 대신에, binary mask를 생성하는 regression layer를 사용한다. 

binary mask는 다음과 같다. 

$$
DNN(x;\Theta) \in \mathbb{R}^N
$$

$$$ \Theta $$$는 네트워크의 파라미터이고,  $$$N$$$은 픽셀의 총 갯수이다. output이 고정된 차원의 네트워크를 가진다면, mask의 고정된 사이즈는 $$$N = d \times d$$$로 예측할 수 있다. 이미지 사이즈로 조정된 이후에, binary mask는 하나 또는 여러개의 객체를 나타낸다. 이 픽셀이 지정된 클래스의 객체의 boundary box 안에 있으면 특정 픽셀에 값 1을 가지고, 그렇지 않으면 0이 되어야 한다.


네트워크는 이미지 $$$x$$$의 ground truth mask(실제값) $$$m \in [0,1]^N$$$을 예측하는 $$$L_2$$$ 에러를 최소화 하면서 학습이 된다.

식 min

위 식에서 Sum의 범위는 binary masks라고 불리는, bounding box처리된 객체가 포함된 이미지들의 training set D까지이다. 

기본 network가 매우 non-convex하고 최적이 보장되지 않기 때문에, loss function에 ground truth mask에 결정되는 각각의 결과에 대한 다양한 가중치를 이용하여 규제를 줄 필요가 있다. 직관으로는 대다수의 객체들이 이미지 크기에 비해 상대적으로 작고, 네트워크는 매 output에 0을 할당하는 변변찮은 해결책을 내놓기 쉽다. 이러한 바람직하지 못한 현상을 피하기 위해, 파라미터 $$$ \lambda \in \mathbb{R}^+$$$에 의한 ground truth mask에 속하는 0이 아닌 value들에 일치하는 결과의 가중치를 증가시키는 것이 도움이 된다. 수식의 앞부분이 규제 역할을 하는 것이다. 만약에 $$$ \lambda$$$ 를 작게 선택한다면, ground truth value가 0인 결과의 에러는 1일 경우에 비해 상당히 적은 처벌을 받을 것이다. 그리고 signal이 약해도 0이 아닌 값을 예측하도록 network를 유도할 것이다.

위의 규제식 부분만 가져오면 $$$m+\lambda $$$ 가 대각성분인 이루어진 diagonal matrix가 만들어지는데,  $$$  \lambda$$$가 작을수록 $$$ L_2$$$ error 또한 줄어들게 된다. 1일 경우에 loss가 더 높아서 0일 경우에 처벌이 덜 된다고 한 것이다. 그리고 1일 경우에 처벌이 더 강하므로 signal이 약해도 0이 아닌 value를 예측하도록 유도한다는 것이다.  

여기서는 225 X 225를 input으로 받고 output으로는 24 X 24 mask를 예측하는 네트워크를 사용하였다. 


## ==5. Precise Object Localization via DNN-generated Masks ==

위의 접근만으로도 high-quality의 masks를 생성해낼 수 있지만, 추가적인 문제가 더 존재한다. 

* 첫 번째로, 하나의 object mask는 서로 옆에 있는 객체들을 명확히하기에 불충분하다. 


* 두 번째로, 제한된 output의 size 때문에, 기존 이미지에 비해 더 작은 masks를 생성해낸다.

* 마지막으로 input에 full image를 쓰기 때문에, 작은 객체는 매우 적은 수의 input 뉴련에 영향을 미치기 때문에 인식하기가 어렵다.

위 issue들을 어떻게 다뤘는지는 앞으로 나올 내용에 포함되어 있다. 

#### ==5.1 Multiple Masks for Robust Localization==

목표가 bounding box를 만드는 것이기 때문에, 하나의 network로 object box mask를 예측하고, 4개의 추가적인 networks로 box의 절반 정도를 예측한다.

( 아래,위,왼쪽,오른쪽, 다음과 같이 표현된다. $$$ m^h,h \in \{ full,bottom,top,left,right \}$$$)

이 다섯 가지 예측은 지나치게 완벽하지만, 불확실성을 줄이고, masks의 실수에 대해 처리하는데 도움이 된다. 또한 동일한 유형의 객체 두 개를 서로 옆에 배치하면 생성된 5개 마스크 중 적어도 두 개 이상의 마스크가 객체를 같이 감싸지 않으므로 객체를 애매하지 않게 할 수 있다. 이렇게 하면 여러 개의 객체를 인지할 수 있을 것이다.

training time에, object box를 5개의 masks로 변환해줘야 한다. masks가 기존 이미지에 비해 더 작을 수 있기 때문에, ground truth mask의 크기를 network output의 크기만큼 낮춰주어야 한다. 


이미지 내의 객체의 존재를 나타내는 직사각형 $$$ T(i,j)$$$는,  network의 output $$$ (i,j)$$$ 로부터 예측된다. 이 직사각형의 좌측 상단 모서리는 $$$ (\frac{d_1}{d}(i-1),\frac{d_2}{d}(j-1))$$$ 로 표현되고, 사이즈는 $$$ \frac{d_1}{d} \times \frac{d_2}{d}$$$이다. $$$d$$$는 output mask의 크기이고, $$$ d_1,d_2$$$는 각각 이미지의 높이와 너비이다.  training 동안에 box  $$$ bb(h)$$$에 포함되는 $$$T(i, j)$$$의 일부로 예측되는 값을 $$$m(i, j)$$$로 할당한다.

식 베이지안

$$$ bb(full)$$$은 ground truth object box와 동일하다. $$$ h$$$에 의해서 $$$ bb(h)$$$는 기존 box의 4개의 면적에 해당한다. 

ground truth box $$$ bb$$$에 대한 결과 $$$ m^h(bb)$$$는 type $$$ h$$$(상,하,좌,우)인 network training time에 쓰인다. 

이 시점에서, output layer가 다섯 개의 masks를 모두 생성하는 모든 mask에 대해 하나의 network를 training 할 수 있다는 점에 유의해야한다. 그리고 5개의 localizer는 대다수의 layers와 특성들을 공유하고 있을 것이다. 이는 localizers들이 같은 객체를 다루고 있다는 점에서 매우 자연스러운 것이다. 좀 더 과감하게 생각해보면, 동일한 localizer로 다른 classes들을 다룰 수 있다고 생각해 볼 수 있다. 동일하게 작동할 것으로 보인다. 

#### ==5.2 Object Localization from DNN Output==

비록 output이 input 이미지보다는 작지만, binary mask를 input 이미지의 크기로 재조정함으로 해결할 수 있다. 목표는 output mask의 좌표에서의 좌측 상단 모서리 $$$ (i,j)$$$ 와 우측 하단 모서리 $$$ (k,l)$$$을 통 파라미터로 표시된 bounding box $$$ bb = (i,j,k,l)$$$을 추정하는 것이다.

이를 위해서 score : $$$ S$$$를 사용할 것인데, 이는 bounding box $$$ bb$$$와 masks간에 일치하는지를 표현하고, highest score를 가지는 boxex를 추론한다. bounding box가 mask에 의해 얼만큼 덮여있는지를 추정하는 것이다. 

식 (2)

여기서 $$$(i,j)$$$로 인덱싱된 모든 network output을 합하고 $$$m = DNN(x)$$$으로 표시한다. 위의 score 식을 5가지 mask 타입에 확장시켜본다면, final score은 다음과 같다.

식 (3)


$$$ \bar{h}$$$는 $$$ h$$$의 반대쪽 반을 말한다. 예를 들어, top mask는 top mask에 의해 잘 덮여있어야하고 bottom mask에 덮여있으면 안된다. 만약에 $$$ h = full$$$ 인 경우, $$$ \bar{h}$$$는 full masks가 $$$ bb$$$ 밖으로 벗어나면 score에 페널티를 받는 $$$ bb$$$ 주변의 직사강형 박스를 의미한다. 위의 요약으로는, 5개의 masks가 일치한다면 score는 높게 나타날 것이다. 

가능한 bounding boxes를 철저하게 찾기위해 위의 식 Eq.(3)을 score에 사용할 것이다. bounding boxex의 차원 평균이 이미지 차원 평균의 [0.1,...,0.9]와 같도록 하고, 10개의 다른 측면 비율을 training data에서의 객체의 boxes를 k-means clustering해서 추정한다. stride = 5 pixels로 이미지를 90개의 boxes를 이용해서 slide한다. 정확한 연산 횟수는 $$$5(2\times \#Pixels + 20\times\#Boxes)$$$ 회이며, 여기서 첫 번째 항은 통합 마스크 계산의 복잡성을 계산하고, 두 번째 항은 box score 계산을 설명한다.

final set of detection을 만들기 위해 2가지의 필터링을 해야한다. 

* 첫 번째로, boxes는 Eq.(2)에서 규정된 score를 매우 높게 유지해야한다.(ex. 0.5보다 높게)
* 두 번째로, 관심있는 classes와 positively 하기 classified된 것들로 학습된 DNN classifier를 적용해서 가지치기를 하는 것이다. 

#### ==5.3 Multi-scale Refinement of DNN Localizer==

network output에 대한 불충분한 해결책에 대한 문제는 다음 두 가지 방법으로 해결될 수 있다. 

* 첫 번째로, 여러개의 scale과 몇 개의 큰 sub-windows에 DNN localizer를 적용함으로써 해결할 수 있다. 

* 두 번째로, 추론된 bounding boxes에 DNN localizer를 적용하여 탐지를 개선한다. (Fig.2의 윗부분)

이미지를 더 높은 해상도를 내뱉는 network의 출력으로 덮기를 원하지만 동시에 각 객체가 적어도 하나의 창 안에 들어가고 이러한 창의 수가 적기를 원한다. 

위의 목표를 달성하기 위해서 3가지 scales를 이용한다. 
full image / 주어진 scale의 window 크기가 이전 scale의 window 크기의 절반인 다른 2 scales.

각 scale의 image를 창으로 덮어서 이 window들이 작은 중첩을 갖도록 한다(20% 정도). 중요한 것은 가장 작은 scale에서의 windows는 더 높은 해상도로 localization 할 수 있다. 

예측할 때에는, DNN을 모든 windows에 대해 적용한다. 이는 sliding window 방법과 조금 다른데, 왜냐하면 이미지당 약 40개 이하의 적은 windows를 평가해야하기 때문이다. 각 scale에서 생성된 object masks는 maximum 연산에 의해 병합된다. 이렇게 하면 image 크기의 mask 3개가 각각 다른 크기의 물체를 '보기' 시작하게 된다. 구현시, scale당 상위 5개의 탐지를 수행하여, 총 15개의 탐지가 발생했다. 

localization을 더 개선시키기 위해서는 DNN regression을 refinement해야한다. 고해상도에 localizer를 적용하면 탐지에서의 정확성을 증가시킬 수 있다. 

알고리즘

앞서말한 5절의 이야기를 코딩으로 풀어내는 것이다. 

## ==6. DNN Training==

이 network의 주목할만한 특징은 "단순함"이다. 분류기가 복잡한 구조없이 단순히 mask generation layer로 바뀌기 때문이다. 하지만 학습을 위해서는 대량의 학습 데이터를 필요로 한다. 데이터에는 각기 다른 크기의 객체가 거의 모든 지역에서 등장해야한다. (이미지의 구석구석 객체가 등장해줘야 학습이 잘 된다. 예외없이.)

mask generator를 학습시킬 때는, 각각의 이미지를 60%는 negative로 40%는 positive 샘플로 나누어서 몇 천개의 샘플들을 생성해낸다. 샘플이 관심있는 객체의 bounding box와 겹치지 않을 경우 negative라고 간주한다. Positive의 경우는 최소 80%의 영역으로 object bounding boxes와 겹쳐야한다. crops는 너비가 이전에 규정된 최소 scale과 전체 이미지의 너비 사이에 일정하게 분포되도록 샘플링한다.

detections를 최종 개선하는 classifier를 학습시키기 위해 유사한 준비단계를 거친다. 똑같이 60% 의 negative / 40%의 positive로 이미지에서 샘플링을 한다. Negative의 경우에는 샘플의 bounding boxes와 ground truth object boxes간에 Jaccard-similarity가 0.2 이하일 때를 말하고, positivesms 최소 0.6을 넘겨야 한다. 그리고 가장 유사한 bounding box를 가진 객체의 class를 labeling해야한다. 추가적으로, extra negative class는 regularizer와 필터의 질을 향상시키는 역할을 한다. 

localization 학습이 classification 학습보다 어렵기 때문에, high-quality & low-level filters가 적용된모델의 가중치를 이용해 시작하는 것이 중요하다. 그러기 위해서는, 먼저 classification network를 훈련하고 그 모든 layers의 가중치들을 localization에 재사용할 것이다. 


## ==7. Experiments==

**Dataset** : Pascal Visual Object Challenge (VOC) 2007 데이터 셋을 이용해서 test했다. 20개의 classes를 가지는 5,000장의 이미지를 test하는데 이용하였다. 파라미터 수가 많기 때문에, train / validation을 이용하였다. 성능평가를 위해서는 Precision-recall cureves와 class당 average-precision를 사용하였다. 

**Evaluation** : Table.1에 결과가 있다. 이 논문의 방식은 DetectorNet이다. detection score는 softmax classifier를 통해 계산되었다. 

초기 학습이 끝나고, training set에 대해 두 차례 negative를 찾았다. 그리고 원래 훈련 세트에 200만 개의 예를 더하고 False-Positive (실제:F / 예측 :P)비율을 줄였다.

Fig.3에서의 detection 사례처럼, 5개의 masks가 생성되어 시각화 되어있는데, 이는 DetectorNet이 큰 객체뿐만 아니라 작은 객체도 정확하게 찾아낼 수 있다는 것을 알 수 있다. 

공통적인 오탐지는 객체가 유사하게 보이거나 localization이 부정확할 때 였다. (Fig.3의 맨 아랫줄 왼쪽 사진)
후자의 문제, 즉 객체의 localization이 부정확한 경우는 training data에서의 객체의 정의가 애매하기 때문이다. 어떤 이미지는 새의 머리만 보이고, 다른 건 몸통만 보인다. 많은 경우에서, 몸과 얼굴이 모두 한 이미지에 있을 때 몸과 얼굴을 탐지하는 것을 알 수 있다. (즉, 객체의 모든 면이 다 training image에서 보여야한다.)

마지막으로, 정밀화(개선) 단계는 인식의 품질에 크게 기여한다. 이는 Fig.4에서 볼 수 있는데, precision-recall curve 그래프를 보면 DetectorNet의 첫 stage와 그 이후 정밀화를 거친 두 가지 경우가 존재한다. 눈에 띄게 좋아진 모습을 볼 수 있는데, 이는 True-Positive로 localized된 것이 많아져서 score가 증가된 것이다.

## ==8. Conclusion ==

DNN 기반의 mask generation regression을 이용한 간단한 공식을 통해 multi-scale에서 세밀하게 진행될 경우,강력한 결과를 낼 수 있다는 것을 알았다. 이러한 결과는 학습시 약간의 계산 비용으로 발생되며, 객체 유형 및 mask 유형별로 network를 학습해야 한다. 앞으로는 cost를 줄이고, 하나의 network로 여러개의 classes를 가진 객체를 인식하고, 확장해서 더 많은 classes를 가진 객체들 또한 인식하는 것을 목표로한다. 

Object Detection에 DNN이 적용된 것을 보니 앞으로 읽을 더 발전된 Object Detection 논문에 대해서 더 궁금하고, 빨리 읽어봐야겠다는 생각이 든다.