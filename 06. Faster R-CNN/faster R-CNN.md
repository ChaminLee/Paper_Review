# < Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks >

## *'RPN + Sharing Computation' Colabo*

## ==0. Abstract==
이전에 SPPnet, Fast R-CNN등이 등장하면서 running time을 줄이는등 성과를 내고있었다. 위 논문에서는, 모든 이미지의 convolutional features를 공유하면서 region proposal을 구하는데에 cost-free에 가깝게하는 Region Proposal Networks (RPN)에 대해 설명할 것이다. 
RPN은 fully convolutional network로서 동시에 물체의 경계를 예측하고, 물체의 위치에 대한 점수를 예측한다. RPN은 고품질의 region proposals를 만들기 위해서 end-to-end로 학습된다. Fast R-CNN에서 쓰였던 것과 같다. 여기서 더 나아가 RPN과 Fast R-CNN의 convolutional features를 공유함으로써 하나의 network로 합칠 것이다. 

## ==1. Introduction==
근래 Object Detection의 진보는 region proposal 방법의 성공과 R-CNN에 의해 이루어졌다. Region Proposal 방법중 하나인 Selective Search는 저레벨 features를 기반으로 픽셀들을 병합(greedy 알고리즘에 따라 / 유사도에 따라 점점 통합하는 것 )하는 가장 인기있는 방법이다. 하지만 현재에는 EdgeBoxes가 속도와 품질에 대해 최고의 tradeoff를 보여주고 있다. 그럼에도 불구하고, region proposal 단계에서는 시간을 많이 소비하고 있다.

 Proposal 연산속도를 가속하는 명백한 방법은 GPU를 위해 재구현하는 것이다. 하지만 재구현은 detection network 하류를 무시하고, 이에 따라 연산을 공유할 수 있는 매우 중요한 기회를 놓치게된다. 

이 논문에서, detection network 연산에서 proposal 연산을 거의 cost-free에 가깝게하여 효율적인 해결책을 제시하는 것을 알고리즘 변화를 통해 보여줄 것이다. RPN은 convolutional network를 공유하는 최신 detection network이다. 테스트 시에, convolutions를 공유함으로써, proposal 연산을 위한 한계비용(한 단위를 증가시킬 때 총비용이 얼마나 변화하는지를 나타내는 말)이 작다.

 Fast R-CNN처럼 region-based detectors에 쓰이는 convolutional feature maps를 이용하여 region proposals를 만드는데도 쓸 수 있다는 것을 알아냈다. convolutional features의 맨 위에, 추가적인 새로운 convolutional layers인, RPN(동시에 region 경계와 각 위치에 대한 객체의 점수를 예측하는)을 만들었다. RPN은 fully convolutional network 처럼 detection proposals를 생성해내기 위해 end-to-end로 학습되어진다. 

RPN은 넓은 범위의 척도와 종횡비(aspect ratio)를 가지고 효율적으로 region proposals를 예측할 수 있게 설계되었다.

Fig.1 사진
다양한 scale & size문제를 해결하기 위한 위한 서로 다른 전략들이다. (a) 이미지들의 피라미드와 특성 맵들이 그려져있다. 그리고 분류기는 모든 scales에 대해 실행된다. (b) 다양한 scales/sizes의 필터들의 피라미드는 특성맵에 대해서 실행된다. (c) regression function에 reference boxes의 피라미드를 사용한다. 

여기서 다양한 scales와 aspect ratio에 대한 참고역할을 하는 새로운  "anchor boxes" 라는 것을 설명한다. 논문에서의 계획방식은 다양한 scales나 aspect ratio의 이미지나 필터들이 하나하나 열거되는 것을 방지한다는 점에서 Fig.1 (c)에서 regression references의 피라미드와 유사하다. 이 모델은 single-scale 이미지를 사용하여 train/test할 때 성능이 좋았고, 실행 속도도 좋았다.

Object Detection network인 Fast R-CNN과 RPN을 통합하기 위해, proposals는 고정시킨채로, region proposal을 fine-tuning하는 것과 object detection을 위한 fine-tuning하는 것을 번갈아가며 하도록 학습 방법을 기획했다. 이 방법은 빠르게 수렴하고, 두 작업 간에 공유된 convolutional features를 포함함 통합된 network를 생성해낸다. 
또한 RPN + Fast R-CNN은 기존의 Selective Search의 계산량을 줄여준다.

Faster R-CNN + RPN은 COCO2015 대회의 ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation에서 우승을 거머쥐었다. RPN은 데이터로부터 region을 제안하는 방법을 완전히 학습하기에, 이에 따라 더 깊고 더 표현적인 기능으로부터 이득을 볼 수 있다. 이러한 결과를 통해 논문에서의 방법이 실용적인 사용에 대한 해결책에 대해 비용-효율적일 뿐만아니라 detection 정확도를 높이는데도 효율적인 방법이라는 것을 알 수 있다.

## ==2. Related Work==
##### ==Object Proposals.==
Object Proposal 방법은 detectors와 독립적인 외부적인 모듈로 채택되었다. 

##### ==Deep Networks for Object Detection.==
R-CNN은 proposal regions를 object 카테고리나 배경으로 분류하도록 CNN을 end-to-end로 학습한다. R-CNN은 주로 분류기 역할을 하지만, 객체의 경계(bounds)를 예측하지는 못한다. (bounding box regression에 의한 refining을 제외하고) 

Fig.2
Faster R-CNN은 object detection을 위한 단일의, 통합된 네트워크이다. RPN 모듈은 이 통합된 네트워크의 'Attention' 역할을 한다. 

MultiBox 방법론은 마지막이 동시에 다양하게 클래스와 무관하게 박스를 예측하고, single-box를 일반화하는 fully connected layer인 network로부터 region proposals를 생성한다. 이 MultiBox proposal network는 논문에서의 fully convolutional 계획과는 다르게, 단일 이미지 조각이나, 복합적이고 커다란 이미지 조각에 대해서 적용된다. MultiBox는 proposal과 detection network간에 특성들을 공유하지 않는다.
Fast R-CNN에서는 공유된 convolutional features를 이용하여 end-to-end 로 detector를 학습할 수 있어서 놀라운 성능과 속도를 보여준다.

## ==3. Faster R-CNN==

논문에서 말하는 Faster R-CNN은 두 가지 모듈로 이루어져있다. 

* Region을 제안하는 Deep fully convolutional network
* 제안된 regions를 사용하는 Fast R-CNN detector

전체 시스템은 단일적이고, object detection을 위해 통합된 네트워크이다. (Fig.2) RPN 모듈은 'attention' 메커니즘이라는 NN의 인기 있는 방식을 사용하여 Fast R-CNN 모듈이 어디를 봐야 하는지 알려준다. 뒤에 나올 3.1에서는 region proposal을 위한 network의 특성과 설계에 대해서 설명할 것이다. 3.2에서는 공유된 특성들을 가지고 두 가지 모듈을 학습하는 알고리즘에 대해 설명할 것이다.

### ==3.1 Region Proposal Networks==

RPN은 (모든 크기의) 이미지를 입력으로 받고, 출력으로는 각각 object에 대한 점수(class인지, background인지)를 지닌 직사각형 모양의 object proposals 세트를 출력한다. 이러한 과정을 fully convolutional network로 모델링한다. 궁극적인 목표가 Fast R-CNN과 연산을 공유하기 위함이기에, 논문에서는 두가지 nets가 공통적인 convolutional layers sets를 공유한다고 가정한다.

* ZF model = 공유가능한 Conv layers : 5
* VGG 16 = 공유가능한 Conv layers : 13

위 두가지 모델에 대해서 실험을 진행했다.

region proposals를 생성하기 위해서, 공유된 마지막 convolutional layer를 통해서 출력된 convolutional feature map 위를 작은 network로 슬라이딩 할 것이다. 각 슬라이딩 윈도우(nxn)는 저차원 특성으로 매핑된다. 이 특성들은 두 개의 fully connected layers(box-regression layer(reg) & box-classification layer(cls))로 보내지게 된다. 이 논문에서는 입력 이미지의 receptive field가 크다는 것을 감안하여, n을 3으로 사용한다. 

#### ==3.1.1 Anchors==

각 sliding-window 위치에서, 동시에 다수의 region proposals를 예측한다. 여기서 각 위치에 대한 가능한 최대 region proposals의 수는 k로 표시된다. 그래서 reg layers는 좌표를 포함하기에 4k, cls layers는 각 proposals에 대해 객체인지 아닌지에 대한 확률을 추정하기에 2k이다. k proposals는 k reference box에 대한 매개변수로 표시되며, 이를 "Anchor"라고 부른다. (sliding window의 각 위치에서 Bounding Box의 후보로 사용되는 상자)

Fig.3
왼쪽 : RPN / 오른쪽 : RPN proposals를 이용한 detections(PASCAL VOC 2007). 넓은 범위의 scales와 aspect ratios로 객체를 detects한다.

Anchor는 sliding window의 중앙에 위치하며, scale&aspect ratio와 연관된다. 기본값으로 3 scales & 3 aspect ratios를 사용하여 각 sliding position에 대해 k=9 anchors를 도출한다.

##### ==Translation-Invariant Anchors==

접근방식중에 중요한 특성은 anchor와 anchor에 상대적인 proposals 연산하는 기능면에서 모두 translation invariant하다는 것이다. 만약에 이미지속의 객체가 이동했다면 proposal도 이동되어야 한다. 그리고 proposal을 예측하는 것도 마찬가지로 전/후 이미지에서 같은 proposal을 만들어야한다. (MultiBox는 보장되지 않고 / 논문의 방법은 보장된다 ) 그리고 MultiBox에 비해 파라미터 수가 적기 때문에 작은 양의 데이터에 대해서 발생하는 overfitting 문제의 위험이 낮게 예상된다. 

##### ==Multi-Scale Anchors as Regression References==

Fig.1에서 본 것 처럼, multi-scale 예측을 위한 두 가지 방법이 있다. 

* 이미지 / 특성 피라미드 기반 (DPM, CNN 기반의 방법들)

	이미지들은 다양한 scales에 대해 resized되고, feature maps는 Fig.1의 각 scale에 대해 계산된다. 이 방법은 종종 유용하지만 시간이 많이 소요된다. 

* 다양한 scale의 sliding windows를 feature maps에 사용하는 것이다. 

	예를 들어, DPM에서 다른 aspect ratio의 모델은 각각 다른 필터 사이즈(5x7, 7x5)를 이용하여 개별적으로 학습되어진다. 만약에 다양한 scale을 사용한다면 Fig.1의 (b)에서 말하는 필터들의 피라미드로 생각될 수 있다. 이 방식은 보통 첫 번째 방식과 같이 채택되어서 사용된다. 

비교로, anchor-based 방법은 anchors의 피라미드를 구성하는데 이는 더 비용 효율적이다. 이 방법은 다양한 anchor boxes의 scales & aspect ratios를 이용하여 bounding boxes를 예측하고 분류하는 방법이다. 

이는 이미지와 single scale의 feature maps에 의존하고, single size의 필터들을 사용한다. 이러한 multi-scale anchors는 추가적인 비용없이 특성들을 공유하는데 핵심 요소이다.

#### ==3.1.2 Loss Function==

RPN을 학습시키기 위해서는, 두 개의 클래스 라벨을 각 anchor에 할당해야한다. Positive label을 다음 두 가지 anchors이다.

< Positive Anchor >
* Ground-truth box와 가장 높은 IoU를 가지는 anchor/anchors
* Ground-truth box와 IoU > 0.7 이상인 anchor 

두 번째 방법이 Positive sample을 찾지 못하는 (ex. IoU=0.6) 경우가 희소하게 있기 때문에 첫 번째 방법을 채택한다. 

< Negative Anchor >
* Ground-truth box와 IoU < 0.3 인 anchor

Positive, negative 둘 다 아닌 anchors는 (목표)학습에 쓰이지 않는다.

다음과 같은 목표 함수인 multi-task loss를 최소화하려한다. 

$$L(\{p_i\},\{t_i\}) = \frac{1}{N_{cls}}\sum L_{cls}(p_i,p_i^*) +
\lambda\frac{1}{N_{reg}}\sum p_i^*L_{reg}(t_i,t_i^*) $$

* $i$ : mini-batch에서의 anchor의 index
* $p_i$ : anchor $i$가 객체인지 아닌지에 대한 예측 확률
* $p_i^*$ : ground-truth label, 1=positive(객체) / 0=negative(배경)
* $t_i$ : 예측된 bounding box의 4개의 좌표
* $t_i^*$ : positive anchor와 연관있는 ground-truth box
* $L_{cls}$ : 객체인지 아닌지, 두 클래스에 대한 log loss
* $L_{reg}$ : $L_{reg}(t_i,t_i^*) = R(t_i-t_i^*)$, smooth L1 loss를 상용한다.
* $N_{cls}$ : normalization 값(mini-batch 크기 만큼)
* $N_{reg}$ : normalization 값(anchor locations의 갯수) 
* $\lambda$ : 기본값은 10이고, reg/cls간에 동등하게 가중치를 부여하기 위한 term

bounding box regression의 $t_i,t_i^*$의 4개의 좌표들이다.

$$t_x=(x-x_a)/w_a, t_y = (y-y_a)/h_a \\
t_w = log(w/w_a),  t_h = log(h/h_a) \\
t_x^* = (x^*-x_a)/w_a,  t_y^* = (y^*-y_a)/h_a \\
t_w^* = log(w^*/w_a), t_h^* = log(h^*/h_a)$$

$x,y,w,h$는 각각 box의 중앙 좌표, 넓이, 높이를 나타낸다.
$x,x_a,x^*$는 각각 예측된 box, anchor box, ground-truth box이다.

#### ==3.1.3 Training RPNs==

Mini-batch는 positive/negative anchor를 보유하고 있는 단일 이미지로부터 생성된다. 256개의 anchors를 이미지에서 샘플링하여 mini-batch의 loss를 계산한다. (Anchor의 비율 Positive:negative=1:1) 만약에 positive sample이 128개 보다 적다면 나머지를 negative로 채운다. 
즉, RPN에서 weights 는 conv layer의 weight 와 cls / reg layer의 weight 이다. 이 weight들의 최적값을 찾는 것이 RPN의 목적이다.

### ==3.2 Sharing Features for RPN and Fast R-CNN==

features를 공유한 networks를 학습하는 3가지 방법에 대해 볼 것 이다.

1) Alternating training
먼저 RPN을 학습시킨다. 그리고 proposals를 Fast R-CNN을 학습하는데 사용한다. Fast R-CNN으로 tuned 된 network를 RPN을 초기화할 때 사용한다. 이 과정을 계속 반복한다. 

2) Approximate joint training
RPN과 Fast R-CNN을 하나의 network로 통합한다. 

3) Non-approximate joint training
RPN으로 예측된 bounding boxes를 input으로 한다. 이 과정에서는 box 좌표와 다른 RoI pooling layer가 필요하다. 

##### ==4-Step Alternating Training.==

1) RPN을 학습시킨다. RPN은 ImageNet-pre-trained 모델로 초기화되고, region proposal task를 위해 end-to-end로 fine-tuned 된다. 

2) (1) RPN에서 생성된 proposals를 사용한 Fast R-CNN으로 분리된 detection network를 학습시킨다. 이 network도 ImageNet-pre-trained 모델로 초기화된다. (여기서 두개의 networks는 convolutional layers를 공유하지 않는다) 

3) RPN 학습을 초기화하기 위해 detector network를 사용하는데, 공유된 convolutional layers를 고정하고, RPN에 대해 고유한 layers만 조정한다.

4) 공유된 convolutional layers를 고정하고, Fast R-CNN의 고유 layers를 fine-tune한다. 

### ==3.3 Implementation Details==

Multi-scale feature extraction은 정확도를 향상시키지만, 정확도대비 속도(tradeoff)가 그렇게 좋지 못하다. 
앞서 말한 것 처럼, 논문에서의 방식은 다양한 크기의 regions를 예측하는데 이미지 피라미드 / 필터 피라미드가 필요없기에 running time을 절약할 수 있다.
 이미지는 모두 single scale (resize = 짧은 변 기준 600) 3개의 scale과 3개의 aspect ratio를 가지고 9개의 anchor 생성 함. 다량의 이미지를 생성 하지 않기 때문에 효율적이다.
이미지에 경계에 존재하는 anchor는 학습시에 무시된다. 즉, 사진당 20000개의 anchor중 6000개만 사용한다. (이 중에서 256개만 샘플링해서 batch에 사용). 하지만 test에서는 무시하지 않는다.
NMS를 사용해서 전체 6000개의 anchor 중, IOU > 0.7 이상인 2000개의 proposals만 남긴다. 즉, 학습 시에는 2000개의 proposals을 학습하지만, test시에는 속도 및 정확도 테스트를 위해 다양한 top-N proposals을 사용하여 실험을 진행하였다.

## ==4. EXPERIMENTS==

### ==4.1 Experiments on PASCAL VOC==

RPN + Fast R-CNN이 300개의 proposals를 사용하며 59.9%의 mAP를 달성했다. RPN을 사용함으로써 convolutional 연산을 공유하여, SS / EB보다 더 빠른 속도를 낼 수 있다.

##### ==Ablation Experiments on RPN.==
1) RPN과 Fast R-CNN 사이에서 convolutional layers를 공유했을 때의 효과에 대해 보여준다.

Table.2

Table.2를 봐도 RPN+ZF shared일 때, mAP가 가장 높은 것을 볼 수 있다. 또 다른 놀라운 것은, top-ranked 6000 RPN proposals(without NMS)의 mAP는 55.2%로, NMS가 mAP에 악영향을 미치지 않는다는 것을 알 수 있다. 

$cls$의 점수같은 경우는 가장 높게 ranked된 proposals의 정확도에 따른다는 것을 알았다. 반면에 $reg$를 없이 사용할 경우 raw한 anchor들만을 사용해서 정확한 detection에는 불충분하다. regressed box bounds가 고품질의 porposals를 만들기 때문이다. 그리고 RPN+VGG16이 RPN+ZF보다 성능이 더 좋다는 것을 확인했다.

##### ==Performance of VGG-16.==

Table.3

위에서 보여진 것 처럼, SS보다 RPN+VGG를 통해 생성된 proposals가 더 정확하다.

Table.6,7
더 자세한 숫자들을 보여준다. 

Table.5

VGG16은 detection과 proposal에 총 198ms가 소요된다. convolutional features가 공유되었을 때는, RPN만 해도 추가 layers를 계산하는데 10ms밖에 걸리지 않는다. 

##### ==Sensitivities to Hyper-parameters.==

Table.8

3 scale, 3 aspect ratio를 볼 수 있다. 각 위치에 대해 하나의 anchor만 사용했을시에는, 3~4% mAP가 떨어지는 것을 볼 수 있다. 3 scales, 3 aspect ratio를 사용하는 것이 mAP가 더 높다. 

3 scales / 1 aspect ratio vs 3 scales / 3 aspect ratio의 경우에 mAP가 크게 차이나지 않는 것을 보아 두 요소가 각각 독자적으로 성능에 영향을 주지 않는다는 것을 알 수 있다.

Table.9

$\lambda$값이 변해도 mAP에는 큰 변화가 없기에 넓은 범위의 $\lambda$에 대해 둔감하다는 것을 알 수 있다.

##### ==Analysis of Recall-to-IoU.==
Fig.4

이 metric은 평가하는 것 보다 proposal을 진단하는데 더 적합하다.
N proposals는 앞선 방식의 top-N ranked를 통해 생성된 것이다. Fig.4를 보면 더 적은 proposals에서 다른 방법들에 비해 더 높은 Recall을 보여주는 것을 보아 RPN이 더 훌륭한 방법임을 알 수 있다. 

##### ==One-Stage Detection vs. Two-Stage Proposal + Detection.==
Table.10

이 실험은 단계적으로 region proposals와 object detection하는 것이 효과적이라는 것을 정당화시켜준다. 논문에서의 방법인 Two-stage Proposal + Detection이 mAP 58.7%로 더 좋은 결과를 내는 것을 알 수 있다. 

### ==4.2 Experiments on MS COCO==
Table.11

mAP@.5 와 mAP[.5, .95]에 대한 차이는 negative sample을 규정하는 차이와 mini-batch 크기에 원인이 있다고 추측했다.

##### ==Faster R-CNN in ILSVRC & COCO 2015 competitions==

깊이를 100 layers 더 늘려도 여전히 유효하다는 것을 보여주었다.

### ==4.3 From MS COCO to PASCAL VOC==

Table.12

대규모 데이터는 deep neural network를 향상시키는데 매우 중요하다.

## ==5. CONCLUSION==

region proposal 생성을 위해서는 "RPN"이 효율적이고 정확하다는 것을 증명했다. 하류 detection network와 convolutional features를 공유함으로써, region proposal 단계는 거의 cost-free에 가까워졌다. 
학습된 RPN은 또한 region proposal의 품질을 향상시키고 전체 object detection 정확도 또한 향상시킨다. 

RPN과 Sharining convolutional features(computation)가 핵심 키워드인 것 같다.
