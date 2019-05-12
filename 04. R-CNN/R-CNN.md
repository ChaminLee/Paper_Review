# < R-CNN : Rich feature hierarchies for accurate object detection and semantic segmentation Tech report >
## * Region Proposal X CNN*

## ==0. Abstract==

R-CNN은 2가지 아이디어를 기반으로 탄생했다. 

(1) localize와 object segment를 위해 high-capacity CNN을 상향식으로 region proposal을 적용할 수 있다. (region proposal + CNN)

(2) labeled된 training data가 부족할 때, domain 기반으로 fine-tuning거친 사전 지도학습(pre-trained)을 이용해서 성능을 매우 높일 수 있다. (pre-trained + fine-tuning)

## ==1. Introduction==

Visual recognition 분야에서는 CNN이전에 SIFT 와 HOG를 이용하거나, ensemble 효과를 통해 조금의 성능 발전을 이루었다. 이후에 Lecun은 back propagation을 통한 SGD로 CNN을 효율적으로 학습시킬 수 있다는 것을 보여주었다.  2012년에 Krizhevsky가 ILSVRC image classification에서 CNN을 이용하여 높은 accuracy를 보여주어서, CNN에 대한 관심이 다시 불 붙기 시작했다. 

그러면서 ImageNet에 대한 CNN classification의 결과를 PASCAL VOC object detection에 일반화할 수 있을지가 궁금증으로 드러났다. 이에 대해, 본 논문에서는 CNN이 기존의 HOG 기반의 시스템들에 비해 보다 더 PASCAL VOC에서의 object detection의 성능을 높게 이끌 수 있다는 것을 보여준다. 이를 해내기 위해, 두 가지 문제에 집중하고있다. 두 가지 문제는 다음과 같다. 

* deep network를 이용하여 객체를 Localizing 하는 것


* 조금의 labeled 된 detected data만을 이용한 고성능 모델을 훈련시키는 것

image classification과 달리 object detection은 이미지 내의 객체에 대해 localizing을 필요로한다. CNN localization 문제를 "recognition using regions" (object detection & semantic segmentation에서 성공적)로 해결한다.

Test-time의 과정을 다음과 같다.

(1) 2,000개 정도의 input image에 대해 category-independent한 region proposals를 생성해낸다.

(2) 각 proposal에 대해 CNN을 이용하여 일정한 길이를 가진 feature vector를 추출한다.

(3) 그리고 선형 SVM을 이용하여서 각 regions를 category별로 분류한다. 

Region proposals과 CNN을 합쳐서 사용하기 때문에 이름이 R-CNN이다. 

OverFeat은 sliding-window CNN을 탐지에 사용하고 ILSVRC2013 까지만해도 최고의 성능을 보여주었다. 하지만 R-CNN(mAP:31.4%)이 OverFeat(mAP:24.3%) 확실히 능가하는 것을 알 수 있었다. 

 두 번째로 직면하는 문제로는 labeled data가 부족하고 현재 이용가능한 양으로는 large CNN을 훈련시키기에서는 부족하다는 것이다. 
 
data가 부족할 때, 대용량 CNN을 훈련을 하기 위해서는 작은 dataset에 대해 domain-specific fine-tuning을 거치고, 큰 보조 dataset에 대한 supervised pre-training을 이용하는 것이 효육적이라고 말한다. 실험을 통해 fine-tuning이 mAP를 8%정도 성능 향상을 시켰다는 것을 확인했다. 

이 시스템은 상당히 효율적이다. class-specific 연산은 오직 작은 matrix-vactor 와 greedy NMS(non-maximum suppression)뿐이다. 

NMS(non-maximum suppression)은 중심 픽셀을 기준으로 8방향의 픽셀 값들을 비교하여 중심픽셀이 가장 클경우 그대로 두고, 아닐 경우에는 제거하는 과정이다. 

사진 3바3

중심 픽셀의 값이 제일 크지 않으므로 제거하게 되는 것이다. 이를 sliding하면서 진행하면서 뭉개진 직선의 주변 값들을 지워주면서 원래의 직선을 구할 수 있다. 

결과사진

이 논문에서는 간단한 bounding-box regression 방법을 통해 mislocalization(오류)을 줄일 수 있다는 것을 증명해냈다. 그리고 R-CNN이 regions에 대해 작동하기 때문에 semantic segmentation으로 확장될 수 있다는 것이 당연하다는 것을 알고 가야한다. 

## ==2. Object Detection with R-CNN==

object detection 시스템은 3가지 모듈로 이루어져있다.

* 첫 번째 모듈은 category-independent region proposals를 생성한다. proposals는 detectors에 사용할 수 있는 후보 탐지 세트이다.

* 두 번째 모듈은 각 region으로 부터 고정된 길이의 feature vector를 추출하는 large CNN이다.

* 세 번째 모듈은 class-specific linear SVM이다. 

### ==2.1. Module design==

##### ==Region proposals.==

Fig.2 사진

class에 무관하게 image에서 region을 추출하는 것이다. 추출시에는 selective search를 사용한다. 

##### ==Feature extraction.==

region proposal에서 4096차원의 feture vector를 추출한다. features는 mean-substracted 227X227 image를 5개의 conv layers, 2 fc layers를 통해 forward propagating하여 계산된다. 그래서 연산을 위해 먼저 사이즈를 조정한다. 사이즈나 region후보의 aspect ratio(종횡비)에 관계없이, 요구되는 사이즈를 위해 tight bounding box 주변의 모든 픽셀드를 warpping(워핑)한다. ( 이후에 더 설명 )


- - -


** < Appendix A : Object proposal transformations >** 

input은 227 X 227. CNN inputs로 유효한 두 가지 object proposals 변환 방법에 대해 평가했다. 

Fig.7

(1-1) "tightest square with context" : 각각의 object proposal을 가장 좁은 square로 감씨고, 해당 사각형에 포함된 이미지를 CNN input size로 확장하는 것.  [ (B) 참고 ]

(1-2) “tightest square without context” : 기존의 object porposal을 둘러싼 이미지 내용을 제외한다. [ (C) 참고 ]

(2) "warp" : 각 object proposal을 비등성적으로 CNN input size로 변환한다. [ (D) 참고]

context padding(p)는 object proposal을 둘러싸는 가장자리 크기를 말한다. p=0이면 Fig.7의 위에 줄과 같고, p=16일 경우 아래의 줄과 같다. 배경을 얼마나 가져가느냐의 차이이다.
모든 방법에서 직사각형이 이미지 크기를 벗어나면 missing data는 이미지의 평균으로 대체된다.

p=16을 이용한 warping이 다른 방법들 보다 성능이 좋은 것으로 나타났다. 
- - -


### ==2.2. Test-time detection==

(1) test image에 대해 selective search로 2,000dml region proposals를 추출한다.

(2) 각 proposals를 warp하고 CNN을 통해 features를 계산한다. 

(3) 각 class에 대해, 추출된 feature vector를 input으로 SVM으로 score를 계산한다.

(4) 이미지에 대해 모든 scored regions이 주어지면, NMS(non-maximum suppresion)를 이용하여 class socre가 높은 region과 IoU(Intersection-over-Union)가 정해준 threshold 보다 큰 region들을 제거한다. 
 
##### ==Run-time analysis.==

두 가지 특성이 detection을 효율적으로 만든다. 

(1) 첫 번째로, 모든 CNN 파라미터들을 모든 카테고리에 대해 공유된다. 

(2) 두 번째로, CNN으로 연산된 feature vectors는 다른 접근방식들과 비교했을 때, 저차원이다. 

결론적으로 효율성은 region proposals만이 이끌어낸 것이 아니라, features를 공유한 것도 한 몫했다는 것이다. 

### ==2.3. Training==

##### ==Supervised pre-training.==

large auxiliary dataset인 ILSVRC에 대해 pre-trained된 CNN을 이용한다. 

##### ==Domain-specific fine-tuning.==

CNN을 detection 그리고 새로운 도메인에 적용하기 위해서 warped region proposals만을 이용해서 SGD를 사용해서 CNN 파라미터들을 훈련시킨다. 

object detection을 하기 위해 classification layer를 (Object class 개수(N) + background(1))으로 바꾸고 초기화해준다. 

* Positive Sample : ground-truth box와 IoU 값이 0.5이상인 region (class와 무관하게 )

* Negetive Sample : Positive Sample 이외의 나머지 region

Mini-Batch = 128을 구성하기 위해 SGD iteration 마다, 모든 class의 positive sample 32개, 그리고 96개의 background(negative sample)를 사용한다. 

##### ==Object category classifiers.==

차를 탐지하기 위해 binary classifier를 학습하는 상황을 생각해보자. 차를 포함하고 있는 region은 positive sample일 것이다. background region도 유사하게, 차가 없을 때는 Negative sample이어야 한다. positive / negative를 나누는 threshold를 정해는 것은 매우 중요하다. threshold에 따라 mAP가 증가하기도 감소하기도 한다. 선정 방식은 다음과 같다. 

* Positive Sample : 각 class별 object의  ground-truth bounding boxes

* Negative Sample : 각 class별 object의 ground-truth와 IoU가 0.3 미만인 region

- - -

** < Appendix B : Positive vs. negative examples and softmax >**

왜 CNN fine-tuning 과 object detection SVM 학습시에 positive / negative를 다르게 규정하는지에 알아볼 것이다. 

먼저 fine-tuning 시에, 최대 IoU 값을 가지는 object proposal을 ground-truth 에 매핑했고, ground-truth와의 IoU가 최소 0.5 이상 일 때, Positive label을 붙였다. 나머지는 background이다. 

반대로, SVM을 학습할 때는, 각각의 클래스에 대한 positive sample로서 ground-truth boxes만 가져가고, IoU=0.3 미만의 proposal은 해당 클래스에 대한 negetive sample로 처리한다. 0.3 IoU 이상이지만, ground-truth는 아닌 proposals는 무시한다.

처음에 fine-tuning 시에, SVM을 학습할때 사용했던 동일한 positive / negative sample을 사용했다. 하지만 현재의 positive / negative 정의일 때 보다 더 낮은 성능을 보였다. ( P/N 개념 분리(fine-tuning vs SVM)해야한다. ) 그래서 따로 정의해서 사용하는 것이다. 

두 번째로, 왜 fine-tuning 이후에 SVM을 훈련시키는가에 대한 대답은 다음과 같다. 처음에 더 깔끔하게 21-way-softmax regression classifier(50.9%)를 SVM(54.2%)대신에 적용했지만 성능이 좋지 않았다. 이러한 성능 저하는 fine-tuning에서 positive sample의 정의가 정확한 localization를 강조하지 못하고, softmax classifier는 SVM 학습에 쓰인 "hard negative"에서 샘플링하지 않고, negative samples에서 랜덤 샘플링했다는 것 등을 포함한 여러 복합적인 요소에 의해 설명된다.

위 결과는 fine-tuning 후 SVM을 학습하지 않아도 동등한 성능을 얻을 수 있다는 것을 보여준다. 

- - -

### ==2.4. Results on PASCAL VOC 2010-12==

Table.1 사진

Table.1을 보면 R-CNN이 다른 방법들 보다 좋은 성능을 보이는 것을 확인할 수 있다. BB(Bounding Box regression)을 이용할 때, 성능이 더 올라감을 알 수 있다.

### ==2.5. Results on ILSVRC2013 detection==

Figure.3

PASCAL VOC에서 보다 분류해야할 class가 더 많아서 mAP는 낮지만 다른 방법론들에 비해 R-CNN이 더 좋다는 것은 계속적으로 확인할 수 있다. 

## ==3. Visualization, ablation, and modes of error==

### ==3.1. Visualizing learned features==

$$$ pool_5$$$의 feature map을 flatten하고 내림차순으로 정리하여 값이 높은 region을 기준으로 하여 정리한 것이다. 

Figure.4

하지만 두 번째 줄을 보면 구멍이 뚫려있는 물체를 개로 착각하는 등, 마지막 줄에 빛이 비치는 것을 잘못 착각하기도 한다. 이를 통해 network는 모양, 텍스쳐, 색상 및 재료의 특성에 영향을 받는 것을 알 수 있다. 이후의  fully connected layer  $$$  fc_6$$$는 풍부한 특징의 커다란 구성 세트를 모델링 할 수 있다. 


### ==3.2. Ablation studies==

##### ==Performance layer-by-layer, without fine-tuning. ==

Table.2

$$$ pool_5$$$ 는 9X9X256으로 9216 차원이고,  $$$ fc_6$$$은 4096차원,  $$$ fc_7$$$은 4096차원이다. 

Table.2에서 1~3 줄을 보면 $$$ fc_7$$$이 $$$ fc_6$$$ 보다 더 좋지 못한 것을 볼 수 있다. 이를 통해 mAP가 줄어들지 않더라도 CNN의 파라미터를 줄일 수 있음을 알 수 있다. 더욱 놀라운 것은 $$$fc_7$$$과 $$$fc_6$$$을 모두 제거하면 $$$pool_5$$$가 CNN parameters의 6%만을 사용하여 계산되지만 상당히 좋은 결과를 얻을 수 있다는 점이다.

##### ==Performance layer-by-layer, with fine-tuning. ==

fine-tuning을 거친 CNN의 결과를 볼 것이다. Table.2의 4~6번째 줄을 보면 54.2%로 증가한 것을 볼 수 있다. 이를 통해 $$$ pool_5$$$는 일반적이며, 도메인별 학습을 할 경우에 개선이 된다는 것을 볼 수 있다. 

##### ==Comparison to recent feature learning methods.==

이 부분은 그냥 읽어보면 될 것 같다. 이전 방법론들과 R-CNN의 성능을 수치적으로 비교해석한 것이다. 

### ==3.3. Network architectures ==

Table.3 사진

R-CNN 의 성능에는 어떤 architecture로 구성하느냐가 큰 영향을 끼친다. 학습시 유일한 차이는 GPU 메모리에 맞추기 위해 더 작은 minibatches를 사용했다는 것뿐이다. Table.3를 보면 O-Net을 이용한 R-CNN이 T-Net을 이용한 R-CNN의 성능을 능가하는 모습을 볼 수 있다.

### ==3.4. Detection error analysis ==

Fig.5

background나 object classes에 대한 confusion보다 Loc으로 나타나는 poor localization이 errors의 주요 원인이다. 

Fig.6

Object detection을 하고자 하는 object의 특징들인 occlusion (occ), truncation (trn), bounding-box area (size), aspect ratio (asp), viewpoint (view), part visibility (part) 등의 문제가 있을 때의 성능을 보여준다. 이전의 방법론인 DPM보다는 R-CNN이 더 나은 것을 볼 수 있고, R-CNN 중에서도 fine-tuning, bounding-box regression을 활용했을 때 더 좋은 성능을 보임을 알 수 있다. 

### ==3.5. Bounding-box regression ==

selective search region proposal의 $$$ pool_5$$$ 특성을 고려하여 새로운 detection window를 예측하기위해 선형회귀 모델을 훈련시킨다. 이는 mAP를 3~5%정도 개선시켜준다. 

**< Appendix C : Bounding-box regression >**

SVM 이후에, class-specific bounding box regressor을 이용하여 새로운 bounding box를 예측한다.
여기서 N training pairs $$$\{(P^i,G^i)\}_{i=1,...,N}$$$이고,
$$$P^i = (P_x^i,P_y^i,P_w^i,P_h^i)$$$로 중앙점 좌표와 width, height를 나타낸다. Ground-truth bounding box는 다음과 같이 표현된다.
$$$G^i = (G_x^i,G_y^i,G_w^i,G_h^i)$$$. 목표는 P를 ground-truth box G에 매핑하는 변형을 학습하는 것이다.
변형을 다음의 $$$d_x(P),d_y(P),d_w(P),d_h(P)$$$ 파라미터화 했다. 처음 두 개는 P의 bounding box의 중심의 scale-invariant 변환을 말하고, 뒤의 두 개는 P의 bounding box의 폭과 높이에 대한 로그 공간 변환을 말한다. 그리고 아래의 식을 이용하여 input proposal P를 변환하여 ground-truth box $$$\hat{G}$$$를 예측할 수 있다.


$$$\hat{G_x} = P_wd_x(P)+P_x ....(1)$$$
$$$\hat{G_y} = P_hd_y(P)+P_y ....(2)$$$
$$$\hat{G_w} = P_wexp(d_w(P)) ....(3)$$$
$$$\hat{G_h} = P_hexp(d_h(P)) ....(4)$$$

$$$d_*(P) = w_*^T \phi(P)$$$에서 $$$\phi(P)$$$는 P(proposal)의 feature vector이고, $$$ w_*$$$ 는 각 함수 $$$ d_*(P)$$$에 대한 ridge regression를 통해 학습되어지는 가중치 계수이다. 

$$w_* = argmin_{\hat{w}_*} \sum_i^N(t_*^i - w_*^T \phi_5(P^i))^2 + \lambda||\hat{w}_*||^2$$

L2 loss 형태에 regularization이 추가된 모습이다. 

regression target $$$ t_*$$$는 다음과 같이 정의된다.

$$t_x = (G_x - P_x)/P_w$$

$$t_y = (G_y - P_y)/P_y$$

$$t_w = log(G_w/P_w)$$

$$t_h = log(G_h/P_h)$$

regularization은 중요하다. 만약에 P가 ground-truth boxes와 멀리 있다면 P를 G로 변환하는 것은 말이 안된다. 이는 학습 실패로 이어질 것이다. 

그리고 test-time시에 각 proposal과 predict를 딱 한 번 scoring 한다. iterating이 결과를 증진시키지 않는다는 것을 알았기 때문이다. 

### ==3.6. Qualitative results ==
그냥 더 추가적인 detected 이미지들을 보여주고 있다. 
## ==4. The ILSVRC2013 detection dataset ==

### ==4.1. Dataset overview ==

val과 test 셋은 annotated되어 있어야한다. 200개의 classes중 하나와 BB가 함께 labeled 되어있어야 한다.
반면에 train images는 annotated 되어있지 않아야 한다. 
### ==4.2. Region proposals ==
selective search 결과 이미지당 2403개의 region proposals가 생성되었다. 
### ==4.3. Training data ==
Training data는 R-CNN에서 3가지 부분에 요구된다. 

(1) CNN fine-tuning

(2) detector SVM training

(3) bounding-box regressor training

### ==4.4. Validation and evaluation ==
val1 + train 1k로 fine-tuned 된 CNN을 사용하여 fine-tuning과 feature computation이 재실행되는 것을 피한다.

### ==4.5. Ablation study ==
Table.4
training data, fine-tuning, and bounding-box regression의 양이 다름에 따른 결과를 보여준다. 
### ==4.6. Relationship to OverFeat ==
OverFeat은 R-CNN보다 속도가 더 빠르긴 하다. sliding window가 warped되지 않았기 때문에 속도가 더 빠르다. 
## ==5. Semantic segmentation ==

##### ==CNN features for segmentation. ==

Table.5

* full : region의 shape을 무시하고 CNN features를 warped window에 바로 연산한다.(detection에 했던 것 처럼 )

* fg : regiond의 가장 앞쪽 mask에만 CNN features를 연산한다. 이 때, 배경을 평균으로 바꾸는데, 그래야 나중에 평균을 뺐을 때, 베경이 0이 된다. 

* full + fg : 간단하게 두 가지를 이은 것이다. 


##### ==Results on VOC 2011. ==
Table.5를 보면 fc6이 항상 fc7보다 성능이 좋다. 

Table.6

이전의 방법론 보다는 full+fg R-CNN fc6이 성능이 더 좋다. 

## ==6. Conclusion ==

R-CNN의 방법론은 성능을 상당히 향상시켰다. Region proposal을 이용한 상향식 방법과 학습 데이터가 부족해도 큰 CNN을 학습할 수 있다는 것을 통해 성능을 증진시켰다. 그리고 supervised pre-training and domain-specific fine-tuning이 성능을 높혔다고 위 논문에서는 말한다!

Region proposal + pre-training + fine-tuning  이 세 가지로 이루어낸 쾌거라고 생각한다. 이후에 등장하는 fast R-CNN, faster R-CNN보다는 시간도 오래걸리고 성능도 낮지만, 이러한 아이디어를 먼저 생각해냈다는 것에 굉장하다고 생각한다. 