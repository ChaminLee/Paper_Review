# **< Deep Residual Learning for Image Recognition > **
## *"Residual(잔차)의 반란"*

## ==0. Abstract==

Resnet은 Deep neural networks 일수록 더 학습하기 어렵다는 문제를 인식하고 이를 해결하기위해 만들어졌다. 그래서 이전의 학습 방법들과 달리 residual(잔차)을 학습하는 방법으로 더 깊은 networks를 이전보다 더 쉽게 학습시키도록 만들었다. 

residual networks는 더 optimize하기 쉽고, 더 깊은 모델로 부터 쉽게 accuracy를 얻을 수 있다는 것을 증명해냈다. 

VGG nets 보다 8배 깊은 152개의 layers를 가진 residual nets로 ImageNet을 학습하고 평가하여 3.57%라는 매우 작은 error를 보이며 ILSVRC 2015의 왕좌에 올랐다. 

이외에도, CIFAR-10, COCO의 detection, segmentation등에 좋은 성능을 보여 매우 성능이 좋은 nets라는 것을 알 수 있다.  

## ==1. Introduction==

DCNN(Deep Convolutional Neural Networks)은 image classification 분야에 있어서 획기적인 발전을 이끌었다.
최근까지의 network들은 depth가 중요한 요소로 여겨졌고, 더 깊은 모델일수록 더 좋은 성능이 나온다고 보여줬었다. 

사진training graph
( 왼쪽은 Training error& 오른쪽은 Test error이고
각각 layer가 20,56이다. )

이에 따라 depth에 대해 " Model의 depth를 늘리는 것 만으로 좋은 성능을 도출해낼 수 있을까?" 라는 궁금증이 발현됐다. 하지만 Gradient Vanishing / exploding의 문제때문에 depth가 증가할지라도 어느정도 선에 다다르면 성능이 떨어지는 모습을 보였다. 이러한 현상을 "degradation"이라고 하는데, 

이는 다시 말하면 depth가 증가함에 따라 accuracy가 포화되어( 더는 증가 X ) degrade가 점점 빨라진다는 것이다. 이러한 degradation 문제는 Overfitting이 원인이 아니라, Model의 depth가 깊어짐에 따라 training-error가 높아진다는 것이다. 

더해진 layers는 identity mapping 된 것이고, 다른 layers는 조금 얕은 모델로부터 학습되어 복제된 것이다. 이러한 모델 Solution은 deep한 모델일지라도 shallow한 모델보다 높은 training error를 가질 필요가 없다는 것을 보여준다. 

사진 Model

기존의 Model을 추구하기보다는 residual mapping을 통한 것이 degradation의 문제를 해결할 수 있다고 설명하고 있다. 

기존의 networks를 H(x) 라고 했을 때,
$$F(x) = H(x) - x$$
이를 위 식에 mapping 시킨다.

$$F(x) = H(x) - x$$
를  $$$H(x)$$$ 

에 근사시키는데, 이러한 과정(Residual mapping)을 거치는 것이 이전의 Model보다 optimize하기 더 쉽다고 말한다. 
($$$ F(x)$$$

는 잔차에 해당하고 x는 기존 input을 의미한다, 

이에 기반해 이를 feedforward neural networks에 적용한 것은 "Shortcut connections"라고 한다. Short connection은 한 개 이상의 layers를 skipping하는 것을 의미한다. 또한 Shortcut connection는 direct mapping을 하는 대신에 identity mapping을 한다. 다음 사진을 보면 이해할 수 있다. Plain layers의 경우가 direct mapping을 의미한다. 

cs231n 사진 

Identity shortcut connection은 extra parameter를 추가하는 것도 아니고, computational complexity를 추가하는 것도 아니다. 그 전체 network는 SGD와 backpropagation을 통해 학습될 수 있다. 

ResNet은 ImageNet을 통해 degradation문제와 모델 평가에 대해 검증하였다. 먼저 첫 번째로, ResNet의 deep한 model은 optimize하기 쉽지만, 그 반대인 plain networks는 depth가 깊어짐에 따라 높은 training error을 보였다. 그리고 두 번째로, deep한 residual networks는 depth가 깊어져도 이전의 plain networks 보다 쉽게 accuracy를 도출해낼 수있다. 이러한 비슷한 현상들이 다른 Datasets에도 나타났다. 그래서 extremely deep해도 recognition tasks에 있어서 훌륭한 generalization performance를 보인다!
 그리고  다른 vision 문제나 non-vision 문제에도 적용될 수 있을거라고 예상하고 있다. 


## ==2. Related work==

Vector 양자화에 있어서 residual vectors를 encoding하는 것이 기존의 vectors를 encoding하는 것 보다 더 효과적이라고 말한다. 그리고 이전에는 Multigrid method가 사용되었었다. Multigrid 는 이산화 계층을 사용하여 미분 방정식을 풀기 위한 알고리즘이다. 

그리고 ResNet의 정리는 항상 identity shortcuts는 닫혀있지 않고, 계속 정보가 흐른다

Related work의 내용은 그냥 참고로 읽어보면 된다. 

## ==3. Deep Residual Learning==

#### ==3.1 Residual Leraning==

앞서 말한 것과 같이 H(x)를 기존의 Networks라고 보자. 한 가정은 여러 비선형 layers가 복잡한 함수에 근사할 때, F(x)+x를 H(x)에 근사시키는 것이 더 낫다고 말한다. 

이러한 reformulation은 degradation에 대한 이전의 상식에 반하는 현상이다. Introduction에서 layers가 추가될수록, 더 deep한 모델일수록 training error가 shallow한 모델보다 더 낮은 error을 가질 수 없다는 상식이다. 

#### ==3.2 Identity Mapping by Shortcuts==

ResNet은 아무리 적게 쌓인 layers일지라도 residual learning을 적용한다. Fig.2에서 보인 block을 다음과 같이 정의한다. 

$$ y=F(x,{W_i}) + x \ \ \ \ \ \ \ \ \ \ (1)$$

여기서 x는 input, y는 output vectors이다. $$$F(x,{W_i})$$$

는 학습될 residual mapping을 의미한다. 

Fig.2의 그림을 수식으로 나타내면 다음과 같다.

$$
F = W_2\sigma(W_1x)
$$

여기서 σ 는 ReLU를 의미하고 biases는 notation을 간략화하기 위해 생략했다. 

앞서 나왔던 내용중에 extra parameter와 computational complexity가 추가적으로 필요없다는 것이 기존의 plain networks와 residual networks를 비교하는데 매우 중요하고 매력적인 요소가 될 것이다. 

위의 수식의 경우에서 $$$x \ 와 \ F$$$  

는 차원이 같아야한다. 만약 그렇지 못할 경우 (input/output의 channel이 바뀔경우) linear projection $$$W_s$$$

를 통해 차원을 맞춰줄 수 있다. 

$$
y = F(x,{W_i}) + W_sx \ \ \ \ \ \ \ (2)
$$

이어서 나올 Experiments를 통해 identity mapping이 degradation 문제와 효율성의 문제를 충분히 해결할 수 있다는 것을 보여줄 것이고, W_s는 오직 차원을 맞추는데만 사용된다.

만약에 F가 Single layer라면 (1)번식은 linear layers와 유사하다. 

$$
F(x,{W_i})
$$

는 multiple convolutional layers를 표현할 수 있다. 

#### ==3.3 Network Architectures==

Plain network와 residual network를 비교하기 위해 ImageNet의 data를 이용해보자. 

layers사진 

1. Plain Network

Plain Network는 그림의 가운데에 위치한 그냥 layers를 쌓기만한 그림이다. 

Conv layers는 대부분 3X3 filters를 가지고 있고, 다음의 간단한 규칙을 따른다. 

(1) 각 layers는 같은 크기의 output feature mapd을 가지고, 같은 수의 filters를 갖는다. 

(2) 만약에 feature map size가 반으로 줄어들었다면, time-complexity를 유지하기 위해 filters의 수는 두 배가 된다. 

downsampling을 하기 위해서 stride가 2인 conv layers를 통과시킨다. 그리고 Network는 global average pooling과 1000-way-fully-connected layer with sofrmax로 끝난다. 

이 모델은 VGG nets보다 filters의 수가 적고 lower complexity를 가진다는 것이 장점이다. 

2. Residual Network

위의 Plain network에 기반하여, shortcut connections를 추가한 residual version ResNet이다. input과 output의 차원이 같다면 identity shortcuts는 바로 사용될 수 있다. 차원이 증가한다면 두 가지 경우를 생각해야한다. 

(A) shortcut은 계속 identity mapping을 하고, 차원을 증가시키기위해 zero-padding을 해야한다. 이 경우가 왜 Extra parameter가 필요없는지 설명해준다. 

(B) 식 (2)에서 projection shortcut을 할 때 차원을 맞추기 위해 1X1 conv를 사용한다. 

위 두 가지 options는 shortcuts이 feature map을 2 size씩 건너뛰기 때문에 stride를 2로 사용한다.


#### ==3.4 Implementation==

ImageNet data에 적용시켜봤다. 

image는 [256,480]사이에서 scale을 증대시키기위해 randomly하게 sampled되었다. 

이미지 각각에 평균 픽셀값을 빼주는 Per-pixel mean substracted와 함께, 224X224 crop이 이미지에 대하여 ramdomly smapled되었다. 

standard color augmentation이 사용되었고, 각각의 convolution 다음과 activation 이전에 Batch normalization이 적용되었다. 

mini batch size - 256과 함께 SGD를 이용했다.

learning rate는 0.1에서 시작해서 error가 local minimum에 빠질 때마다 1/10씩 나눠줬다. 

model은 $$$60 \times 10^4$$$

만큼 iteration을 돌았다. 

weight decay = 0.0001, momentum = 0.9를 사용했고, dropout을 쓰지 않았다. 

test 시에는 10-crop testing을 적용하였다. 

10-crop이란 image에 대해 총 10번 잘라서 보는 것인데, 각 모서리에 대해 총 8번 중앙부에 2번 진행한다.

10-crop 사진
(큰 모서리 4번 + 작은 모서리 4번 + 작은 중앙 + 큰 중앙 )

그리고 최적의 결과에 fully convolutional을 적용했고, multiple scale에 대해서 점수를 평균화했다. 

## ==4. Experiments==

#### ==4.1 ImageNet Classification==

1,000 개의 class를 가진 ImageNet 2012 classification dataset 을 사용하였다. 128장의 training images를 사용하고, 평가를위해 5만장의 validation images를 사용하였다. 그리고 최종 결과를 내기위해 10만장의 test images를 사용하여 top 1 ~ 5의 error rate를 평가하였다. 

###### ==4.1.1 Plain Networks==

처음에는 18, 34 - layer를 가지는 plain network로 평가했다. 

table2의 표를 보면 34-layer의 validation error가 18-layer의 error보다 높게 나왔다. 여기서 degradation의 문제를 관측했다. Figure4를 보면 layer가 더많은 plain-34가 더 높은 Training error를 보이는 것을 알 수 있다. 

이러한 문제에 대해 optimization problem이 vanishing gradients때문에 일어났을 것이라고 생각하지는 않았다. 왜냐하면 BN을 통해 학습되었고 backward, forward 모두 문제가 없었기 때문이다. 그 말은 즉슨, 아직 34-layer plain ent은 경쟁력이 있다는 소리이다. 추측으로는 deep plain nets가 기하급수적으로 낮은 수렴률을 가지고 있어서 training error를 줄이는 영향을 준다는 것이다. 아마 이 이유에 대해서는 추후에 더욱 더 연구될 것이다. 

###### ==4.1.2 Residual Networks==

그리고 18, 34 - layer를 가지는 residual network를 평가했다. Table2와 Figure4를 보면서 비교해보면, 먼저 layer가 깊어질수록 error가 2.8%나 줄어들었다. 그리고 더 중요한 것은 Figure4에서 training error가 줄어들고 있고, valdiation data에 대해 더 일반화를 잘하고 있다는 것을 알 수 있다. 이로써 degradation 문제를 잘 해결했다고 말할 수 있다. 그리고 34-layer plain net에 비교하여 34-layer ResNet은 3.5%나 더 낮은 error를 보인다(trainind error 감소 또한 성공적). 이 비교는 deep한 system의 residual을 학습하는 것이 더 효율적이라는 것을 증명해냈다. 마지막으로 Figure4를 보면 ResNet이 더 빠르게 optimization을 하는 모습 (빠르게 수렴)을 볼 수 있다. 

###### ==4.1.3 Identity vs Projection Shortcuts==

앞서봤지만, identity shortcuts는 training에 도움이 된다는 것을 확인했다. Table3를 통해 우리는 3가지 options를 비교할 수 있다. 

(A) zero-padding shortcuts가 차원을 증대시키기위해 사용되었고, 모든 shortcuts는 parameter-free하다.(위의 가장 기존 ResNet모델)

(B) projection shortcuts는 차원을 증대시키는데에 사용되었고, 다른 shortcuts는 identity하다.

(C) 모든 shortcuts는 projections이다. 

Table3 사진

Table3을 보면 3가지 옵션이 적어도 plain nets보다는 낫다는 것을 알 수 있다. 

(B)가 (A)보다 조금 나은데 이는 (A)의 zero-padding과정에 residual learning이 없기 때문이라고 말한다. 

(C)도 (B)보다 조금 좋은 성능을 보이는데, 이는 extra parameters가 많은 projection shortcuts에 의해 설명된 것이 성능에 기인했다고 말한다. 

(A),(B),(C)의 차이가 작은걸 보면 projection shortcuts는 degradation 문제를 해결하는데는 필수적이진 않다는 것을 알 수 있다. 그래서 memory를 덜 잡아먹기위해서 (C)는 앞으로 사용하지 않는다. 

Identity shortcuts는 아래 설명될 Bottleneck Architectures의 복잡성을 증가시키지 않는데 특히 중요하다.

###### ==4.1.4 Deeper Bottleneck Architectures==

더 deep한 모델을 만들기 위해 2가 아닌 3-layers를 쌓는다.

figure5사진

3-layers는 1X1 , 3X3, 1X1의 convolution이고, 1X1 layers는 차원을 감소하고 증가(복원)시켜, 더 작은 input/output 차원으로 Bottleneck을 일으킨다.

Figure.5를 보면 두 개의 time complexity도 비슷하다. 
앞서 말한 것 처럼 identity shortcut는 time complexity와 model size를 줄이는데 중요하다. 이 과정이 projection으로 바뀌게 되면 time complexity와 model size가 배가 된다. 따라서 identity shortcut이 bottleneck designs를 더 효율적으로 만들어준다. 

###### ==4.1.5 50-layers ResNet==

3-layer bottleneck block을 Table1의 50-layers, option은 (B)를 택해서 차원을 증가시켰다. 

###### ==4.1.6 101-layers and 152-layers ResNet==

위와 같은 option에서 depth만 증가시켰다. 확실이 기존의 34-layers보다 상당한 차이가지며 좋은 성능을 보였다. degradation의 문제는 찾지 못하고 깊이가 더 깊어짐에 따라 더 좋아지는 것을 목격했다. 

###### ==4.1.7 Comparisons with State-of-the-art Methods==

table5사진

ResNet을 포함한 Table.5의 모델들을 ensemble하여 3.57%의 error를 도출해내며 ILSVRC 2015의 왕좌에 올랐다. 


#### ==4.2 CIFAR-10 and Analysis==

CIFAR-10에서도 위와 마찬가지로 학습을 진행하였다. 초반부는 겹치는 내용이 많기에 그냥 읽어보면 된다. 

######== 4.2.1 Analysis of Layer Responses==

figure7사진

ResNets은 plain nets에 비해 일반적으로 더 적은 반응을 보였다. 이 결과는 즉, 앞선 기본 가정이었던 residual functions가 non-residual functions 보다 일반적으로 더 0에 가까워질 것이라는 것을 지지해준다. 

######== 4.2.2 Exploring Over 1000 layers==

table6사진

figure6 사진

1202-layer network는 110-layer network와 유사한 training error 보였지만, 실제 성능은 더 좋지 못했다. 이는 Overfitting의 문제로 보인다. dataset의 크기에 비해 layer가 불필요하게 많다는 것이 그 이유이다. 

만약에 maxout, dropout같은 regularization을 쓴다면 성능이 향상되겠지만 이 paper에서는 dropout을 사용하지 않기 때문에, 이는 추후에 알아보자.

#### ==4.3 Object Detection on PASCAL and MS COCO==

table7,8 사진

마찬가지로 recognition tasks에서도 빛을 발했다. COCO datasets에서 6.0% 더 높은 수치를 기록하였다, 이는 28%의 상대적인 발전이었다. 

## ==5. 마무리==

이로써 ResNet의 Review가 끝이 났다. 수학적으로는 어렵진 않았지만 방법론적으로 획기적이었던 것 같다. 잔차를 평가의 기준으로 보기만 했었지, 학습할거라고는 아무도 생각하지 않았을 것이다. 논문을 보면서도 계속 생각이 들었지만, "잔차의 반란" 이라는 말이 어울리는 것 같다. 