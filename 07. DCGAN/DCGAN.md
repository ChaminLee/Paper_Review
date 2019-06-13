# < UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL  GENERATIVE ADVERSARIAL NETWORKS >

## *"상상력이 탄생시킨 DCGAN"*

## ==0. Abstract==
Computer Vision 분야에서 CNN을 이용한 지도 학습 방법은 많이 채택되어왔다. 하지만 비교적 비지도 학습 CNN은 주목받지 못했다. 이 논문에서는 지도 학습과 비지도 학습간의 성과 차이를 줄이기 위해 노력할 것 이다. 
이를 DCGAN이라고 부를 것이다. 
 
## ==1. Introduction==

* DCGAN이 대부분의 세팅에서 안정적으로 학습하는지 보고, 평가할 것이다. 
* 다른 비지도 학습 방법들과 비교하기 위해서, 이미지 분류 과제에 학습되었던 식별자(D)를 이용할 것이다.
* GAN에 의해 학습된 필터들을 시각화 할 것이고 경험에 의거하여, 특정한 물체를 그리도록 학습된 특정한 필터를 보여줄 것이다. 
* 생성자(G)가 벡터 연산적인 특성이 있어서 쉽게 조작하여 다양한 의미의 샘플을 생성할 수 있음을 보여줄 것이다.

## ==2. Related Work==

### ==2.1 Representation Learning From Unlabeled Data==

* Clustering (ex. K-means)
* Auto-encoder
* Ladder structures

### ==2.2 Generating Natural Images==

이미지 생성모델은 잘 연구되었고, 두 가지 범주로 구분된다 : Parametric & Non-Parametric

< Non-Parametric >
이 모델은 기존 이미지의 데이터 베이스에서 일치하며, 종종 이미지 조각과 일치한다. 그리고 다음과 같은 기술에 사용되었다.

* Texture Synthesis
* Super-Resolution
* In-Painting

< Parametric >
이미지 생성을 위한 parametric 모델들은 광범위한 분야에 대해 연구되었다. 하지만 아직까지 실제 세계의 자연스러운 이미지를 생성해내는 것은 최근까지도 성공적이지 못하다. 조금의 성공이 있었어도 흐릿함이 큰 문제로 발생했다. GAN에 의해 생성된 이미지는 노이즈와 이해불가한 문제에 대한 단점이 있다. 이를 Laplacian Pyramid를 이용하여 고품질의 이미지를 보여줬지만, 노이즈 때문에 주견이 없는 것이 문제이다. 최근에 deconvolution으로 자연스러운 이미지를 생성해냈지만 이는 supervised taks generators에게는 효력이 없었다.

### ==2.3 Visualizing The Internals Of CNNs==

CNN은 블랙박스 방법이라고 끊임없이 비판당한다. 하지만 deconvolution을 사용하고 maximal activations를 필터링함으로써, network에서 각 convolution 필터의 대략적인 목적을 찾을 수 있다는 것을 보여주었다. 마찬가지로 입력에 gradient descent를 사용하면 특정 필터 하위세트를 활성화하는 이상적인 이미지를 검사할 수 있다.  

## ==3. Approach And Model Architecture==
고해상도+ 더 깊은 생성 모델을 훈련시키기 위한 다음의 구조를 생각해냈다.

1) Pooling Layers 대신에 strided convolutions(downsample) & fractional-strided convolutions(upsampling)로 대체한다.
2) Conv features의 맨 위의 FC layers를 제거한다. 대신에 Global average pooling으로 모델 안정성을 향상시킨다.(수렴속도는 하락)
3) Batch Normalization으로 inputs를 normalize한다. 이는 깊은 모델의 훈련에서 발생하는 초기화 문제와 gradient 흐름 문제에 대해 도움을 준다. 하지만 모든 layers에 대해 BN을 적용하면 샘플이 변동하고, 모델이 불안정해진다. 그래서 generator의 output layer(이미지)와 discriminator의 input layer(이미지)에는 BN을 적용하지 않는다.
4) Generator에는 ReLU를 사용 ( output은 Tanh ) 
경계가 있는 activation이 모델이 학습을 더 빨리 수렴하도록 학습하고 학습 분포의 색공간을 빠르게 커버하도록 하기 때문이다.
Discriminator에는 LeakyReLU 사용.

## ==4. Details Of Adversarial Tranining==

Fig.1 사진
Fully connected / pooling layers는 사용하지 않았다. 

* Mini-batch SGD ( batch_size = 128 )
* 정규분포(평균=0/표준편차=0.02)로 가중치 초기화
* AdamOptimizer ( lr = 0.001 -> 0.0002로 내림 / momentum = 0.9 -> 0.5로 내림(학습 안정화) )

### ==4.1 LSUN==
침실 사진이 모여있는 데이터셋이다. 이미지 모델들을 통해 생성된 샘플들의 품질은 좋았지만 Overfitting과 단순암기의 학습측면의 문제에 대해서는 문제가 제기된다. 

Fig.2 
Epoch = 1

Fig.3
수렴 이후의 샘플 ( Epoch = 5 )

Fig.3을 통해 이 모델이 단순히 overfitting/memorizing에 의해서 훈련 샘플들을 통해 고품질의 샘플들을 생성해내는 것이 아니라는 것을 증명했다. 

##### ==4.1.1 Deduplication==
Generator가 단순 암기를 통한 샘플 생성하는 것을 방지하기 위해 간단한 image de-duplication 과정을 진행했다. Memorization 문제를 피하기 위해, 이미지들을 autoencoder를 통해 코드로 변환하도록 하고 이 코드와 가까운 값들을 제거한다. 이러한 과정의 결과로 precision은 높아지고 FP(false positive) 비율은 0.01 이하로 나타났다. 추가적으로 275,000 개의 중복된 이미지들을 제거했다. (recall 상승)
### ==4.2 Faces==
1만명의 사람들의 3백만개의 얼굴 이미지로 구성되어있다. 
### ==4.3 ImageNet-1K==
비지도 학습을 위해 자연 이미지 소스를 사용하였다.

## ==5. Empirical Validation Of DCGANs Capabilities==

### ==5.1 Classifying CIFAR-10 Using GANs as a Feature Extractor==

Table.1
DCGAN은 CIFAR-10을 기반으로 fine-tuned 되지 않고, ImageNet-1K로 fine-tuned 됐다.

아직 DCGAN의 성능이 기본 CNN에 미치지 못하지만, 더 발전할 수 있는 부분은 이후의 일로 남겨둔다. ( Discriminator를 fine-tune하는등 )


### ==5.2 Classifying SVHN Using GANs as a Feature Extractor==

DCGAN에서 사용되는 CNN 구조가 동일한 데이터에 대해 동일한 구조를 가진 순수하게 지도된 CNN을 학습하고 64개 이상의 하이퍼파라미터 시도를 무작위 검색을 통해 이 모델을 최적화함으로써 모델 성능의 주요 기여 요소가 아님을 검증한다.

## ==6. Investigating And Visualizing The Internals Of The Networks==

Table.2

Error rate가 가장 낮은 것이 DCGAN임을 알 수 있다.

### ==6.1 Walking In The Latent Space==

Latent space(Z)에서 이동할 때, 급작스럽지 않고 천천히 부드럽게 변화해야한다. 급작스러운 변화가 일어났을 경우에는 Memorization이라고 판단한다. 반면에 부드럽게 변화가 일어나는 경우 이미지 생성에 대해 의미적인 변화를 줄 수 있다.( 객체를 추가하거나 제거하거나 ) 이는 모델이 관련이 있고, 흥미있는 부분에 대해 학습했다고 생각하기 때문이다. Fig.4 에서 설명된다. 

Fig.4
맨 윗줄을 보면 천천히 부드럽게 이미지가 변하는 것을 볼 수 있다. 6번째 줄에서는 없던 커다란 창문이 생기는 것을 볼 수 있고, 맨 마지막 줄에서는 TV가 창문으로 변하는 것을 볼 수 있다. 

### ==6.2 Visualizing The Discriminator Features==

이 논문은 DCGAN을 대규모 데이터셋에 대한 비지도 학습을 통해 흥미 있는 특성들을 위계질서있게 학습할 수 있다는 것을 증명했다.  

Fig.5
오른쪽은 지도된 backpropagation을 시각화 한 것이다. 위 사진을 통해 discriminator가 어떠한 부분을 활성화하여 인식하고 있는지를 보여준다. 

### ==6.3 Manipulating The Generator Representation==

##### ==6.3.1 Forgetting To Draw Certain Objects==

샘플의 품질은 generator가 주요 장면의 구성요소를 대표하는 특정한 물체인 침대, 창문, 램프, 문, 다양한 가구를 학습하는 것을 시사한다.

Fig.6
윗 줄은 수정되지 않은 샘플 / 아랫 줄은 "창문"을 제거하는 필터를 적용한 샘플이다. 실제로 창문이 사라지기도 하고 다른 물체로 대체되기도 했다. 이러한 것 처럼 다른 물체 또한 제거하고 generator가 그리도록 할 수 있다. 

##### ==6.3.2 Vector Arithmetic On Face Samples==

WordEmbedding에서 King-man+woman을 하면 "queen"이 나오는 것 처럼 이미지도 벡터 연산이 가능하다는 것을 보여줬다. Z를 이용하여 연산이 의미적으로 작용하는 것을 보여줬다.

Fig.7
각 column은, 샘플들의 Z는 평균값이다.
 
Fig.8
보간을 이용하여 얼굴을 왼쪽에서 보는 것에서 오른쪽에서 보는 것 처럼 바꾸는 것이다. 

이를 통해 네트워크가 scale, rotation, position등에 대해 이해하고 있다는 것을 알게 되었다. 

## ==7. Conclusion And Future Work==

DCGAN이 GAN의 불안정함을 해소해주긴 했지만, 오래 학습시킬 경우 oscillating mode나 collapse 문제가 발생할 수 있다는 불안정한 모습을 보이긴 한다. 이 점에 대해서는 해결하기 위한 노력을 필요로 한다.

무궁무진한 상상력 속에서 탄생한 아이디어 같다. 이미지에서 의미적인 무언가를 연산한다는 것도 신박하고, NN의 블랙박스를 어느 정도 해결했다는 것이 대단하다. 핵심으로는 CNN을 GAN에 적용했다는 것, FC layers를 없애 parameter의 수를 줄였다는 것, 단순암기(overfitting)을 막기 위해 노력한 과정등이 있는 것 같다. 논문이 나온 뒤 한참후에 읽었지만 지금 봐도 놀라운 것 같다. 
