# < LSGAN : Least Squares Generative Adversarial Networks >

## *Least Square's effect*

## ==0. Abstract==

최근 비지도 학습에 대한 GAN의 성능이 성공적임을 증명했다. 기존의 Vanilla GAN은 분류기로써 discriminator를  가정하고, sigmoid cross entropy loss function을 사용했다. 하지만 이 loss function이 학습과정에서 "Gradient Vanishing" 문제를 발생시킬 수 있다는 것을 발견했다. 이러한 문제를 해결하기 위해서는 이 논문에서 말하는 discriminator의 loss function에 least square를 채택하는 Least Square Generative Adversarial Networks를 사용해야한다. 새로운 loss function으로 인해 더 좋은 품질의 이미지를 생성할 수 있고, 학습과정도 안정적으로 취할 수 있다. 

## ==1. Introduction==

기존의 GAN은 비지도 학습에서도 성공적이었다. 이미지 생성에도 큰 성공을 거두었지만, GAN이 생성한 이미지의 질은 아직 실제 이미지에 비해서는 덜하다. 원인으로는 기존에 loss function으로 사용하는 Sigmoid cross entropy loss가 gradient vanishing을 불러일으키기 때문에 학습이 제대로 되지 않는다는 것이다. 

하지만 LSGAN에서 말하는 least square를 loss function에 적용시킨다면, 이 문제를 해결할 수 있다. least square는 가짜 샘플들을 결정 경계(decision boundary)에 가깝게 이동시켜준다, 왜냐하면 least square loss function은 알맞게 분류되었지만 결정 경계에서 멀리 떨어진 샘플들에 대해 패널티를 부과하기 때문이다. Fig.1(c)에서 볼 수 있듯이 가짜 샘플들에 대해 패널티를 부과하고, 샘플들이 제대로 분류되었어도 결정 경계쪽으로 이동시킨다. 이러한 특성에 의해서, LSGANs는 실제와 더 가까운 샘플을 생성할 수 있게 되는 것이다. 

Fig.1사진  색깔 표시해야함
(a) 두 loss functions의 decision boundaries 이다. 결정 경계가 실제 데이터 분포를 가로질러야 성공적으로 GAN이 학습했다고 볼 수 있다. 아닐 경우에는 학습이 더딘것이다. 

(b) Sigmoid crosss entropy loss function의 결정 경계이다. G를 업데이트 하기 위한 가짜 샘플들이 올바르 경계안에 있기 때문에 에러가 작다.

(c) Least Square loss function의 결정 경계이다. 가짜 샘플들에 대해 패널티를 부과함으로써, generator가 결정 경계쪽으로 샘플을 생성하도록 힘을 가한다. 


LSGAN의 또 다른 이점으로는 학습 과정의 안정성을 증진시켜준다는 것이다. 앞선 GAN에서는 학습이 불안정하다는 것이 어려운 점 이었다. 이러한 불안정성은 GAN의 목적 함수에 원인이 있다고 말한다. 명확하게 말하면, Vanilla GAN의 목적함수를 최소화하는 것은 vanishing gradients의 문제로부터 시달리고, 이 때문에 generator를 업데이트 하기가 힘든 것이다. 계속 말하지만 이를 least square를 적용하여 문제를 완화한 것이다. 최근에는 Batch normalization을 제거하고 GAN의 안정성을 비교하는 방법들이 제안됐었다. 이러한 제의를 통해, LSGAN 또한 batch normalization 없이 비교적 좋은 상태로 수렴할 수 있다는 것을 발견했다. 

* Discriminator에 대해 least square loss funtion을 사용하는 LSGAN을 제안한다. LSGAN의 목적함수를 최고화하는 것은 결국 Pearson $\mathcal{X}^2$ divergence를 최소화하는 것이라는 것을 보여준다. 이러한 결과는 vanilla(regular) GAN보다 LSGAN이 더 실제같은 이미지를 생성할 수 있다는 것을 증명하는 실험 결과이다. 또한 LSGAN의 안정성을 증명하기 위해 여러 비교 실험들이 진행됐었다.

* LSGAN의 두 가지 네트워크 구조가 설계되었다. 첫 번째로는 112X112 해상도의 이미지를 생성하는 것인데, 다양한 종류의 데이터셋(침대,교회...)으로 평가되었다. 이 구조의 LSGAN이 다른 최신 방법들 보다 더 좋은 이미지를 생성했다는 것을 보여주었다. 두 번째는 클래스가 매우 많은 데이터에 대해 진행되었다. 3470개의 클래스가 있는 중국어 손글씨 데이터셋으로 평가했고, 읽을 수 있는 글자를 생성할 수 있다는 것을 보여주었다.  

## ==2.Related Work==

기존에 RBMs, DBNs, DBMs등을 기반으로 사용하였지만 이들은 모델을 학습시키기 위한 근사법인 분배함수나 사후 분포를 계산하기 어려운 문제가 있다. VAE같은 경우도 많이 사용되지만, variational lower bound를 최대화하여 학습하기 때문에, 생성된 이미지가 흐릿한 문제를 발생시킬 수 있다. 

위의 방법들과 비교하였을 때, GAN을 학습하는데에는 근사법(approximation method)을 필요로 하지 않는다. VAEs 처럼, GANs 또한 미분가능한 네트워크를 통해 훈련될 수 있다.

비지도 학습에 대해 GAN이 다양한 tasks에 대해 적용되고 있다. Image-generation / Image super-resolution / text to image synthesis / image to image translation등등 다양한 분야에 적용되어서 좋은 성능을 보이고 있다. 
 GAN의 큰 성공에도 불구하고, 생성된 이미지의 질을 향상시키는 것은 여전히 도전적인 문제이다. 이를 해결하기 위해서 DCGANs( Conv layer 첫 도입), LAPGANs( Laplacian pyramid 사용) 과 같은 방법론이 등장했다. 

Fig.2 사진 
(a) Sigmoid cross entropy loss function
(b) least square loss function

생성된 샘플을 실제 데이터 통계와 매칭하기 위해  discriminator의 중간 layer에서 MSE(Mean Squared Error)를 최소화한다.

이외에도 distance 측정에 문제가 있다는 WGAN, 실제 샘플 Loss보다 가짜 샘플 Loss가 더 커야 vanishing gradient를 막을 수 있다는 Loss-sensitive GAN등이 출현하였다. 

## ==3. Method==

LSGANs의 공식 / 이점 / 네트워크 구조 순으로 진행한다.

### ==3.1 Generative Adversarial Networks==

Regular GAN은  discriminator $D$, generator $G$가 동시에 학습한다. $G$는 데이터 $x$에 대한 분포 $p_g$를 목표로 학습하도록 하고, $G$는 균등 혹은 정규분포 $p_z(z)$로 부터 온 $z$를 샘플링하는 것부터 시작하여, 미분가능한 네티워크를 통해 입력변수 $z$를 데이터 공간  $G(z;\theta_g)$에 매핑한다.  반면에 $D$는 분류기$D(x;\theta_d)$로 이미지가 G에서 온 것인지 실제 데이터에서 온 것인지를 인식한다. Minimax 식은 다음과 같다.

$$\displaystyle\min_G \displaystyle\max_D V_{GAN}(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log 1-D(G(z))]$$

### ==3.2 Least Squares Generative Adversarial Networks==

Generator를 업데이트 할 때, loss funtion은 실제 데이터로부터는 멀지만 알맞는 결정 경계에 위치한 샘플들에 대해 vanishing gradients 문제를 야기할 수 있다. 이 문제를 해결하기 위해, LSGAN을 사용하는 것이다. 여기서 a-b coding을 사용한다. 

* $a$ : 가짜 데이터 라벨
* $b$ : 실제 데이터 라벨
* $c$ : G 입장에서 D가 가짜 데이터를 보고 진짜라고 믿기를 원하는 값

LSGAN의 목적 함수는 다음과 같다.

$\displaystyle\min_D V_{LSGAN}(D)=\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x)-b)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z))-a)^2]$

$\displaystyle\min_{G} V_{LSGAN}(G)=\frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z))-c)^2]$

#### ==3.2.1 Benefits of LSGANs==

이점은 총 두 가지로 볼 수 있다.

* (1) 더 실제와 같은 이미지를 생성할 수 있다.

	Fig.1$(b)$에서 결정 경계 안에 있지만 멀리 떨어져있는 샘플들에 대해 Fig.1 $(c)$ 처럼 패널티를 주어 이동시켜 더 실제와 같은 이미지를 생성할 수 있는 것이다. 

* (2) 학습을 안정화 할 수 있다.
	
	결정 경계로부터 멀리 떨어져있는 샘플들에게 패널티를 부여함으로써 gradients를 더 생성하여 gradient vanishing을 완화한다. 그러므로 학습이 안정화되는 것이다. 


#### ==3.2.2 Relation to f-divergence==

Original GAN 논문에서, 저자는 목적 함수를 최소화하는 것이 결국 JS divergence를 최소화하는 것이라는 것을 보여주었다. 

$$C(G) = KL\Big(p_{data}||\frac{p_{data}+p_g}{2}\Big)+KL\Big(p_{g}||\frac{p_{data}+p_g}{2}\Big) - \log4$$

앞으로의 내용은 LSGAN과 f-divergence의 관계에 대해 알아볼 것이다. 이전의 목적 함수를 확장한 것을 보자.

$\displaystyle\min_D V_{LSGAN}(D)=\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x)-b)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z))-a)^2]$

$\displaystyle\min_{G} V_{LSGAN}(G)=\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x)-c)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z))-c)^2]$

$\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x)-c)^2]$를 $V_{LSGAN}(G)$에 추가한 것은 어차피 최적값을 구하는데 영향을 미치지 않는다. (추가된 식에 G가 없어서)

$G$를 고정시켰을 때, discriminator $D$의 최적은 다음과 같다.

$$D^*(x) = \frac{bp_{data}(x)+ap_g(x)}{p_{data}(x)+p_g(x)}$$

다음의 식에서는 $p_{data}$를 $p_d$로 간단하게 표기한다. Generator의 식을 다음과 같이 변환할 수 있다.

$2C(G) = \mathbb{E}_{x \sim p_{d}}[\log (D^*(x)-c)^2] + \frac{1}{2}\mathbb{E}_{x \sim p_{g}}[\log (D^*(x)-c)^2]$

$= \mathbb{E}_{x \sim p_d} \Big[(\frac{bp_d(x) + ap_g(x)}{p_d(x)+p_g(x)}-c)^2] + \mathbb{E}_{x \sim p_g} \Big[(\frac{bp_d(x) + ap_g(x)}{p_d(x)+p_g(x)}-c)^2]$

$=\int_X p_d(x) (\frac{(b-c)p_d(x)+(a-c)p_g(x)}{p_d(x)+p_g(x)})^2dx+\int_x p_g(x) (\frac{(b-c)p_d(x)+(a-c)p_g(x)}{p_d(x)+p_g(x)})^2dx$

공통인자로 묶은 후 $(p_d(x) +p_g(x))$가 분모와 곱해진다.

$=\int_X (\frac{((b-c)p_d(x)+(a-c)p_g(x))^2}{p_d(x)+p_g(x)})dx$

$(b-a)$로 묶기 위해 분자에 $+bp_g(x) / -bp_g(x)$를 추가한다.

$=\int_X (\frac{((b-c)(p_d(x)+p_g(x))-(b-a)p_g(x))^2}{p_d(x)+p_g(x)})dx$

만약에 $b-c = 1, b-a=2$라고 하면 식은 다음과 같다.

$$2C(G) = \int_X\frac{(2p_g(x)-(p_d(x)+p_g(x)))^2}{p_d(x)+p_g(x)}dx$$

(제곱이니 식정리시에 순서를 바꾼 것이다.)

이는 따라서 $\mathcal{X}^2$Pearson의 식과 같아진다.

$$=\mathcal{X}^2_{Pearson}(p_d+p_g||2p_g)$$

$\mathcal{X}^2_{Pearson}$은 $\mathcal{X}^2_{Pearson}=\int_X \frac{(q-p)^2}{p}dx$과 같이 정의되는 식이다. 

따라서 만약 $b-c=1,b-a=2$일 때, $p_d+p_g$와 $2p_g$와의 $\mathcal{X}^2_{Pearson}$ divergence를 최소화하는 식이 되는 것이다. 

Fig.3
(a) Generator / (b) Discriminator

#### ==3.2.3 Parameters Selection==

앞선 식에서 $a=-1, b=1, c=0$ 으로 설정한다면 다음과 같은 목적 함수를 얻을 수 있다. 

$\displaystyle\min_D V_{LSGAN}(D)=\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x)-1)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z))+1)^2]$

$\displaystyle\min_{G} V_{LSGAN}(G)=\frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z)))^2]$

다른 방법으로는 $G$가 가능한한 실제와 같은 샘플을 만들게하기 위해 $c=b$라고 세팅한다. 0-1 binary coding을 한다면 다음과 같은 목적 함수를 얻을 수 있다. 

$\displaystyle\min_D V_{LSGAN}(D)=\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x)-1)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z)))^2]$

$\displaystyle\min_{G} V_{LSGAN}(G)=\frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z))-1)^2]$

첫 번째 식과 두 번째 식의 성능은 매우 비슷하다. 하지만 모델을 훈련 시킬 때는 아래에 위치한 식을 채택하여 사용할 것이다. (0-1 binary coding)

### ==3.3 Model Architectures==

앞선 Fig.3의 구조는 VGG 모델로부터 영감을 얻은 것이다. 

Fig.4
(a) Generator / (b) Discriminator

Discriminator의 구조는 [DCGAN (논문리뷰)](https://leechamin.tistory.com/222?category=839075) 과 유사하지만 마지막 단이 least square loss function을 썼다는 것이 다르다. DCGAN에 따라서, generator는 ReLU, discriminator는 LeakyReLU를 activation으로 각각 사용한다. 

두 번째 모델은 중국어 손글씨 데이터셋처럼, 수많은 클래스가 존재하는 일에 대한 것이다. GAN을 multiple classes에 학습시켰을 때, 읽을 수 있는 글자가 생성되지 않았다. 그 이유는 입력으로는 multiple classes가 들어가는데, 출력으로는 하나의 class만 나오기 때문이다. 입력과 출력 사이에는 결정적인 관계가 필요하다. 이 문제를 해결하기 위한 방법으로는 Conditional GAN이 있다. 라벨에 조건적인 정보를 생성함으로써 입력과 출력사이에 결정적인 관계를 만들어주기 때문이다. One-hot encoding 같은 경우는 연산량이 많아져서 memory 비용과 연산시간비용에 문제가 있다. 선형 매핑 layer를 사용해서 큰 라벨 벡터들을 작은 벡터로 먼저 매핑하고, 작은 벡터들을 모델의 layer로 연결한다. 요약해서, 모델 구조는 Fig.4와 같고, layers는 연결되어야하고, 실증적으로 결정되어야한다. 이러한 Conditional LSGAN의 목적함수는 다음과 같다. 

$\displaystyle\min_D V_{LSGAN}(D)=\frac{1}{2}\mathbb{E}_{x \sim p_{data}(x)}[\log (D(x|\Phi(y))-1)^2] + \frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z)|\Phi(y)))^2]$

$\displaystyle\min_{G} V_{LSGAN}(G)=\frac{1}{2}\mathbb{E}_{z \sim p_{z}(z)}[\log (D(G(z|\Phi(y)))-1)^2]$

$\Phi(\cdot)$는 linear mapping function을 의미하고, $y$는 label verctors를 의미한다. 

Fig.5사진

## ==4. Experiments==

LSGAN을 평가할 각종 데이터 셋과 그 세부 특징이다. 

Table.1

### ==4.1 Datasets and Implementation Details==

Fig.6

Tensorflow를 이용하여 DCGAN를 기반으로 구현했다. scene과 HWDB의 learning rate는 각각 0.001, 0.0002로 설정했다. Adam optimizer의 $\beta_1$은 0.5로 설정했다. 

### ==4.2 Scenes==

LSGAN을 LSUN 데이터셋의 종류중 5가지, 침대, 부엌, 교회, 식당, 회의실에 대해 학습했다. DCGANs, EBGANs 기본적인 방법과 LSGAN으로 침대를 생성했다.(Fig.5) LSGAN으로 생성된 이미지가 더 좋은 것을 볼 수 있다. 다른 scenes에 대한 LSGAN의 학습결과는 Fig.6에 나와있다. 

### ==4.3 Stability Comparison==

LSGAN과 Regular GAN의 안정성을 비교해 볼 것이다.

Fig.7
Batch normalization(BN)을 제거하고 비교하는 것이다. 
( a ) : Adam / BN없는 $G$ / LSGAN
( b ) : Adam / BN없는 $G$ / Regular GAN
( c ) : RMSProp / BN없는 $G,D$ / LSGAN
( d ) : RMSProp / BN없는 $G,D$ / Regular GAN

먼저 ( a ) : LSGAN의 경우 10번 테스트했을 때, 5번은 성공적으로 이미지를 생성했다. 하지만 ( b ) : Regular GAN은 한 번도 성공하지 못했다. Regular GAN은 mode collapse에 계속 시달리고있다. 
Regular GAN에서는 RMSProp가 데이터의 분포를 학습가능한 반면에 Adam은 학습하는데에 실패했다. 

다른 실험으로는 Gaussian Mixture distribution dataset에 대한 것 이다. Fig.8은 Gaussian kernel density 추정의 결과를 보여준다. Regular GAN이 step 15K부터 mode collapse에 빠지는 것을 볼 수 있다. 반면에 LSGAN은 성공적으로 Gaussian mixture distribution은 학습했다. 

Fig.8사진
위 : LSGAN / 아래 : Regular GAN

### ==4.4 Handwritten Chinese Characters==

Fig.9사진
이 실험을 통해 총 두 가지를 얻을 수 있었다.

* LSGAN으로 생성된 문자는 읽을 수 있었다.
* Label vectors를 통해 정확한 label의 이미지를 생성할 수 있었고 추후에 data augmentation에도 사용될 수 있다. 

## ==5. Conclusions and Future Work==

추후의 과제로는 샘플을 decision boundary로 밀어내는 것 대신에, 실제 데이터로 곧바로 생성된 샘플을 보내는 방법을 설계해야한다고 말하고 있다.

요약해보자면, 기존 GAN의 loss function에 Least Square를 사용함으로써, 학습시 모델의 안정성(gradient vanishing 문제 / mode collapse 문제), 그리고 생성 이미지의 품질 향상에 큰 기여를 했다고 볼 수 있다. 그리고 BN을 제거해도 성능이 좋다는 것을 증명했다. 

