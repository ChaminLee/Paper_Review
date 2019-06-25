# < From GAN to WGAN >

## *GAN의 발전..!*

## ==0. Abstract==

이 논문은 GAN이 왜 학습하기 어려운지에 대해 수학적으로 설명하고, WGAN이 GAN의 학습시에 두 분포간 거리를 측정하는 방법에 어떠한 변화를 주어 학습을 증진시켰는지 알아볼 것 이다. 

## ==1. Introduction==

GAN은 다양한 많은 분야에서 좋은 산출물을 냈다. 하지만 GAN은 학습시 불안정하고, 수렴에 실패하는 문제에 대해 직면하고있다.

그래서 이러한 GAN의 문제를 수학적 배경으로 왜 학습하기 힘든지를 설명하고 학습이 어려운 것을 해결하기 위해 등장하는 수정된 GAN에 대해 설명할 것이다.

## ==2. Kullback–Leibler and Jensen–Shannon Divergence==

GAN에 대해서 살펴보기 전에, 두 확률분포간의 유사성의 정도를 측정하는 두 가지 방법에 대해 알아보자.

##### (1) KL(Kullback-Leibler) Divergence

KL Divergence는 하나의 분포 $p$가 두 번째 확률분포인 $q$로부터 얼마나 거리가 있는지를 측정하는 방법이다. 

$$D_{KL}(p||q) = \int_x p(x) \log\frac{p(x)}{q(x)} dx$$

$D_{KL}$은 $p(x)==q(x)$일 때, 최솟값인 0을 갖는다.
또한 위의 식을 봤을 때, KL Divergence는 "비대칭적"이라는 것을 알 수 있다. 만약에 $p(x)$가 0에 가깝지만, $q(x)$는 0이 아닐 경우 $q$의 효과는 무시된다. (분자가 0이면 분모가 어떤 것이어도 0이기 때문이다.) 이는 두 개의 비슷하고 중요한 분포사이의 유사성을 측정하고 싶을 때 문제를 일으킬 수 있다.

##### (2) Jensen-Shannon Divergence

JSD는 [0,1]범위내에서 두 확률분포의 유사성을 측정하는 또 다른 방법이다. JSD는 대칭적이고, 더 스무스하다. 

$$D_{JS}(p||q) = \frac{1}{2} D_{KL}(p||\frac{p+q}{2}) +\frac{1}{2}D_{KL}(q||\frac{p+q}{2}) $$

어떤 논문에서는 GAN의 큰 성공요인이 비용함수의 KL Divergence를 JSD로 바꾸었기 때문이라고한다. 

Fig.1사진
$D_{KL}$은 비대칭적이고, $D_{JS}$는 대칭적인 것을 볼 수 있다. 

## ==3. Generative Adversarial Network==

GAN은 두 가지 모델로 구성되어있다.

(1) 식별자인 $D$는 주어진 샘플이 실제 데이터 셋에서 왔을 확률을 추정한다. critic처럼 장동하고, 가짜 샘플과 실제 샘플을 구분하는데 최적화되어 있다. 

(2) 생성사 $G$ 노이즈 값인 $z$를 입력으로 받아 조작된 샘플을 만들어낸다. 이는 실제 데이터 분포를 닮기 위해 훈련되기에 실제처럼 샘플을 생성할 수 있다. 다른 말로, 식별자가 가짜 샘플을 받아도 높은 확률을 내뱉게하도록 속일 수 있다. 

Fig.2 사진 

두 모델은 서로에 대해 경쟁하면서 다음과 같은 학습과정을 거친다 : 생성자 $G$는 식별자를 속이기 위해 노력하고, 반면 식별자 $D$는 속지않기 위해 열심히 노력한다. 이러한 제로섬게임이 두 모델의 기능을 모두 증진시킨다.

annotation을 정리하고 넘어가자.

$p_z$ : 노이즈 입력 $z$에 대한 데이터 분포 
$p_g$ : 데이터 $x$에 대한 생성자의 분포
$p_r$ : 실제 샘플 $x$에 대한 데이터 분포

한편으로는, 실제 데이터에 대한 식별자 $D$의 결정이 다음식  $\mathbb{E}_{x \sim p_r(x)}[\log D(x)]$을 최대화함으로써 정확하기를 원한다. 한편, $z \sim p_z(z)$의 분포로 생성된 가짜 샘플 $G(z)$가 주어졌을 때, 식별자의 예측 결과는 $D(G(z))$이다. $\mathbb{E}_{z \sim p_r(z)}[1-\log D(G(x))]$를 최대화함으로써 값을 0에 가깝게한다.

반면에, 생성자는 $D$가 가짜 샘플에 대해 높은 확률을 만들어 낼 가능성을 증가시키도록 훈련된다. 따라서 $\mathbb{E}_{z \sim p_r(z)}[1-\log D(G(x))]$를 최소화하도록 훈련하는 것이다. 

두 개의 양상을 같이 봤을 때, $D$와 $G$는 minimax game을 하면서 다음의 비용함수를 최적화해야한다.

$$\displaystyle\min_{G}\displaystyle\max_{D} L(D,G) = \mathbb{E}_{x \sim p_r(x)}[\log D(x)] +\mathbb{E}_{z \sim p_r(z)}[1-\log D(G(x))]$$

$$=\mathbb{E}_{x \sim p_r(x)}[\log D(x)] +\mathbb{E}_{x \sim p_g(x)}[1-\log D(x)]$$

$\mathbb{E}_{x \sim p_g(x)}[1-\log D(x)]$는 gradient descent 업데이트 하는 동안에 $G$에 영향을 주지 않는다. 

### ==3.1 What is the Optimal Value for D==

$$L(G,D) = \int_x \Big(p_r(x)\log(D(x)) + p_g(x) \log(1-D(x))\Big)dx $$

위의 식 $L(G,D)$를 최대화하기 위한 $D(x)$의 값에 관심이 있다. 다음과 같이 라벨링 해보자.

$$\tilde{x}=D(x), A=p_r(x), B=p_g(x)$$

이를 이용해서 위의 식을 치환하면 다음과 같다.

$$f(\tilde{x}) = A\log \tilde{x} + B \log (1-\tilde{x})$$

$$\frac{df(\tilde{x})}{d\tilde{x}}=A\frac{1}{ln10}\frac{1}{\tilde{x}}-B\frac{1}{ln10}\frac{1}{1-\tilde{x}}$$

$$=\frac{1}{ln10}(\frac{A}{\tilde{x}}-\frac{B}{1-\tilde{x}})$$

$$=\frac{1}{ln10}\frac{A-(A+B)\tilde{x}}{\tilde{x}(1-\tilde{x})}$$

미분값을 0이라고 설정하면, 식별자의 최적인 다음의 값을 얻는다 : $D^*(x) = \tilde{x}^*=\frac{A}{A+B}=\frac{p_r(x)}{p_r(x)+p_g(x)} \in [0,1]$.
생성자가 최적으로 학습되면서 $p_g$는 점점 $p_r$에 가까워진다. 그러다가 $p_g=p_r$일 때, $D^*(x)=\frac{1}{2}$가 된다. 

### ==3.2 What is the Global Optimal==

$L(G,D^*) = \int_x \Big(p_r(x)\log(D^*(x)) + p_g(x) \log(1-D^*(x))\Big)dx$

$=\log\frac{1}{2}\int_xp_r(x)dx + \log\frac{1}{2}\int_xp_g(x)dx$

$=-2\log2$


### ==3.3 What does the Loss Function Represent?==

$D_{JS}$는 다음과 같이 계산된다.

$$D_{JS}(p||q) = \frac{1}{2} D_{KL}(p||\frac{p+q}{2}) +D_{KL}(q||\frac{p+q}{2}) $$

$$=\frac{1}{2}\Big(\log2+\int_xp_r(x)\log\frac{p_r(x)}{p_r +p_g(x)}dx\Big) + \frac{1}{2}\Big(\log2+\int_xp_g(x)\log\frac{p_g(x)}{p_r +p_g(x)}dx\Big)$$

$=\frac{1}{2}\Big(\log4 + L(G,D^*)\Big)$

그러므로

$$L(G,D^*) = 2D_{JS}(p_r||p_g)-2\log2$$

최적의 값은 $L(G^*,D^*)=-2\log2$이다. 

##### ==Other Variations of GAN==

다른 task를 위한 다양한 GAN이 존재한다. 

## ==4. Problems in GANs==

GAN은 학습이 불안정하고 느려 학습이 쉽지 않지만, 실제와 비슷한 이미지를 만들어내는데 성공했다. 

### ==4.1 Hard to Achieve Nash Equilibrium==

GAN의 gradient descent 기반의 학습과정에 문제가 있다는 것은 인식되었다. 두 모델들은 동시에 서로 협력하지 않으며 "Nash Equilibrium"을 찾기 위해 학습된다. 하지만 다른 모델의 비용함수에는 상관없이 비용 독립적으로 모델이 업데이트 된다. 동시에 두 모델의 gradient를 업데이트 하는 것은 수렴을 장담하지 못하기 때문이다. 

예를 들어서 이해해보자. 한 플레이어는 $x$를 조정하여 $f_1(x) = xy$를 최소화하려하고, 그 때 다른 플레이어는 $y$를 업데이트하여 $f_2(y)=-xy$ 식을 최소화하려한다고 보자.

위 식의 gradient를 구하기 위해 편미분을 해보면 다음과 같다. $\frac{\partial f_1}{\partial x}=y$, $\frac{\partial f_2}{\partial y}=-x$, 우리는 $x$를 $x-\eta *y$, $y$를 $y+\eta*x$를 이용하여 동시에 업데이트한다. ($\eta$=learning rate)

$x,y$는 다른 부호를 가지고 있기에, 모든 gradient 업데이트는 큰 진동과 불안정성을 초래한다. 

Fig.3 사진
큰 진폭과 불안정함을 볼 수 있다. 

### ==4.2 Low Dimensional Supports==

manifold 사진

* Manifold : 각 지점 근처의 유클리드 공간과 지역적으로 닮은 기하학적 공간을 말한다. 유클리디안 공간의 차원이 $n$이라면 manifold는 $n$-manifold라고 부른다. 

* Support : 실수형 함수 $f$는 0으로 매핑되지 않는 요소들을 포함하는 하위 집합이다. 

 어떠한 논문에서는 $p_r$과$p_g$의
 low dimensional manifolds에 존재하는 supports에 문제가 있다고 말하고, 이것이 GAN 학습의 불안정성에 어떠한 영향을 주는지에 대해 말한다. 

$p_r$로 표현되어지는 실제 세계의 다양한 데이터셋의 차원은, 인위적으로 높게 나타날 뿐이다. 이는 낮은 차원의 manifold에 집중하는 것으로 밝혀졌다. 이것이 사실 "Manifold Learning"의 기본 가정이다. 실제 이미지를 생각해보면, 주제나 물체가 고정되면 이미지에는 많은 규제들이 따라온다. 예를 들어, 강아지는 두 개의 귀, 꼬리가 있어야하고, 초고층빌딩은 곧고 기다란 몸체를 가져야한다. 이러한 규제는 이미지가 고차원 공간의 자유로운 형태와 멀리 떨어지게한다. 

$p_g$도 저차원 공간의 manifolds에 있다. generator가 100차원 노이즈 인풋 $z$를 이용해 64x64와 같이 더 큰 차원의 이미지를 생성해야할 경우, 4096 픽셀들의 컬러 분포는 100차원의 작은 랜덤 넘버 벡터에 의해 결정되며 이 때 고차원의 공간 전체를 거의 채울 가능성은 매우 낮다.

$p_g$, $p_r$ 모두 저차원 공간의 manifolds에 놓여져있기에 Fig.4처럼 둘은 완벽하게 분리가능할 것이다. 이 둘이 분리가능한 supports를 가지고 있다면, 실제와 가짜를 구분하는데에 100%의 정확도를 가진 완벽한 식별자를 항상 찾을 수 있다. 

Fig.4 사진

### ==4.3 Vanishing Gradient==

식별자가 완벽하다면, $D(x)=1,\forall \in p_r$ 그리고 $D(x)=0,\forall \in p_g$를 보장할 수 있다. 따라서 비용함수 $L$은 0으로 떨어지고, 학습과정에서 비용함수를 업데이트하기 위한 gradient없이 끝나게 된다. Fig.5에서는 식별자가 더 잘 할수록 gradient가 빠르게 줄어드는 것을 볼 수 있다. 

그 결과로, GAN 학습시 다음의 딜레마와 부딪힌다. 

* 식별자의 성능이 나쁠 때, 생성자는 정확한 피드백을 하지 않고, 비용함수는 현실을 반영하지 못한다.
* 식별자가 매우 성능이 좋을 경우, 비용함수의 gradient는 0에 점점 가까워지고, 학습은 매우 느려지거나 멈추게 된다.

이러한 딜레마가 GAN 학습을 매우 어렵게 한다.

Fig.5 사진

### ==4.4 Mode Collapse==

학습하는동안, 생성자는 "Mode Collapse"라 불리는 항상 같은 결과물을 생성하는 현상을 보일 수 있다. GAN의 실패 케이스로 만연하다. 비록 생성자가 식별자를 속일지라도, 복합적인 현실 데이터 분포를 학습하는 것은 실패하고, 매우 적은 다양성과 함께 작은 공간에 갇히게 된다.

Fig.6
Mode Collapse의 결과

### ==4.4 Lack of a Proper Evaluation Metric==

GAN은 태생적으로 학습과정을 나타내는 목적함수가 없다. 좋은 평가지표 없이, 깜깜한 곳에서 일하는 것과 같다. 언제 멈춰야할지 말해주는 사인이 없는 것이다 ; 복수 개의 모델중에서 어느 것이 나은지도 말해주지 못한다. 


## ==5. Improved GAN Training==

다음의 제안들은 GAN의 학습을 증진시키고 안정화하기위해 도움을 주기 위해 제의되었다.

첫 5가지 방법은 GAN의 학습의 빠른 수렴을 위한 실용적인 기술이다. 그리고 마지막 2개는 disjoint distribution 문제를 해결하기 위해 제의되었다. 

##### (1) Feature Matching

Feature matching은 생성자의 결과와 실제 데이터 샘플을 비교하여 기대 수준의 통계값을 얻도록 하는 식별자를 최적화하는 것이다. 비용함수는 다음과 같다. 
$$||\mathbb{E}_{x\sim p_r}f(x) - \mathbb{E}_{z\sim p_z}f(G(z))||_2^2$$ 
$f(x)$는 평균이나, 중간값같은 feature의 통계값을 사용한다. 

##### (2) Minibatch Discrimination

Minibatch discrimination은 식별자가 각 포인트를 독집적으로 처리하는 대신에, 하나의 배치안에서의 다른 데이터간의 관계를 고려하는 것이다. 

하나의 배치안에서는, $c(x_i,x_j)$로 나타나는 샘플의 모든 쌍 사이의 유사도를 측정하고, 한개의 데이터가 같은 배치 내에서 다른 데이터들과 얼마나 가까운지를 나타내는 값, $o(x_i) = \sum_j c(x_i, x_j)$를 계산합니다. 계산된 $o(x_i)$를 모델 입력값에 명시적으로 추가하여 다른 데이터들 간의 관계를 고려하여 학습되도록 한다.

##### (3) Historical Averaging

두 모델(D,G)에 대하여, $||\Theta-\frac{1}{t}\sum_{i=1}^{t}\Theta_i||^2$를 비용함수에 추가한다. $\Theta$는 모델의 파라미터이고, $\Theta_i$는 $i$번째 학습 과정의 파라미터이고, 이는 $\Theta$가 너무 급속도로 바뀔때 학습 과정에 페널티를 준다.

##### (4) One-sided Label Smoothing

식별자를 학습할 때, 0,1로 라벨링을 보내지 않고 0.1, 0.9로 보내는 것이 네트워크의 취약성을 감소시키는 모습을 보였다. 

##### (5) Virtual Batch Normalization(VBN)

minibatch를 사용하는 것이 아니라, 데이터의 고정된 배치를 사용하여 데이터 샘플을 정규화하는 것이다. 이 고정된 배치는 처음에 선택되어지고 학습동안에 바뀌지 않는다. 

##### (6) Adding Noises

4.2절에서 논의된 것 처럼, $p_r$과 $p_g$가 고차원 공간에서 겹치지않고, 이 것이 gradient vanishing 문제를 야기한다는 것도 않다. 인위적으로 분포를 펼쳐서 두 확률분포가 겹칠 확률을 높게 만들기 위해, 식별자 $D$의 입력에 연속적인 noises를 추가하는 방법이다. 

##### (7) Use Better Metric of Distribution Similarity

GAN의 비용함수는 $p_r$과 $p_g$의 분포 사이의 JS divergence를 추정한다. 이 방법은 두 분포가 겹치지 않을 때, 의미있는 값을 도출하지 못한다. Wasserstein 방법은 더 연속적인 값의 범위를 가지기에 JS divergence를 대신한다.

## ==6. Wasserstein GAN(WGAN==

### ==6.1 What is Wasserstein Distance?==

Wasserstein Distance는 두 확률 분포간의 사이를 측정하는 방법이다. Earth Mover's Distance라고도 불려서, 줄여서 EM distance라고한다. 왜냐하면, 한 분포에서 다른 분포의 모양을 닮기 위해 옮겨지는 모래더미의 최소 비용이라고 해석되어진다. 비용은 "옮겨진 모래의 양"X"이동거리"로 계산된다.

비연속적인 아래의 확률을 보자. 예를 들어 두 개의 분포 $P$, $Q$가 있을 때, 4개의 파일이 있고, 모래양의 총합이 10이다. 파일의 모래양은 다음과 같이 할당되어져 있다.

$$P_1=3, P_2=2,P_3=1,P_4=4$$
$$Q_1=1,Q_2=2,Q_3=3,Q_4=3$$

$P$를 $Q$처럼 보이게 바꾸기 위해서는 Fig.7처럼 하면된다.

* 먼저, $(P_1,Q_1)$을 같게하기 위해 $P_1$에서 $P_2$로 2만큼 옮긴다.
* 그리고 $(P_2,Q_2)$을 같게하기 위해 $P_2$에서 $P_3$로 2만큼 옮긴다.
* 마지막으로  $(P_3,Q_3),(P_4,Q_4)$을 같게하기 위해 $Q_3$에서 $Q_4$로 1만큼 옮긴다.

Fig.7 사진

$P_i$와 $Q_i$가 같도록하는 비용을 $\delta_i$라고 하면, $\delta_{i+1} = \delta_i + P_i - Q_i$라는 식을 세울 수 있다. 

$$\delta_0 = 0$$
$$\delta_1 = 0 + 3 - 1 = 2$$
$$\delta_2 = 2 + 2 - 2 = 2$$
$$\delta_3 = 2 + 1 - 4 = -1$$
$$\delta_4 = -1 + 4 - 3 = 0$$

최종적으로 EM distance는 $W=\sum|\delta|=5$가 된다. 

연속적인 분포를 다루게 된다면 거리 식은 다음과 같아진다. 

$$W(p_r,p_g)=\displaystyle{\inf_{\gamma\sim\Pi(p_r,p_g)}}\mathbb{E}_{(x,y)\sim\gamma}[||x-y||]$$

위 식에서, $\Pi(p_r,p_g)$는 $p_r$과 $p_g$사이에서 가능한 모든 결합확률분포 세트를 말한다. 하나의 결합분포 $\gamma\in\Pi(p_r,p_g)$는, 위의 예처럼 모래를 한 번 옮긴 것을 말한다.(여기는 연속적) 정확하게 말하면 $\gamma(x,y)$는 $x$가 $y$분포를 닮도록 하기위해 $x$에서 $y$로 옮겨야하는 모래의 비율이다. 따라서 $\gamma(x, y)$를 $x$에 대한 marigal distribution으로 계산하면 $p_g$와 같아진다. $\sum_x \gamma(x,y)=p_g(y)$ ($x$를 $p_g$를 따르는 $y$가 되도록 흙더미를 옮기고 나면, 마지막 분포는 $p_g$와 같아진다.)  마찬가지로 $y$에 대한 marginal distribution은 $p_r(x)$가 된다. $\sum_y \gamma(x,y)=p_r(x)$

$x$를 시작점으로, $y$를 도착점으로 할 때, 옮겨진 모래의 총량은 $\gamma(x,y)$이고, 이동거리는 $||x-y||$, 비용은 $\gamma(x,y)\cdot||x-y||$이다. 모든 $(x,y)$에 대해 기대비용을 계산하는 식은 아래와 같다.

$$\sum_{x,y}\gamma(x,y)||x-y|| = \mathbb{E}_{x,y\sim\gamma}||x-y||$$

최종적으로, EM distance로 계산되는 값중에 가장 작은 값을 고른다. Wasserstein distance의 정의의 inf는 오직 최솟값에만 관심이 있다는 것을 말한다. (inf = 최소중에서 최대값(greatest lower bound))

### ==6.2 Why Wasserstein is better than JS or KL divergence?==

비록 두 분포가 겹치는 부분 없이 저차원 공간의 manifolds에 위치해도, Wasserstein distance는 여전히 의미있는 값과 연속적으로 거리를 표현한다. 

다음의 예를 보자.
$P$와 $Q$라는 두 분포가 있다고 가정하자.

$$\forall(x,y) \in P, x=0  \& y\sim U(0,1)$$
$$\forall(x,y)\in Q, x=\theta, 0≤\theta≤1 \& y \sim U(0,1)$$

Fig.8

$\theta ≠ 0$일 때,$P$와 $Q$는 겹치지 않는다.

$D_{KL}(P||Q) = \displaystyle\sum_{x=0,y\sim U(0,1)} 1 \cdot \log\frac{1}{0} = + ∞$

$D_{KL}(Q||P) = \displaystyle\sum_{x=\theta,y\sim U(0,1)} 1 \cdot \log\frac{1}{0} = + ∞$

$D_{JS}(P,Q) = \frac{1}{2}(\displaystyle\sum_{x=0,y\sim U(0,1)} 1 \cdot \log\frac{1}{1/2} + \displaystyle\sum_{x=0,y\sim U(0,1)} 1 \cdot \log\frac{1}{1/2} = \log2$

$W(P,Q)=|\theta|$

하지만 $\theta=0$이라면 두 분포는 완벽하게 겹쳐진다.

$D_{KL}(P||Q)=D_{KL}(Q||P)=D_{JS}(P,Q)=0$
$W(P,Q)=0=|\theta|$

$D_{KL}$은 두 분포가 겹치지 않을 때 무한하고, $D_{JS}$는 점프를 해서 $\theta=0$부분에서 미분 불가능하다. Wasserstein 방법만이 연속적으로 측정가능하다. 이는 gradients descents 학습과정의 안정화에 매우 도움이 된다. 

### ==6.3 Use Wasserstein Distance as GAN Loss Function==

$\inf_{\gamma\sim\Pi(p_r,p_g)}$를 계산하기 위해 $\Pi(p_r,p_g)$의 모든 결합분포를 추적하는 것은 불가능하다. 그래서 식을 Kantorovixh-Rubinstein duality를 이용해 변형했다. 

$$W(p_r,p_g)=\frac{1}{K}\displaystyle\sup_{||f||_L ≤K}\mathbb{E}_{x\sim p_r} [f(x)] - \mathbb{E}_{x\sim p_g}[f(x)]$$

sup(least upper bound)은 inf의 반대로, 최댓값을 찾고 싶은 것이다. 

#### ==6.3.1 Lipschitz Continuity==

새로운 형태의 Wasserstein 방법의 $f$는 $||f||_L ≤ K$를 만족해야한다. 즉, $K$-Lipschitz continuous를 만족해야 한다.

모든 $x_1, x_2 \in \mathbb{R}$ 에 대해서 
$$|f(x_1) -f(x_2)| ≤ K |x_1  - x_2|$$ 를 만족하는 실수값 $K≥0$이 존재할 때, 실수형 함수 $f : \mathbb{R} → \mathbb{R}$ 가 $K$-Lipschitz continuous를 만족한다고 한다.

$K$는 $f(.)$의 Lipshitz 상수라고 알려져있다.       모든 곳에서 연속적으로 미분이 가능하면 Lipshitz continuous이다. 미분을 해보면 $\frac{|f(x_1)-f(x_2)|}{|x_1-x_2|}$로, 미분가능하다는 것은 범위가 유한하다는 것을 의미하기 때문이다. 하지만 반대로 Lipschitz continuous라고 해서 모든 점에서 미분 가능한 것은 아니다. (예, $f(x)=|x|$)

#### ==6.3.2 Wasserstein Loss Funtion==

함수 $f$가 $w$를 파라미터로 가진 $K$-Lipschitz continuous functions의 집합, $\{f_w\}_{w \in W}$ 에서 추출되었다고 가정해보자. 수정된 Wassertein-GAN에서 식별자는 좋은 $f_w$를 찾기위해 학습이 되고, 손실함수는 $p_r$과 $p_g$사이의 wasserstein distance를 측정하게 된다.

$$L(p_r,p_g) = W(p_r,p_g) = \displaystyle\max_{w\in W}\mathbb{E}_{x \sim p_r}[f_w(x)] - \mathbb{E}_{z\sim p_r(z)}[f_w(g_\theta(z))]$$

따라서 식별자는 실제 샘플과 가짜 샘플을 구별하는 지표가 아니다. 대신에 Wasserstein distance를 계산하는 것을 돕기 위한 $K$-Lipschitz continuous를 학습한다. 학습시에 비용함수가 줄어들면, Wasserstein distance는 더 작아지고, 생성모델의 결과물은 실제 분포와 점점 더 유사해진다. 

모든 것을 잘 작동시키기 위해서는 $f_w$의 $K$-Lipschitz contunuity를 유지해야한다는 것이 한 가지 큰 문제이다. 이 논문에서는 간단하지만 매우 실용저긴 트릭을 내놓는다 : 매번 gradient를 업데이트하고, 가중치 $w$를 [-0.01,0.01]사이의 값으로 고정시켜서, compact parameter 공간인 $W$가 되도록 한다. 그리고 $f_w$는 하한선과 상한선이 생기면서 Lipschitz continuity를 유지하게 된다. 

알고리즘 사진

기존 GAN과 비교했을 때, WGAN은 다음의 변화를 수행한다. 

* 비용함수가 업데이트 된 후, 가중치는 [-c,c] 범위의 값으로 고정된다.
* Wasserstein distance로부터 파생된 새로운 비용함수는 더 이상 로그형태가 아니다.(logarithm) 식별자 모델은 직접적인 구별을 하지 않고 실제와 생성된 데이터 분포 사이의 Wasserstein 방법으로 측정하는데 도움을 준다. 
* 저자는 실험적으로 RMSProp를 사용할 것을 추천한다. 왜냐하면 모멘텀 기반의 optimizer인 Adam은 학습동안의 모델의 불안정성을 야기할 수 있기 때문이다. 

슬프게도, WGAN은 완벽하지 않다. "가중치를 고정시키는 것은 Lipshitz 를 강제하는 것과 같다"라며 말하기도 한다. WGAN은 여전히 불안정한 학습, 가중치 고정이후 느린 수렴(고정범위가 클 경우), gradient vanishing(고정범위가 너무 작을 때,)으로부터 고통받는다. 

이후의 발전으로는 clipping 보다는 "gradient penalty"가 더 낫다는 논의가 있었다. 

