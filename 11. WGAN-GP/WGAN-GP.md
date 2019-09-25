# < WGAN-GP : Improved Training of Wasserstein GANs >

## *weight clipping $\rightarrow$ Gradient penalty*

## ==0. Abstract==

WGAN에서 critic에 대해 Lipschitz constraint를 강요하기위한 weight clipping이 발생시키는 문제, 즉 undesired behavior로 이끌 수 있다 라는 것을 발견했다. 논문에서는 weight clipping 대신에 다른 것을 제안한다 : 입력에 대하여 critic의 gradient norm을 처벌하는 것이다. 

## ==1. Introduction==

WGAN은 discriminator(논문에선 critic)이 1-Lipshichtz function의 공간에 놓여져있기를 원한다.(저자들은 weight clipping으로 강요하려함)
논문에서 말하고자 하는 것은 다음 3가지와 같다.

1. Toy datasets에 대해 critic의 weight clipping이 undesired behavior를 유발할 수 있다는 것을 증명한다.
2. "Gradient penalty"(WGAN-GP)를 제안함으로인해 같은 문제로부터 시달리지 않을 것이다.
3.  다양한 GAN 구조에대해 안정적인 학습을 증명하고, weigth clipping에 대한 성능 향상, 고품질 이미지 생성, 개별 샘플링이 없이 문자 수준의 GAN 언어모델을 선보인다.

## ==2. Background==

### ==2.1 Generative adversarial networks==

[Vanilla GAN](https://leechamin.tistory.com/177?category=839075)
GAN에 대한 설명은 항상 나오기 때문에 위의 링크를 참고하면 될 것 같다.
그래도 minimax 목적 함수만 적자면 다음과 같다.

$$\displaystyle\min_G \displaystyle\max_D = \mathbb{E}_{x \sim \mathbb{P}_{r}}[\log (D(x))] + \mathbb{E}_{z \sim \mathbb{P}_{g}}[\log (1-D(\tilde{x}))]$$

만약에 각 generator 파라미터가 업데이트되기 전에 discriminator가 최적으로 학습되었다면, 가치 함수가 $\mathbb{P}_r$과 $\mathbb{P}_g$사이의 JS Divergence를 최소화할 수 있지만, 그렇게 하면 discriminator가 포화되는(saturates) vanishing gradients 문제를 유발할 수 있다. 

### ==2.2 Wasserstein GANs==

기존의 WGAN은 GAN이 전형적으로 최소화하려는 divergence(JSD)는 generator의 파라미터에 관련하여 연속적이지 않아 학습에 문제가 발생한다. (JSD의 경우 0에서 불연속적인 것을 말함)

Wasserstein JSD 사진

그래서 Earth-Mover distance $W(q,p)$(q분포를 p분포로 변형하기 위해 옮겨지는 이동량의 최소비용을 구하는 것 / 비용=양X이동 거리)를 대신 사용한다. 유한 가정하에서, $W(q,p)$는 모든 구간에서 연속적이고 거의 모든 곳에서 미분 가능하다. 

WGAN 가치 함수는 Kantorovich-Rubinstein duality를 이용해서 구성된다.

$$\displaystyle\min_G \displaystyle\max_{D \in \mathcal{D}}  = \mathbb{E}_{x \sim \mathbb{P}_{r}}[D(x)] - \mathbb{E}_{\tilde{x} \sim \mathbb{P}_{g}}[D(\tilde{x})]$$

$\mathcal{D}$는 1-Lipshichtz functions의 세트이고, $\mathbb{P}_g$는 $\tilde{x}=G(z),z \sim p(z)$로 다시 한 번 명백하게 정의된다.(노이즈가 입력인 생성모델 분포라는 것) 최적의 discriminator하에서는  generator 파라미터들에 대한 가치 함수를 최소화하여, $W(\mathbb{P}_r,\mathbb{P}_g)$를 최소화한다.

WGAN 함수는 GAN 보다 입력에 대한 gradient가 더 잘 동작하여 generator의 최적화를 더 쉽게하는 critic function을 만든다. 또한 실증적으로, WGAN의 가치 함수가 샘플의 품질과 상관관계가 있다는 것을 찾아냈다. 

critic에 Lipschitz constraint를 적용하기 위해서는, critic의 가중치를 clip하여 compact space [-c,c]에 배치할 것을 제안한다. 


### ==2.3 Properties of the optimal WGAN critic==

왜 weight clipping이 WGAN critic에서 문제를 일으키는지를 이해하기 위해서는 WGAN의 최적의 critic의 특성을 알아야한다. 이는 부록에 자세히 설명되어 있다.

##### ==Proposition 1.==

$\mathbb{P}_r$과 $\mathbb{P}_g$를 compact space $\mathcal{X}$에 속하는 두 분포라고 하자. 그리고, $max_{||f||_{L}≤1}\mathbb{E}_{y \sim \mathbb{P}_r}[f(y)] - \mathbb{E}_{x \sim \mathbb{P}_g}[f(x)]$의 최적의 해결책인 1-Lipschitz function $f^*$가 존재한다. $\pi$를 $\mathbb{P}_r$과 $\mathbb{P}_g$사이의 최적의 연결이고, minimizer로 정의된다 : $W(\mathbb{P}_r,\mathbb{P}_g) = \inf_{\pi \in \Pi(\mathbb{P}_r,\mathbb{P}_g)}\mathbb{E}_{(x,y)\sim \pi}[||x-y||]$, 

$\Pi(\mathbb{P}_r,\mathbb{P}_g)$는 marginals가 각각 $\mathbb{P}_r,\mathbb{P}_g$인 결합 분포 $\pi(x,y)$들의 집합이다.

그러면 $f^*$는 미분가능하기에, $\pi(x=y)=0$, 그리고 $x_t=tx+(1-t)y$ with $0≤t≤1$, 이는 $\mathbb{P}_{(x,y)\sim \pi} [\nabla f^*(x_t) = \frac{y-x_t}{||y-x_t||}] = 1$라고 말한다. 

##### ==Corollary 1.==
$f^*$는 거의 모든 $\mathbb{P}_r,\mathbb{P}_g$하에서 gradient norm 1을 가진다.

## ==3. Difficulties with weight constraints==

WGAN에서 weight clipping이 최적화에 문제를 발생시킨다는 것을 찾았고, 최적화가 잘 되더라도 critic이 걷잡을 수 없는 값(pathological value surface)을 가질 수 있다. 

다양한 weight constraint에 대해 실험을 진행했다. WGAN의 hard clipping, L2 norm clipping, weight normalization, soft(L1 and L2 weight decay), 이 들이 다 비슷한 문제를 보이는 것을 찾아냈다. 

Figure.1 사진
Gradient penalty는 weight clipping처럼 undesired behavior를 발생시키지 않는다. 

(a) : 위는 weight clipping을, 아래는 gradient penalty를 toy dataset에 최적으로 학습시킨 WGAN critic의 value surface이다. weight clipping으로 학습한 critics는 데이터 분포의 higher moments를 잡아내는데 실패했다. generator는 실제 데이터와 가우시안 노이즈에 고정되어있다.

(b) : 왼쪽은 Swiss Roll dataset에 대햇 학습시키는 도중 weight clipping을 사용할 때 발생하는 vanish / explode를 보여주고 gradient penalty 사용시 문제가 발생하지 않는 것을 보여주는 deep WGAN critics의 gradient norm이다.
오른쪽, 위는 weight clipping이 두 값으로만 weights를 쏠리게하는 것을 볼 수 있고/ 아래는 gradient penalty로 그러지 않는 것을 볼 수 있다. 

### ==3.1 Capacity underuse==

weight clipping 제약하에서, 논문에서는 최대 gradient norm $k$를 달성하려고 신경망 구조들이 단순한 기능을 학습하는 것을 알아냈다. 

이를 증명하기 위해, generator $\mathbb{P}_g$는 실제 데이터 분포 + unit-variance 가우시안 노이즈에 고정시키고, WGAN critic을 weight clipping으로 여러 toy 분포에 최적화하기 위해 학습한다. value surface는 Fig.1(a)에 나타나있다. 논문에서는 critic에서의 batch normalization을 생략한다. 

Algorithm

이러한 경우에, weight clipping으로 학습된 critic은 데이터 분포의 higher moments를 무시하지만, 대신 최적의 기능에 대한 매우 간단한 근사치를 모델링한다. 대조적으로, 논문의 접근방식은 이러한 행동으로부터 고통받지 않는다. 

### ==3.2 Exploding and vanishing gradients==

WGAN 최적화 과정이 가중치 제약과 비용 함수간의 상호작용 때문에 어렵다는 것을 관측했다. (clipping threshold인 c를 조심스레 조정하지 않으면 vanishing / exploding이 발생한다.)

Clipping threshold인 $c$를 $[10^{-1},10^{-2},10^{-3}]$으로 다양화하여 Swiss Roll dataset에 대한 critic의 gradient norm을 플롯화했다. 
Fig.1(b)를 통해 더 안정적인 gradient가 vanish / explode를 야기하지 않고, 더 복잡한 네트워클르 학습시키게 한다는 것을 발견했다. 

## ==4. Gradient penalty==

이제는 Lipschitz constraint를 시행할 수 있는 대안을 제안한다. 미분가능한 함수는 모든 곳에서 gradients norm이 1이어야만 1-Lipschtiz이다. 따라서 입력과 관련하여 critic의 결과의 gradient norm을 직접 규제하는 것을 고려한다. 다루기 쉬운 문제를 피하기 위해서, 랜덤 샘플 $\tilde{x} \sim \mathbb{P}_{\tilde{x}}$에 대해 gradient norm에 패널티와 함께 soft한 버전의 제약을 준다. 새로운 목적함수는 다음과 같다.

목적함수 사진

#### ==Sampling distribution==

논문에서는 데이터 분포 $\mathbb{P}_r$으로 부터의 샘플링된 점들의 쌍과 generator 분포 $\mathbb{P}_g$간의 사이에 직선을 따라 $\mathbb{P}_{\hat{x}}$ 샘플링을 정의한다. 이는 최적의 critic이 $\mathbb{P}_r,\mathbb{P}_g$로 부터의 점들을 연결하는 gradient norm 1의 직선을 보유하고 있다는 사실로부터 동기부여 된 것 이다. 모든 곳에서 gradient norm 제약을 주는 것은 어렵기 때문에, 이러한 직선을 따라 시행하는 것으로도 충분하고 실험적으로 좋은 성능을 얻을 수 있다. 

#### ==Penalty coefficient==

논문에서의 모든 실험에 $\lambda = 10$을 사용했다. (toy task나  ImageNet CNN까지도 다양한 모델 구조에 대해 잘 작동하는 것을 확인) 

#### ==No critic batch normalization==

앞선 선행 GAN 구현들에서는 generator & discriminator 모두 batch normalization을 사용하여 학습을 안정화 시키는데 도움을 주려했지만, batch normalization은 discriminator의 단일 입력을 단일 출력으로 매핑하는 문제로부터, 입력의 전체 배치로부터 출력의 배치로 매핑하는 문제로 유형을 변화시킨다.  기존에 전체 배치가 아니라 각 입력에 독립적으로 critic의 gradient norm을 처벌하기 때문에, 논문의 패널티를 주는 학습은 이러한 환경에서 더 이상 유효하지 않다. 이를 해결하기 위해서, 간단하게 모델내의 critic에서 batch normalization을 생략한다.(잘 작동하는 것을 확인했다.) batch normalization 대체로 layer normalization을 추천한다. 

#### ==Two-sided penalty==

논문에서는 gradient의 norm이  1 아래에 머무르기(one-sided penalty) 보다는 1로 향하기(two-sided penalty)를 촉진한다. 실증적으로 critic을 많이 규제하지 않는 것 처럼 보인다. 아마 최적의 WGAN critic은 거의 모든 곳에서 $\mathbb{P}_r, \mathbb{P}_g$하에서 그리고 그 지역의 많은 부분에서 gradient norm 1을 가지고 있기 때문일 것이다. 

## ==5. Experiments==

### ==5.1 Training random architectures within a set==

DCGAN 구조에서 시작해서 Table.1에 대응하는 값으로 변경하여 구조를 바꿨다. 

Table.1

이 세트로부터 200개의 구조를 샘플하고 32x32 ImageNet 에 대해 WGAN-GP, standard GAN을 학습했다. Table.2는 경우의 수를 말한다. 성공의 기준은 inception_score >min_score 이다. WGAN-GP의 경우 많은 구조들이 학습하는데 성공했다. 

Table.2

Fig.2
다른 방법으로 학습된 다른 GAN 구조. WGAN-GP에서만 성공했다. 


### ==5.2 Training varied architectures on LSUN bedrooms==

WGAN-GP에서는 discriminator의 batch normalization 대신에 layer normalization으로 대체한다. Fig.2를 보면 알 수 있듯이, WGAN-GP만 성공하고 다른 모델들은 불안정하거나 mode collapse에 빠진 모습을 볼 수 있다. 

### ==5.3 Improved performance over weight clipping==

weight clipping 보다 학습 속도가 빠르고 샘플의 품질이 높아졌다는 것이 WGAN-GP의 이점이다. 이를 증명하기 위해서 weight clipping을 사용한 WGAN과 gradient penalty를 사용한 논문의 모델을 CIFAR-10에 학습하고 Inception score를 Fig.3 그래프에 그려놨다. 

Fig.3 사진
왼쪽은 iteration에 따른 Inception Score / 오른쪽은 시간에 따른 Inception Score이다. Weight clipping / GP(RMSProp) / GP(Adam) / DCGAN 이다. WGAN-GP는 weight clipping 보다 성능이 좋고 DCGAN과 비슷하다. 

RMSProp를 쓸 때는, weight clipping과 같은 learning rate를 사용하고, Adam 사용시에는, 더 높은 learning rate를 사용한다. 같은 optimizer일지라도, WGAN-GP는 더 빠르게 수렴하고 weight clipping보다 좋은 score를 보인다. 그리고 Adam이 더 좋은 성능을 보인다. DCGAN보다는 수렴속도가 느리지만(오른쪽) 수렴에 있어서 점수가 더 안정적이다. 


### ==5.4 Sample quality on CIFAR-10 and LSUN bedrooms==

동일한 구조에서, WGAN-GP는 standard GAN과 비교할 수 있는 샘플의 품질을 얻었다. Table.3을 통해 다양한 구조의 GAN과의 점수를 비교해두었다. 

Table.3
CIFAR-10에 대한 Inception score이다. Supervised에서는 SGAN을 제외하면 가장 좋은 성능을 보인다. 

deep ResNet을 이용하여 128X128 LSUN 침대 이미지를 학습하고 Fig.4에 나타냈다. 

Fig.4 사진

### ==5.5 Modeling discrete data with a continuous generator==

연속적인 공간에 대해 정의된 generator를 가지고 비연속적인 분포를 모델링하는 것에는 문제가 있었다. 이 문제에 대한 예로서, Google Billion Word 데이터셋에서 문자 수준의 GAN 언어 모델을 학습한 것을 볼 수 있다. 

모델은 빈번하게 철자 오류를 범하지만(문자를 독립적으로 출력하기 때문에) 그럼에도 불구하고 언어의 통계에 대해 많은 것을 학습하고 있다. 

Table.4 

모델은 별도의 샘플링 없이 latent vector로 부터 직접 one-hot character embedding하는 것을 학습한다. 기존의 GAN과는 비교할 만한 결과를 얻지는 못했다. 

WGAN과 GAN의 성능차이는 다음으로 설명될 수 있다.

Simplex : $\Delta_n =\{p \in \mathbb{R}^n : p_i ≥ 0, \sum_i p_i=1 \}$

Simplex에 대한 Set of vertices : $V_n = \{p \in \mathbb{R}^n : p_i \in \{0,1\}, \sum_i p_i = 1\} \subseteq \Delta_n$

단어의 크기 $n$, 크기의 시퀀스 $T$에 대한 분포 $\mathbb{P}_r$이 있다면, $\mathbb{P}_r$은 $V_n^T=V_n \times \cdots \times V_n$에 대한 분포이다. $V_n^T$가 $\Delta_n^T$의 서브셋이기에, $\mathbb{P}_r$을 $\Delta_n^T$에 대한 분포라고 다뤄도 된다. ($V_n^T$가 아닌 모든 점에 확률값 0을 할당하여서)

$\mathbb{P}_r$은 $\Delta_n^T$에 대해 비연속적이지만, $\mathbb{P}_g$는 $\Delta_n^T$에 대해 연속분포이다. 두 분포에 대한 KL divergences는 무한하고, JS divergence는 포화된다. 

Fig.5 사진
(a) LSUN 침대에 대한 모델의 negative critic loss가 학습함에 따라 최소로 떨어지는 것이다.
(b) MNIST의 무작위 1000개 숫자 집합에 대한WGAN training / validation loss은 왼쪽(GP)혹은 오른쪽(weight clipping)을 사용할 때 overfitting을 보여준다. 특히, GP방법에서는, critic이 generator보다 더 빨리 과적합되었고, 이는 training loss를 점차 증가시키고, validation loss는 떨어뜨렸다. 

GAN은 문자 그대로 이러한 차이를 최소화하지는 않지만, discriminator가 $V_n^T$에 있지 않은 모든 샘플을 거부하도록하고 generator에데 의미없는 gradients를 주는 것을 빠르게 학습한다는 것을 의미한다. 
WGAN에서 이러한 현상이 나타나는 것은, Lipschitz constraint가 critic에게 모든 $\Delta_n^T$로부터 $V_n^T$내의 실제 점을 향해 선형 gradient를 제공하기를 강요하기 때문이다.

### ==5.6 Meaningful loss curves and detecting overfitting==

기존의 weight clipping의 중요한 이점은 loss와 샘플 품질에 상관 관계가 있다는 것, 최소를 향해 수렴한다는 것이다. GP가 그 특성을 보존하고 있는지 보여주기 위해서, WGAN-GP를 LSUN 침대 데이터셋게 학습하고 critic의 negative loss를 Fig.5(a)에 그렸다. Generator가 $W(\mathbb{P}_r,\mathbb{P}_g)$를 최소화함에 따라 loss가 수렴됨을 볼 수 있다. 

WGAN-GP는 critic에서의 과적합을 탐지하고 네트워크가 최소화하는 동일한 loss에 대해 과적합을 측정한다.

## ==6. Conclusion==

Gradient penalty를 적용함으로 인해서 기존의 weight clipping이 보이는 문제를 해결할 수 있었다. (undesired behavior) 그리고 더 좋은 성능을 보인다. 

추가적으로 standard GAN에 penalty를 적용해보는 것도 discriminator가 더 스무스한 decision boundaries를 학습하도록 격려되어 학습이 안정화 될 것으로 보기도 한다. 
