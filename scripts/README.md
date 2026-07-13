# 《The Geometry of Auditing Hidden Fidelity under Visible-Score Optimization》 완전 이해 가이드

> **대상 독자**: 확률·통계와 머신러닝을 한두 과목 정도 들었거나, 아직 익숙하지 않은 대학생·대학원생  
> **목표**: 직관에서 출발해 논문의 수식, 정리(theorem), 증명 아이디어, 실험, 한계와 실무적 함의까지 스스로 설명할 수 있게 되는 것  
> **원문**: *The Geometry of Auditing Hidden Fidelity under Visible-Score Optimization* (anonymous submission, 13쪽)  
> **용도**: 개인 학습용 해설

---

## 이 문서를 읽는 방법

이 문서는 네 층으로 구성되어 있다.

1. **직관 층**: 논문이 무슨 문제를 다루는지 그림으로 이해한다.
2. **기초 수학 층**: 확률분포, 기댓값, KL divergence, exponential tilt 등 필요한 배경을 익힌다.
3. **논문 층**: 정의와 정리를 하나씩 읽고, 왜 맞는지 증명 아이디어를 이해한다.
4. **연구자 층**: 실험 설계, 결과 해석, 한계, 새로운 연구 질문까지 검토한다.

처음 읽을 때는 증명 세부를 모두 붙잡으려 하지 않아도 된다. 각 절의 **“한 문장 요약”**과 **“왜 필요한가”**만 먼저 이해한 뒤, 두 번째 읽기에서 수식을 따라가면 된다.

---

# 목차

1. [논문의 핵심을 10분 안에 이해하기](#1-논문의-핵심을-10분-안에-이해하기)
2. [전체 기호표](#2-전체-기호표)
3. [확률과 통계 기초](#3-확률과-통계-기초)
4. [언어모델 샘플링: logits, softmax, temperature](#4-언어모델-샘플링-logits-softmax-temperature)
5. [KL divergence를 완전히 이해하기](#5-kl-divergence를-완전히-이해하기)
6. [Exponential tilt와 KL-regularized optimization](#6-exponential-tilt와-kl-regularized-optimization)
7. [Likelihood ratio, importance sampling, ESS](#7-likelihood-ratio-importance-sampling-ess)
8. [감사 문제의 정확한 설정](#8-감사-문제의-정확한-설정)
9. [희귀 영역과 audit-relevant rarity](#9-희귀-영역과-audit-relevant-rarity)
10. [정리 1: Coverage barrier](#10-정리-1-coverage-barrier)
11. [정리 2: Light-tailed auditability phase](#11-정리-2-light-tailed-auditability-phase)
12. [Free active auditing은 왜 KL과 무관한가](#12-free-active-auditing은-왜-kl과-무관한가)
13. [정리 3: Audit-pressure conservation law](#13-정리-3-audit-pressure-conservation-law)
14. [정리 4: Size-biased block oracle](#14-정리-4-size-biased-block-oracle)
15. [다단계 최적화와 KL 합성](#15-다단계-최적화와-kl-합성)
16. [모니터 감사를 위한 함수공간 배경](#16-모니터-감사를-위한-함수공간-배경)
17. [정리 5: Exact monitor duality](#17-정리-5-exact-monitor-duality)
18. [정리 6: Statistic sufficiency](#18-정리-6-statistic-sufficiency)
19. [Structured monitor complexity와 정리 7](#19-structured-monitor-complexity와-정리-7)
20. [실험 전체 설계](#20-실험-전체-설계)
21. [실험 E1: Coverage barrier](#21-실험-e1-coverage-barrier)
22. [실험 E2: Audit pressure](#22-실험-e2-audit-pressure)
23. [실험 E3: Monitor complexity](#23-실험-e3-monitor-complexity)
24. [실험 E4/E5: 실제 GRPO 정책](#24-실험-e4e5-실제-grpo-정책)
25. [강건성 실험과 추가 결과](#25-강건성-실험과-추가-결과)
26. [논문의 가장 중요한 인사이트](#26-논문의-가장-중요한-인사이트)
27. [논문이 주장하지 않는 것과 한계](#27-논문이-주장하지-않는-것과-한계)
28. [자주 생기는 오해](#28-자주-생기는-오해)
29. [정리별 한 페이지 치트시트](#29-정리별-한-페이지-치트시트)
30. [연습문제와 해설](#30-연습문제와-해설)
31. [권장 학습 순서](#31-권장-학습-순서)
32. [용어 사전](#32-용어-사전)

---

# 1. 논문의 핵심을 10분 안에 이해하기

## 1.1 문제 상황

언어모델을 개선할 때 흔히 다음과 같은 **싸고 자동으로 계산 가능한 점수**를 사용한다.

- LLM judge의 점수
- verifier 또는 process reward model의 점수
- 모델 자신의 confidence
- reward model의 보상

논문은 이런 점수를 **visible score** \(S\)라고 부른다.

하지만 실제로 알고 싶은 것은 보통 다르다.

- 답이 정말 맞는가?
- 추론이 실제로 타당한가?
- 사용자를 속이지 않았는가?
- 장기적으로 원하는 행동을 했는가?

이런 속성은 비싼 인간 라벨, 전문가 판정, 실제 결과가 있어야 알 수 있다. 논문은 이를 **hidden fidelity**라고 부른다.

핵심 문제는 다음이다.

> 모델을 visible score에 맞춰 강하게 최적화한 뒤, hidden fidelity가 실제로 좋아졌는지 확인하려면 얼마나 많은 비용이 드는가?

## 1.2 가장 중요한 그림

최적화 전 모델의 출력분포를 \(\mu\), 최적화 후 분포를 \(q\)라고 하자.

최적화는 높은 \(S\)를 가진 출력에 확률을 집중시킨다. 그 결과, 최적화 후에는 흔하지만 최적화 전에는 매우 드문 영역 \(A\)가 생긴다.

예를 들어:

- 최적화 후 \(q(A)=1/2\): 새 모델 출력의 절반이 \(A\)에 있다.
- 최적화 전 \(\mu(A)=10^{-6}\): 옛 모델에서는 백만 번에 한 번만 \(A\)가 나온다.

감사자는 \(A\)에서 hidden fidelity가 어떻게 변했는지 알아야 한다. 그런데 옛 데이터만 보면 \(A\)를 거의 못 본다.

논문은 이 영역의 희귀도를

\[
r = \log \frac{1}{\mu(A)}
\]

로 정의한다.

그리고 이 하나의 숫자 \(r\)가 세 가지 감사 장벽을 가격화한다고 주장한다.

### 장벽 1: Coverage barrier

baseline \(\mu\)에서만 라벨을 얻으면 필요한 샘플 수가 대략

\[
n_{\text{base}} \asymp e^r
\]

로 증가한다.

### 장벽 2: Audit-pressure barrier

감사자가 샘플링 분포를 바꿔 \(A\)로 갈 수 있지만, 분포 이동에 KL 비용 \(K\)가 든다. 라벨 수를 \(n\)이라 하면

\[
K+\log n \gtrsim r.
\]

즉, **분포를 많이 움직이거나, 샘플을 많이 쓰거나, 둘 중 하나는 해야 한다.**

### 장벽 3: Monitor-complexity barrier

라벨 대신 trace monitor를 쓰더라도, monitor가 최적화가 만든 밀도 변화 \(g=dq/d\mu-1\)를 표현하지 못하면 큰 blind spot이 남는다.

\[
\text{최악의 보이지 않는 fidelity 손실}
=
\rho\,\operatorname{dist}_{L^1(\mu)}(g,V).
\]

여기서 \(V\)는 monitor가 표현할 수 있는 함수들의 공간이다.

## 1.3 KL budget과 희귀도의 관계

특히 논문이 분석하는 **exponential tilt**

\[
q_t(o)
=
\frac{\mu(o)e^{tS(o)}}{\mathbb E_\mu[e^{tS}]}
\]

에서는, 점수분포가 적당히 light-tailed이면

\[
r=d+o(d),
\qquad
d=D(q_t\Vert\mu).
\]

따라서 baseline 라벨 비용은

\[
n_{\text{base}}=e^{d+o(d)}
\]

가 된다.

즉, 최적화 KL budget \(d\)가 1 증가할 때마다 감사 비용이 대략 \(e\approx 2.718\)배씩 증가할 수 있다.

## 1.4 논문의 최종 메시지

이 논문은 “점수 최적화가 항상 해롭다”고 말하지 않는다.

말하고 싶은 것은 다음이다.

> 점수 최적화가 baseline에서 희귀한 영역에 확률을 집중시키면, 실제 품질이 좋아졌든 나빠졌든 **그 사실을 확인하는 것 자체가 어려워진다.**

---

# 2. 전체 기호표

| 기호 | 의미 |
|---|---|
| \(X\) | prompt 또는 context |
| \(O\) | 하나의 observable episode: prompt, trace, answer, diagnostics |
| \(\mu\) | 최적화 전 baseline 분포 |
| \(\mu_x\) | prompt \(x\)가 주어졌을 때 baseline 출력분포 |
| \(q\), \(q_t\) | 최적화 후 분포 |
| \(S(O)\) | 최적화에 쓰는 visible score |
| \(F\) | 비싸게 관측되는 실제 fidelity label |
| \(m(O)=\mathbb E[F\mid O]\) | episode의 조건부 평균 fidelity |
| \(t\) | exponential tilt의 강도 |
| \(\Lambda(t)=\log\mathbb E_\mu[e^{tS}]\) | log-normalizer, cumulant generating function |
| \(d(t)=D(q_t\Vert\mu)\) | 최적화 KL budget |
| \(A\), \(A_\eta\) | 최적화 분포에서 일정 질량 \(\eta\)를 차지하는 고점수 영역 |
| \(r=\log(1/\mu(A))\) | audit-relevant rarity |
| \(\Delta_q(m)\) | 최적화 전후 fidelity 평균 변화 |
| \(L=dq/d\mu\) | likelihood ratio |
| \(g=L-1=dq/d\mu-1\) | density shift |
| \(n\) | 감사 라벨 수 |
| \(K\) | 감사자가 샘플링 분포를 움직이는 KL budget |
| \(V\) | monitor feature의 선형 span |
| \(\rho\) | 숨은 residual의 크기 |
| \(\operatorname{dist}_{L^1(\mu)}(g,V)\) | \(g\)와 monitor 공간 \(V\) 사이의 \(L^1\) 거리 |
| \(\epsilon\) | 허용 추정 오차 |
| \(\delta\) | 실패 확률 |
| \(\eta\) | 최적화 분포에서 tail 영역이 차지하는 질량; 논문에서는 주로 \(1/2\) |

---

# 3. 확률과 통계 기초

## 3.1 확률변수와 확률분포

확률변수는 무작위 결과를 숫자 또는 객체로 표현한 것이다.

언어모델에서는 한 번의 생성 결과 전체를 \(O\)라고 둘 수 있다.

\[
O\sim\mu
\]

는 “\(O\)가 baseline 분포 \(\mu\)에서 생성된다”는 뜻이다.

이산적인 경우 \(\mu(o)\)는 출력 \(o\)가 나올 확률이다. 연속적인 경우에는 확률밀도라고 생각하면 된다.

## 3.2 기댓값

함수 \(h(O)\)의 기댓값은

\[
\mathbb E_\mu[h(O)]
\]

로 쓴다.

이산 경우:

\[
\mathbb E_\mu[h]
=
\sum_o \mu(o)h(o).
\]

연속 경우:

\[
\mathbb E_\mu[h]
=
\int h(o)\,d\mu(o).
\]

기댓값은 “분포에 따라 반복 샘플링했을 때의 장기 평균”이다.

예를 들어 correctness \(F\in\{0,1\}\)라면

\[
\mathbb E[F]=P(F=1)
\]

이므로 기댓값이 곧 정확도다.

## 3.3 조건부 기댓값

논문은

\[
m(O)=\mathbb E[F\mid O]
\]

를 사용한다.

뜻은 “episode \(O\)를 관찰했을 때 기대되는 실제 fidelity”다.

- grading이 완전히 결정론적이면 \(m(O)\)는 거의 \(0\) 또는 \(1\)이다.
- 사람마다 평가가 다르거나 결과에 잡음이 있으면 \(m(O)\)는 \(0\)과 \(1\) 사이일 수 있다.

## 3.4 표본평균과 추정

\(Y_1,\dots,Y_n\)이 독립이고 \(0\le Y_i\le1\)이면

\[
\bar Y_n=\frac1n\sum_{i=1}^nY_i
\]

는 \(\mathbb E[Y]\)의 자연스러운 추정량이다.

Hoeffding 부등식은

\[
P\left(|\bar Y_n-\mathbb E[Y]|>\epsilon\right)
\le
2e^{-2n\epsilon^2}
\]

를 준다.

실패 확률을 \(\delta\) 이하로 만들려면 대략

\[
n
\gtrsim
\frac{1}{\epsilon^2}\log\frac{1}{\delta}
\]

가 필요하다.

이 결과가 논문의 free-active auditing 비용이 \(d\)와 무관한 이유다.

## 3.5 독립성과 “한 번도 못 볼 확률”

어떤 사건 \(A\)가 한 번 샘플에서 나올 확률이 \(p\)라면, \(n\)번 모두 \(A\)를 못 볼 확률은

\[
(1-p)^n.
\]

\(p\)가 작으면

\[
(1-p)^n\approx e^{-np}.
\]

따라서 한 번이라도 볼 확률을 상수 수준으로 만들려면

\[
np\gtrsim1,
\qquad
n\gtrsim\frac1p
\]

가 필요하다.

이 단순한 사실이 coverage lower bound의 핵심이다.

## 3.6 Bernoulli KL

두 Bernoulli 분포 \(\mathrm{Ber}(a)\), \(\mathrm{Ber}(b)\) 사이 KL은

\[
\operatorname{kl}(a\Vert b)
=
a\log\frac{a}{b}
+
(1-a)\log\frac{1-a}{1-b}.
\]

논문의 audit-pressure theorem은 복잡한 전체 transcript를 “tail을 한 번이라도 맞췄는가?”라는 Bernoulli 사건으로 압축한 뒤 이 KL을 사용한다.

## 3.7 Lipschitz 함수

함수 \(f\)가 \(L\)-Lipschitz라는 뜻은

\[
|f(x)-f(y)|\le L|x-y|
\]

이다.

즉, 입력이 조금 바뀔 때 출력이 너무 갑자기 뛰지 않는다.

논문은 hidden fidelity curve가 완전히 제멋대로인 경우뿐 아니라, 꽤 매끄러운 Lipschitz 함수여도 감사 장벽이 남는다는 것을 보인다.

## 3.8 \(O(\cdot)\), \(o(\cdot)\), \(\Theta(\cdot)\)

- \(f(d)=O(g(d))\): \(f\)가 \(g\)보다 상수배 이상 빠르게 커지지 않는다.
- \(f(d)=\Omega(g(d))\): \(f\)가 \(g\)보다 상수배 이상 작아지지 않는다.
- \(f(d)=\Theta(g(d))\): 같은 차수로 증가한다.
- \(f(d)=o(g(d))\): \(f(d)/g(d)\to0\).

예:

\[
r=d+o(d)
\]

는 \(r/d\to1\)이라는 뜻이다. 정확히 같다는 뜻은 아니다.

---

# 4. 언어모델 샘플링: logits, softmax, temperature

## 4.1 Logit

언어모델은 다음 토큰마다 정규화되지 않은 점수 \(z_i\), 즉 logit을 출력한다.

예:

| 토큰 | logit |
|---|---:|
| `4` | 8 |
| `5` | 5 |
| `3` | 3 |

logit 자체는 확률이 아니다.

## 4.2 Softmax

softmax는 logits를 확률로 변환한다.

\[
P(i)=\frac{e^{z_i}}{\sum_j e^{z_j}}.
\]

지수함수 때문에 logit이 조금 더 큰 토큰이 훨씬 높은 확률을 얻는다.

## 4.3 Generation temperature \(\tau\)

temperature \(\tau\)를 적용하면

\[
P_\tau(i)
=
\frac{e^{z_i/\tau}}
{\sum_j e^{z_j/\tau}}.
\]

- \(\tau<1\): 분포가 뾰족해진다. 높은 확률 토큰을 더 자주 고른다.
- \(\tau>1\): 분포가 평평해진다. 드문 토큰도 더 자주 나온다.
- \(\tau\to0\): 거의 greedy decoding.
- \(\tau=1\): 원래 모델 분포.

## 4.4 Temperature와 논문의 tilt parameter \(t\)는 같은가?

**아니다.**

논문의 \(t\)는 전체 trace에 부여된 visible score \(S(O)\)를 얼마나 강하게 반영할지 정한다.

\[
q_t(O)\propto \mu(O)e^{tS(O)}.
\]

생성 temperature \(\tau\)는 매 토큰의 모델 logit을 나누는 decoding 조절값이다.

둘 다 분포를 바꾸지만 방향이 다르다.

- \(t\): “judge score가 높은 전체 trace”를 선호한다.
- \(\tau\): “원래 모델 토큰 확률분포의 sharpness”를 바꾼다.

특수한 경우에 수학적으로 비슷한 꼴을 가질 수 있지만, 논문에서는 분명히 다른 역할이다.

## 4.5 감사자가 temperature를 바꾸는 이유

감사자가 baseline 샘플만 쓰면 optimized tail을 거의 못 볼 수 있다. temperature를 바꾸면 평소에 덜 나오는 trace를 더 자주 생성할 수 있으므로 tail 접근성이 바뀔 수 있다.

하지만 높은 temperature가 반드시 원하는 고점수 tail로 가는 것은 아니다. 논문의 E2 실험에서는 temperature sampler가 큰 sequence-level KL을 쓰면서도 optimized tail에 효율적으로 도달하지 못했다.

---

# 5. KL divergence를 완전히 이해하기

## 5.1 정의

분포 \(q\)가 \(\mu\)에 대해 절대연속일 때

\[
D(q\Vert\mu)
=
\mathbb E_q\left[
\log\frac{dq}{d\mu}
\right].
\]

이산 경우:

\[
D(q\Vert\mu)
=
\sum_o q(o)\log\frac{q(o)}{\mu(o)}.
\]

단위는 자연로그를 쓰면 **nat**다.

## 5.2 직관 1: 예상 log-surprise 차이

\(q\)에서 자주 나오는 샘플을 \(\mu\)가 얼마나 놀랍게 여기는지를 평균낸 값이라고 볼 수 있다.

\(q(o)\)는 큰데 \(\mu(o)\)가 작으면

\[
\log\frac{q(o)}{\mu(o)}
\]

가 크다. 즉 새 모델이 옛 모델에게 매우 희귀한 출력을 자주 만들수록 KL이 커진다.

## 5.3 직관 2: 코딩 비용

\(\mu\)에 맞춘 압축코드로 실제 \(q\) 데이터를 인코딩하면, \(q\)에 맞춘 최적 코드보다 샘플당 평균적으로 얼마나 더 긴 코드가 필요한지를 KL이 나타낸다.

## 5.4 KL의 성질

1. \(D(q\Vert\mu)\ge0\).
2. \(D(q\Vert\mu)=0\) iff \(q=\mu\) 거의 모든 곳에서.
3. 대칭이 아니다.

\[
D(q\Vert\mu)\ne D(\mu\Vert q)
\]

일 수 있다.

4. 일반적인 거리처럼 삼각부등식을 만족하지 않는다.
5. 데이터 처리 부등식이 성립한다.

어떤 관측 함수 \(T\)로 정보를 줄이면

\[
D(q_T\Vert\mu_T)
\le
D(q\Vert\mu).
\]

즉 데이터를 압축하면 두 분포를 구별하기 더 어려워질 뿐이다.

## 5.5 간단한 계산 예

\[
\mu=(0.5,0.5),\qquad q=(0.9,0.1).
\]

그러면

\[
D(q\Vert\mu)
=
0.9\log\frac{0.9}{0.5}
+
0.1\log\frac{0.1}{0.5}.
\]

수치로 약 \(0.368\) nat다.

## 5.6 왜 KL이 최적화 budget인가?

정책 \(q\)가 score를 높이고 싶지만 baseline \(\mu\)에서 너무 멀어지는 것은 막고 싶다고 하자.

\[
\max_q
\left\{
\mathbb E_q[S]
-
\frac1\beta D(q\Vert\mu)
\right\}.
\]

첫 항은 높은 점수를 원하고, 둘째 항은 분포 이동을 벌점으로 준다.

\(\beta\)가 크면 score를 강하게 추구하고, 작으면 baseline에 가깝게 머문다.

이 최적화의 해가 exponential tilt다.

## 5.7 Path KL과 chain rule

여러 단계로 정책을 바꿀 때 전체 trajectory의 likelihood ratio가 단계별로 곱해지면 log-likelihood ratio는 합이 된다.

따라서 path KL도 조건부 KL increment들의 합으로 분해된다.

\[
D(P_R\Vert P_0)
=
\sum_r \kappa_r.
\]

논문은 이 성질을 이용해 여러 최적화 단계를 거친 감사 압력이 누적될 수 있음을 설명한다.

---

# 6. Exponential tilt와 KL-regularized optimization

## 6.1 정의

\[
q_t(o)
=
\frac{\mu(o)e^{tS(o)}}
{Z(t)},
\qquad
Z(t)=\mathbb E_\mu[e^{tS}].
\]

또는

\[
\frac{dq_t}{d\mu}(o)
=
e^{tS(o)-\Lambda(t)},
\qquad
\Lambda(t)=\log Z(t).
\]

각 기호는 다음과 같다.

- \(o\): 한 개의 episode
- \(\mu(o)\): baseline에서 episode \(o\)의 확률
- \(S(o)\): visible score
- \(t\): score 최적화 강도
- \(e^{tS(o)}\): score 보너스
- \(Z(t)\): 확률 합을 1로 만드는 normalizer
- \(q_t(o)\): 최적화 후 확률

## 6.2 왜 지수함수인가?

KL-regularized objective를 풀면 자연스럽게 나온다.

이산 경우를 보자.

\[
\max_{\{q(o)\}}
\sum_o q(o)S(o)
-
\frac1t\sum_o q(o)\log\frac{q(o)}{\mu(o)}
\]

subject to

\[
\sum_o q(o)=1.
\]

라그랑주 승수 \(\lambda\)를 넣으면

\[
\mathcal L
=
\sum_o q(o)S(o)
-
\frac1t\sum_o q(o)\log\frac{q(o)}{\mu(o)}
+
\lambda\left(\sum_o q(o)-1\right).
\]

\(q(o)\)로 미분하여 0으로 두면

\[
S(o)
-
\frac1t\left(\log\frac{q(o)}{\mu(o)}+1\right)
+
\lambda=0.
\]

정리하면

\[
\log\frac{q(o)}{\mu(o)}
=
tS(o)+\text{상수}.
\]

지수화하면

\[
q(o)\propto\mu(o)e^{tS(o)}.
\]

따라서 exponential tilt는 임의로 가정한 것이 아니라 KL-regularized score maximization의 정확한 해다.

## 6.3 \(\Lambda(t)\)의 의미

\[
\Lambda(t)=\log\mathbb E_\mu[e^{tS}]
\]

는 cumulant generating function이다.

미분하면

\[
\Lambda'(t)=\mathbb E_{q_t}[S],
\]

\[
\Lambda''(t)=\operatorname{Var}_{q_t}(S).
\]

즉 \(\Lambda\)는 tilt 아래 score 평균과 분산을 담고 있다.

## 6.4 KL budget 공식

\[
\log\frac{dq_t}{d\mu}
=
tS-\Lambda(t).
\]

따라서

\[
d(t)
=
D(q_t\Vert\mu)
=
t\mathbb E_{q_t}[S]-\Lambda(t).
\]

이 \(d(t)\)가 논문에서 최적화 압력을 나타내는 핵심 좌표다.

## 6.5 Prompt-conditional tilt

실제 RL에서는 prompt 분포를 바꾸지 않고, 각 prompt에서 출력 정책만 바꾼다.

\[
\mu(dx,do)=\pi(dx)\mu_x(do)
\]

라고 하면

\[
\frac{dq_{t,x}}{d\mu_x}(o)
=
e^{tS(x,o)-\Lambda_x(t)}.
\]

prompt marginal \(\pi(x)\)는 그대로다.

논문의 LM 실험은 모든 tilt, threshold, tail을 prompt별로 계산한다. 전역 score bin을 사용하면 문제 난이도 차이와 tilt 방향이 섞여 큰 bias가 생기기 때문이다.

## 6.6 Gaussian 예

baseline score가

\[
S\sim N(0,1)
\]

이면 exponential tilt 후

\[
S\sim N(t,1).
\]

KL은

\[
d=\frac{t^2}{2}.
\]

optimized median tail은

\[
A_t=\{S\ge t\}
\]

이며 \(q_t(A_t)=1/2\)다.

baseline에서는

\[
\mu(A_t)=P(N(0,1)\ge t)
\]

로 매우 작다.

Mills ratio에 의해

\[
\log\frac1{\mu(A_t)}
=
\frac{t^2}{2}
+
\log t
+
O(1)
=
d+\frac12\log d+O(1).
\]

따라서 leading order에서 \(r\approx d\)다.

---

# 7. Likelihood ratio, importance sampling, ESS

## 7.1 Radon–Nikodym derivative

\[
L(o)=\frac{dq}{d\mu}(o)
\]

를 likelihood ratio 또는 density ratio라고 한다.

이산 경우에는 단순히

\[
L(o)=\frac{q(o)}{\mu(o)}.
\]

해석:

- \(L(o)=10\): \(q\)에서 \(o\)가 baseline보다 10배 자주 나온다.
- \(L(o)=1\): 변화 없음.
- \(L(o)=0.1\): 10분의 1로 줄었다.

## 7.2 Density shift \(g\)

논문은

\[
g=L-1=\frac{dq}{d\mu}-1
\]

를 사용한다.

\[
\mathbb E_\mu[g]=0
\]

이다. 왜냐하면

\[
\mathbb E_\mu[L]=\int \frac{dq}{d\mu}d\mu=\int dq=1.
\]

## 7.3 분포 변화와 fidelity shift

\[
\Delta_q(m)
=
\mathbb E_q[m]-\mathbb E_\mu[m].
\]

첫 항을 \(\mu\) 기준으로 바꾸면

\[
\mathbb E_q[m]
=
\mathbb E_\mu[Lm].
\]

따라서

\[
\Delta_q(m)
=
\mathbb E_\mu[(L-1)m]
=
\mathbb E_\mu[gm].
\]

이 식이 논문 전체를 연결한다.

## 7.4 Importance sampling

\(\mu\) 샘플만 있는데 \(q\) 평균을 알고 싶다면

\[
\mathbb E_q[Y]
=
\mathbb E_\mu[LY].
\]

따라서 추정량

\[
\widehat{\mathbb E_q[Y]}
=
\frac1n\sum_{i=1}^nL(O_i)Y_i,
\qquad O_i\sim\mu
\]

를 생각할 수 있다.

하지만 \(L\)이 극단적으로 크면 분산이 폭발한다. baseline에서 매우 드문 tail의 샘플 하나가 거의 모든 가중치를 차지할 수 있다.

이 현상이 coverage barrier와 밀접하다.

## 7.5 Self-normalized importance sampling

실무에서는

\[
\hat\theta
=
\frac{\sum_iL_iY_i}{\sum_iL_i}
\]

를 자주 쓴다.

정규화되지 않은 weight의 scale 문제를 줄이지만, 일반적으로 유한표본 bias가 있다. 논문의 size-biased block oracle에서는 sampling law와 결합하여 특정 평균에 대해 정확한 unbiasedness가 성립한다.

## 7.6 Effective sample size

가중치 \(w_i\)에 대해 흔히

\[
\mathrm{ESS}
=
\frac{(\sum_iw_i)^2}{\sum_iw_i^2}
\]

를 사용한다.

모든 weight가 같으면 ESS는 \(n\)이다. 한 샘플이 거의 모든 weight를 가지면 ESS는 1에 가깝다.

논문의 고-\(d\) 실험에서는 tilt가 몇 개 trace에 집중하면서 prompt별 median ESS가 8 이하로 내려가기도 한다. 이 때문에 score를 직접 본 positive control조차 finite-sample에서 어려워진다.

---

# 8. 감사 문제의 정확한 설정

## 8.1 감사 목표

\[
\Delta_t(m)
=
\mathbb E_{q_t}[m(O)]
-
\mathbb E_\mu[m(O)].
\]

감사자는 이를 오차 \(\epsilon\), 실패확률 \(\delta\)로 추정하고 싶다.

즉 추정량 \(\widehat\Delta\)가

\[
P\left(
|\widehat\Delta-\Delta_t(m)|>\epsilon
\right)
\le\delta
\]

를 만족하게 만들고 싶다.

## 8.2 왜 평균 변화만 추정하는가?

논문은 hidden fidelity 전체 구조를 복원하지 않고, 최적화로 인한 평균 변화라는 가장 기본적인 감사 질문을 다룬다.

이 간단한 목표조차 어려워진다면 더 복잡한 안전성 감사도 어려울 가능성이 크다.

## 8.3 세 가지 감사 oracle

### Baseline-only

라벨이 붙은 샘플을 오직 \(\mu\)에서 얻는다.

### Free active

\(\mu\)와 \(q_t\)에서 직접 라벨을 얻을 수 있다.

### KL-budgeted active

감사자가 새로운 query law \(P\)를 만들 수 있지만

\[
D(P\Vert\mu^n)\le K
\]

라는 전체 transcript KL budget을 지켜야 한다.

마지막 조건은 adaptive sampling도 포함한다. 즉 이전 결과를 보고 다음 query를 바꾸더라도 전체 transcript가 baseline 독립샘플 law에서 얼마나 멀어졌는지를 제한한다.

## 8.4 감사자가 새로운 샘플링 분포를 만드는 이유

baseline-only 감사자는 optimized tail \(A\)를 거의 못 볼 수 있다. 새로운 sampling law \(\nu\)를 만들면 \(A\)를 더 자주 방문할 수 있다.

실제 방법:

- temperature 변경
- prompt suffix 추가
- search 또는 rejection sampling
- best-of-\(k\)
- verifier-guided sampling
- 현재 optimized policy 자체에서 샘플링

하지만 이 움직임은 공짜가 아니다.

1. 계산비용이 든다.
2. 원래 정책에서 멀어진 특이한 query가 될 수 있다.
3. tail에 실제로 접근하지 않고 다른 방향으로만 분포를 바꿀 수 있다.
4. adaptive search까지 포함하면 전체 탐색 과정의 정보 비용을 측정해야 한다.

그래서 논문은 이 비용을 KL budget \(K\)로 추상화한다.

---

# 9. 희귀 영역과 audit-relevant rarity

## 9.1 Tail set \(A_\eta\)

각 prompt \(x\)에서 고점수 threshold \(\tau_{\eta,x}\)를 정해

\[
A_{\eta,x}
=
\{o:S(x,o)\ge\tau_{\eta,x}\}
\]

가 optimized 분포 \(q_x\)에서 질량 \(\eta\)를 갖도록 한다.

\[
q_x(A_{\eta,x})=\eta.
\]

논문은 주로 \(\eta=1/2\)를 사용한다.

즉, “optimized policy의 상위 절반을 구성하는 score 영역”이다.

## 9.2 왜 절반인가?

절반은 임의지만 편리하다.

- optimized policy에 중요도가 상수 수준이다.
- 너무 작은 극단 tail만 보는 것이 아니다.
- baseline에서 얼마나 희귀한지 안정적으로 비교할 수 있다.

다른 고정 \(\eta\in(0,1)\)도 원리는 같다.

## 9.3 Rarity

prompt mixture 전체에서

\[
r_\eta(q;S)
=
-\log \mu(A_\eta).
\]

논문은 prompt별 baseline tail mass를 평균한 뒤 로그를 취한다.

\[
\mu(A_\eta)
=
\mathbb E_{X\sim\pi}
[
\mu_X(A_{\eta,X})
].
\]

이 값은 baseline 한 번 샘플이 optimized region에 들어갈 실제 확률이므로 감사 난이도에 직접 연결된다.

## 9.4 \(r\)와 \(d\)의 역할 차이

- \(d=D(q\Vert\mu)\): 전체 분포 변화량
- \(r=\log1/\mu(A)\): 감사에 중요한 특정 영역의 baseline 희귀도

exponential tilt + light-tailed score에서는 \(r\approx d\).

하지만 일반적인 trained policy에서는 total KL이 length, formatting, entropy 등 score와 무관한 방향에 쓰일 수 있다. 그러므로 \(d\)가 커도 \(r\)은 작을 수 있다.

논문의 GRPO 실험이 바로 이 차이를 보여준다.

---

# 10. 정리 1: Coverage barrier

## 10.1 정리의 직관

두 가능한 세계 \(f_0,f_1\)가 tail \(A\) 밖에서는 완전히 같고, \(A\) 안에서만 fidelity가 다르다고 하자.

감사 샘플이 \(A\)를 한 번도 방문하지 않으면 두 세계의 관측 데이터는 동일하다. 따라서 어떤 추정기도 둘을 구별할 수 없다.

## 10.2 정리의 형태

\(p=P(A)\)이고, \(f_0,f_1\)가 \(A^c\)에서 같으며 감사 목표 값이 \(2\epsilon\)보다 크게 다르면, uniform error probability가 \(\delta\) 이하인 baseline-only auditor는

\[
(1-p)^n\le2\delta
\]

를 만족해야 한다.

\(p\le1/2\)이면

\[
n
\ge
\frac{1}{2p}\log\frac{1}{2\delta}.
\]

## 10.3 증명 아이디어

“No-hit event”를

\[
E_n=\{O_1,\dots,O_n\notin A\}
\]

라고 하자.

이 사건에서 두 세계의 label law는 정확히 같다.

두 세계의 목표값 차이가 \(2\epsilon\)보다 크므로, 하나의 추정값이 두 목표 모두에서 \(\epsilon\) 이내일 수 없다.

따라서 \(E_n\)이 발생하면 적어도 한 세계에서는 실패한다.

두 세계의 실패확률 합이 최소 \(P(E_n)\)이므로, 각 실패확률을 \(\delta\) 이하로 만들려면

\[
P(E_n)=(1-p)^n\le2\delta.
\]

## 10.4 왜 \(1/p\)가 나오는가?

작은 \(p\)에 대해

\[
-\log(1-p)\approx p.
\]

따라서

\[
n
\gtrsim
\frac{\log(1/\delta)}{p}.
\]

즉 tail이 baseline에서 \(p=e^{-r}\)만큼 희귀하면

\[
n\gtrsim e^r.
\]

## 10.5 Audit coverage modulus

논문은 함수 클래스 \(\mathcal F\) 안에서 목표값 차이 \(\gamma\)를 숨길 수 있는 가장 작은 baseline 질량을

\[
C_{\mathcal F,Q}(\gamma)
\]

로 정의한다.

이 값이 작을수록 적은 영역 안에 큰 fidelity 차이를 숨길 수 있어 감사가 어렵다.

정리 1은

\[
n^*_{\text{base}}
\gtrsim
\frac{1}{C_{\mathcal F,Q}(2\epsilon)}
\log\frac1\delta
\]

형태의 lower bound를 준다.

## 10.6 중요한 해석

이것은 단순히 “covariate shift가 있다”는 말보다 구체적이다.

> 중요한 것은 모든 곳의 overlap이 아니라, **optimizer가 질량을 옮긴 바로 그 영역에 baseline label coverage가 있는가**이다.

---

# 11. 정리 2: Light-tailed auditability phase

## 11.1 Weibull-type score

baseline score density를

\[
p_\alpha(s)
\propto
e^{-s^\alpha/\alpha},
\qquad
\alpha>1
\]

로 둔다.

- \(\alpha=2\): Gaussian과 같은 꼴
- \(\alpha>1\): exponential보다 빠르게 감소하는 light tail

## 11.2 정리

고정된 작은 \(\epsilon\)과 \(\delta\)에 대해

\[
\log n^*_{\text{base}}(t,\epsilon,\delta;\mathcal F_L)
=
d(t)+o(d(t)).
\]

즉

\[
n^*_{\text{base}}
=
e^{d(t)+o(d(t))}.
\]

또

\[
d(t)
=
\frac1\alpha
t^{\alpha/(\alpha-1)}
+
o\left(t^{\alpha/(\alpha-1)}\right).
\]

## 11.3 왜 tilt가 특정 위치에 집중하는가?

baseline density와 tilt weight를 곱하면

\[
e^{-s^\alpha/\alpha}e^{ts}
=
e^{ts-s^\alpha/\alpha}.
\]

지수부

\[
\phi_t(s)=ts-\frac{s^\alpha}{\alpha}
\]

를 최대화하는 \(s_t\)를 찾으면

\[
\phi_t'(s)=t-s^{\alpha-1}=0
\]

이므로

\[
s_t=t^{1/(\alpha-1)}.
\]

최적화된 score는 대략 이 위치 근처에 집중한다.

## 11.4 Baseline에서는 왜 \(e^{-d}\)인가?

baseline density를 \(s_t\)에서 보면

\[
e^{-s_t^\alpha/\alpha}
=
e^{-t^{\alpha/(\alpha-1)}/\alpha}.
\]

그 지수의 크기가 바로 leading-order KL \(d(t)\)와 같다.

따라서 optimized-central region의 baseline mass가

\[
e^{-d(t)+o(d(t))}
\]

가 된다.

## 11.5 Lower bound

optimized-central region에 Lipschitz ramp 형태로 두 fidelity 함수를 다르게 만든다.

- \(q_t\) 아래에서는 이 영역이 상수 질량을 가지므로 목표값 차이는 상수다.
- \(\mu\) 아래에서는 질량이 \(e^{-d+o(d)}\)다.

정리 1을 적용하면

\[
n\ge e^{d-o(d)}.
\]

## 11.6 Upper bound

감사자가 tilt의 위치를 알고 있다고 하자.

optimized-central interval을 score bin으로 나누고 baseline label로 각 bin의 fidelity를 추정한다.

각 중요한 bin의 baseline mass가 \(e^{-d+o(d)}\)이므로 모든 bin을 충분히 hit하려면

\[
n\le e^{d+o(d)}
\]

규모면 된다.

lower와 upper가 일치해 sharp phase가 된다.

## 11.7 정리의 범위

- 모든 \(\epsilon\)에 대한 주장이 아니라 고정된 충분히 작은 \(\epsilon\)이다.
- single-context sharp asymptotic이 기본이고 LM 실험은 prompt-conditional finite mixture analogue다.
- light-tail이 핵심이다. score가 cap에 걸려 saturate하면 slope-one 관계가 깨질 수 있다.

---

# 12. Free active auditing은 왜 KL과 무관한가

\(\mu\)와 \(q_t\)에서 각각 직접 라벨 샘플을 얻는다고 하자.

\[
\hat\Delta
=
\frac1n\sum_{i=1}^nY_i^{(q)}
-
\frac1n\sum_{i=1}^nY_i^{(\mu)}.
\]

각 \(Y\in[0,1]\)이므로 Hoeffding으로

\[
n
=
O\left(
\epsilon^{-2}\log\frac1\delta
\right)
\]

이면 충분하다.

중요한 점:

- \(q_t\)에서 샘플링하면 optimized tail을 절반 정도의 빈도로 자연스럽게 본다.
- baseline에서 그 tail이 얼마나 희귀한지는 더 이상 문제되지 않는다.

하지만 이 oracle은 “\(q_t\) 샘플을 공짜로 얻을 수 있다”는 강한 가정이다. 실제 배포에서는 원하는 query distribution을 구현하는 비용이 있다. 이 간극이 Section 4의 audit-pressure barrier로 이어진다.

---

# 13. 정리 3: Audit-pressure conservation law

## 13.1 두 세계

tail \(A\) 안에서만 fidelity가 다르다고 하자.

\[
m_0\equiv1,
\qquad
m_1=1-b\mathbf1_A.
\]

optimized distribution에서 \(q(A)=\eta\)이므로 목표 평균 차이는 \(b\eta\)다.

## 13.2 Hit event

감사 과정에서 한 번이라도 \(A\)를 query한 사건을

\[
H=\{\text{some query hits }A\}
\]

라고 한다.

baseline 독립샘플 \(n\)개라면

\[
p_n
=
P_{\mu^n}(H)
=
1-(1-p)^n,
\qquad
p=\mu(A).
\]

## 13.3 전체 transcript KL budget

감사자가 adaptive하게 query해도, 각 세계에서 전체 transcript law \(P_m\)가

\[
D(P_m\Vert\mu^n)\le K
\]

를 만족한다고 하자.

데이터 처리 부등식으로 전체 transcript를 \(H\in\{0,1\}\)로 압축하면

\[
\operatorname{kl}(P_m(H)\Vert p_n)\le K.
\]

즉 baseline에서 매우 드문 hit event를 감사자가 높은 확률로 만들려면 KL을 지불해야 한다.

## 13.4 Exact minimax risk

정리는

\[
\frac{b\eta}{2}(1-\rho_K)
\]

가 exact minimax risk라고 말한다. 여기서

\[
\rho_K
=
\sup\{\rho:\operatorname{kl}(\rho\Vert p_n)\le K\}.
\]

\(\rho_K\)는 KL budget \(K\)로 hit probability를 최대 어디까지 끌어올릴 수 있는지를 나타낸다.

## 13.5 Conservation law

상수 확률로 \(A\)를 hit하려면 대략

\[
K+\log n
\gtrsim
\log\frac1{\mu(A)}
=
r.
\]

왜 그런가?

baseline에서

\[
p_n\le np.
\]

상수 \(\rho\)까지 hit probability를 올리는 Bernoulli KL은 대략

\[
\operatorname{kl}(\rho\Vert p_n)
\approx
\rho\log\frac1{p_n}.
\]

따라서

\[
K
\gtrsim
\log\frac1{np}
=
\log\frac1p-\log n.
\]

정리하면

\[
K+\log n\gtrsim\log\frac1p.
\]

## 13.6 경제학적 해석

희귀도 \(r\)라는 총 가격이 있다.

- 라벨을 \(e^r\)개 쓰면 \(K\approx0\)으로 가능하다.
- query law를 KL \(r\)만큼 움직이면 \(n=O(1)\)로 가능하다.
- 중간 조합도 가능하다.
- 둘 다 거의 0으로 만들 수는 없다.

## 13.7 Gaussian corollary

\[
S\sim N(0,1),\qquad q_t=N(t,1),\qquad d=t^2/2.
\]

optimized median tail \(A_t=\{S\ge t\}\)에 대해

\[
r
=
d+\frac12\log d+O(1).
\]

따라서

\[
K+\log n\gtrsim d.
\]

“Optimization pressure creates audit pressure at rate one”의 정확한 의미다.

## 13.8 Smooth class에서도 유지

두 점짜리 극단적인 fidelity 함수뿐 아니라 Lipschitz smooth class에서도 같은 lower bound가 유지된다.

즉 장벽이 불연속적인 adversarial label에만 의존하는 인공물은 아니다.

---

# 14. 정리 4: Size-biased block oracle

## 14.1 목적

정리 3은 lower bound다. 실제로 그 경계에 가까이 도달하는 sampling law가 있는지 보여야 tightness를 주장할 수 있다.

## 14.2 Block 구성

한 block에 baseline 샘플 \(m\)개가 있다고 하자.

\[
O_1,\dots,O_m\sim\mu.
\]

각 likelihood ratio를

\[
L_i=\frac{dq}{d\mu}(O_i)
\]

라고 한다.

block의 평균 likelihood ratio는

\[
\bar L_m=\frac1m\sum_{i=1}^mL_i.
\]

새 block law를

\[
\frac{dP_m^\star}{d\mu^m}
=
\bar L_m
\]

로 정의한다.

## 14.3 직관

이 law는 다음 mixture와 같다.

> block의 한 좌표는 \(q\)에서 뽑고, 나머지 \(m-1\)개는 \(\mu\)에서 뽑되, 어느 좌표가 \(q\)인지는 균등하게 숨긴다.

따라서 block마다 적어도 하나의 optimized-like sample을 넣으면서, 어느 하나의 샘플을 완전히 \(q\)로 바꾸는 것보다 KL 비용을 \(\log m\)만큼 분산시킬 수 있다.

## 14.4 KL 비용

정리는

\[
D(P_m^\star\Vert\mu^m)
\le
\mathbb E_q\log(1+L/m)
\]

를 보인다.

log-likelihood ratio가 \(d\) 근처에 집중하면

\[
D(P_m^\star\Vert\mu^m)
=
[d-\log m]_++o(d).
\]

여기서

\[
[x]_+=\max(x,0).
\]

즉 block size \(m\)를 \(e^d\)까지 키우면 KL 비용을 거의 0으로 만들 수 있고, 작은 \(m\)이면 남은 차이만큼 KL을 낸다.

## 14.5 추정량

block 안에서

\[
\hat q_m
=
\frac{\sum_iL_iY_i}{\sum_iL_i}
\]

를 사용한다.

size-biased block law와 결합하면 목표 \(q\)-평균에 대해 정확히 unbiased가 된다.

여러 block을 평균하면

\[
B=O(\epsilon^{-2}\log1/\delta)
\]

개 block으로 정확도를 달성한다.

## 14.6 경계와의 관계

block당

\[
K/B+\log m\gtrsim d.
\]

전체 샘플 \(N=Bm\)이므로 상수 \(\epsilon,\delta\)에서 대략

\[
K+\log N=\Theta(d).
\]

lower bound와 같은 phase다.

## 14.7 실무적 의미

이 oracle은 반드시 실용적인 알고리즘이라는 뜻이 아니다. 정확한 likelihood ratio \(L\)가 필요하며 event-tilted law도 구현하기 어려울 수 있다.

역할은 다음이다.

> 정보이론적으로 conservation frontier가 단순한 proof artifact가 아니라 달성 가능한 경계임을 보인다.

---

# 15. 다단계 최적화와 KL 합성

정책이 여러 단계로 변경된다고 하자.

trajectory law의 시작을 \(P_0\), 끝을 \(P_R\), 최종 output law를 각각 \(\mu,q\)라고 하자.

최종 output은 trajectory의 함수이므로 데이터 처리 부등식으로

\[
D(q\Vert\mu)
\le
D(P_R\Vert P_0).
\]

단계별 conditional KL increment를 \(\kappa_r\)라고 하면

\[
D(P_R\Vert P_0)
=
\sum_r\kappa_r.
\]

단, audit-relevant tail rarity가 전체 path KL과 동일하게 증가하려면 각 단계의 변화가 최종 density ratio에서 상쇄되지 않아야 한다.

실무적 해석:

- exponential tilt에 가까운 score-directed training이면 단계별 KL을 누적해 audit pressure를 예측할 수 있다.
- 일반 정책에서는 total KL만 믿지 말고 최종 tail rarity \(r\)을 직접 측정하는 것이 안전하다.

---

# 16. 모니터 감사를 위한 함수공간 배경

## 16.1 Monitor feature span \(V\)

monitor가 사용하는 feature들을

\[
v_1(O),\dots,v_k(O)
\]

라고 하자.

이들의 선형결합 전체를

\[
V=\operatorname{span}\{1,v_1,\dots,v_k\}
\]

라고 한다.

예:

- 길이
- 숫자 개수
- 특정 표현 빈도
- embedding coordinate
- linear probe output

## 16.2 보이지 않는 residual

함수 \(h\)가 monitor에 대해 invisible하다는 것은

\[
\mathbb E_\mu[h(O)v(O)]=0
\qquad
\forall v\in V
\]

라는 뜻이다.

이를

\[
h\in V_\infty^\perp
\]

라고 쓴다.

이런 \(h\)는 monitor feature와 상관이 없으므로 baseline 데이터에서 monitor로 탐지하기 어렵다.

## 16.3 Fidelity class

논문은

\[
m=\frac12+\rho h,
\qquad
\|h\|_\infty\le1,
\qquad
h\perp V
\]

를 고려한다.

\(\rho\le1/2\)이면 \(m\in[0,1]\)이다.

즉 baseline에서는 monitor가 아무 신호도 못 보는 bounded residual이지만, 분포가 \(q\)로 바뀌면 평균이 크게 달라질 수 있는지를 묻는다.

## 16.4 \(L^1\) 거리

\[
\|f\|_{L^1(\mu)}
=
\mathbb E_\mu|f(O)|.
\]

함수 \(g\)와 공간 \(V\) 사이 거리는

\[
\operatorname{dist}_{L^1(\mu)}(g,V)
=
\inf_{v\in V}
\mathbb E_\mu|g-v|.
\]

monitor feature로 density shift \(g\)를 얼마나 잘 근사할 수 있는지를 나타낸다.

## 16.5 \(L^1/L^\infty\) duality

기본 사실:

\[
\|f\|_1
=
\sup_{\|h\|_\infty\le1}
\mathbb E[fh].
\]

최적의 \(h\)는 대략 \(\operatorname{sign}(f)\)다.

subspace \(V\)에 대한 거리도

\[
\operatorname{dist}_{L^1}(g,V)
=
\sup_{\substack{\|h\|_\infty\le1\\h\perp V}}
\mathbb E[gh]
\]

라는 dual form을 가진다.

정리 5는 정확히 이 사실을 감사 문제에 적용한다.

---

# 17. 정리 5: Exact monitor duality

## 17.1 정리

\[
\inf_{m\in\mathcal M_{\rho,V}}
\Delta_q(m)
=
-\rho\,
\operatorname{dist}_{L^1(\mu)}(g,V),
\]

여기서

\[
g=\frac{dq}{d\mu}-1.
\]

## 17.2 전개

\[
m=\frac12+\rho h.
\]

그러면

\[
\Delta_q(m)
=
\mathbb E_\mu[gm].
\]

\(\mathbb E_\mu[g]=0\)이므로 상수 \(1/2\) 항은 사라진다.

\[
\Delta_q(m)
=
\rho\mathbb E_\mu[gh].
\]

\(h\)는 \(\|h\|_\infty\le1\), \(h\perp V\) 조건을 만족한다.

dual characterization에 의해 가능한 가장 음의 값은

\[
-\rho\operatorname{dist}_{L^1}(g,V).
\]

## 17.3 직관

monitor가 잘 해야 하는 것은 baseline correctness를 예측하는 것이 아니다.

monitor가 진짜로 필요한 능력은

> **최적화가 확률질량을 어디에서 빼서 어디로 옮겼는지**, 즉 \(g\)를 표현하는 것

이다.

\(g\)와 monitor space 사이 거리가 크면, monitor feature와 baseline에서 완전히 무상관이면서 optimized distribution에서 큰 fidelity loss를 일으키는 residual이 존재한다.

## 17.4 Constructive worst-case residual

\(v^\star\in V\)가 \(g\)의 best \(L^1\) approximation이라면

\[
h(O)\approx-\operatorname{sign}(g(O)-v^\star(O))
\]

가 worst-case residual 역할을 한다.

이 residual은 monitor가 설명하지 못하는 density-shift 방향에 맞춰져 있다.

## 17.5 주의

이 정리는 worst-case bounded fidelity residual에 대한 정확한 결과다.

실제 correctness label이 반드시 이 worst-case residual과 같다는 뜻은 아니다.

실험 E3도 실제 correctness 손실을 직접 증명한 것이 아니라, 학습된 monitor가 가진 **certified worst-case blind spot**을 측정한다.

---

# 18. 정리 6: Statistic sufficiency

## 18.1 Statistic \(T\)

monitor가 전체 \(O\)를 보지 않고 요약 통계

\[
T=T(O)
\]

만 본다고 하자.

예:

- 길이
- embedding
- 별도 모델의 출력
- 몇 비트짜리 코드

\(T\)가 생성하는 정보집합을 \(\sigma(T)\)라고 한다.

## 18.2 정리

\(T\)-invisible residual에 대한 worst-case loss는

\[
-\rho
\operatorname{dist}_{L^1(\mu)}
\left(
g,L^1(\sigma(T))
\right).
\]

그리고 모든 bounded invisible residual을 완전히 감사할 수 있는 필요충분조건은

\[
g
\text{가 }T\text{의 함수이다.}
\]

즉

\[
g(O)=\phi(T(O))
\]

인 어떤 함수 \(\phi\)가 존재해야 한다.

## 18.3 직관

두 episode가 같은 \(T\) 값을 가지는데 likelihood ratio \(g\)는 다르다면, monitor는 그 둘을 구분할 수 없다.

그 차이를 따라 fidelity residual을 숨길 수 있다.

반대로 \(g\)가 \(T\)만으로 완전히 결정되면 monitor는 최적화의 density shift를 모두 알고 있다.

## 18.4 왜 score \(S\)를 보면 충분한가?

exponential tilt에서는

\[
g
=
e^{tS-\Lambda(t)}-1.
\]

즉 \(g\)가 \(S\)의 함수다.

따라서 \(S\)를 정확히 보는 monitor는 이론적으로 density shift를 완전히 표현할 수 있다.

실험의 analytic S-oracle distortion이 0인 이유다.

---

# 19. Structured monitor complexity와 정리 7

## 19.1 비트 수만으로는 장벽이 약하다

놀랍게도 \(g\)라는 스칼라를 몇 단계로 quantize하면 적은 비트로도 \(L^1\) 오차를 줄일 수 있다.

Gaussian tilt에서는 고정 distortion에 필요한 level 수가

\[
M=\Theta(\sqrt d)
\]

라서 bit 수는

\[
\log_2M
=
\frac12\log_2d+O(1)
\]

에 불과하다.

따라서 “monitor output bit가 적으니 어렵다”는 식의 lower bound는 의미가 약하다.

## 19.2 Structured complexity

실제 monitor는 arbitrary quantizer가 아니라 특정 구조를 가진다.

- 저차 다항식
- 제한된 feature dictionary
- 작은 neural network
- generic embedding 위의 probe

이런 family가 optimized tail에 충분한 질량을 배치할 수 있는지가 핵심이다.

## 19.3 Tail-leakage coefficient

\[
\beta_V(A)
=
\sup_{v\in V,v\ne0}
\frac{\mathbb E_\mu[|v|\mathbf1_A]}
{\mathbb E_\mu|v|}.
\]

해석:

> \(V\) 안의 함수가 전체 \(L^1\) 질량 중 얼마를 희귀 tail \(A\)에 집중시킬 수 있는가?

\(\beta_V(A)\)가 작으면 monitor family가 tail에 반응하기 어렵다.

## 19.4 일반 lower bound

논문은

\[
\operatorname{dist}_{L^1(\mu)}(g,V)
\ge
\frac{
q(A)-\mu(A)-2\beta_V(A)
}{
1+\beta_V(A)
}
\]

를 보인다.

\(q(A)\)가 상수이고 \(\mu(A)\to0\), \(\beta_V(A)\to0\)이면 distortion이 상수 이하로 내려가지 않는다.

## 19.5 Structured tail-leakage principle

복잡도 \(R\)에 따른 family \(V_R\)가 있고

\[
\beta_{V_R}(A_d)
\le
\exp\{\psi(R)-d/2+o(d)\}
\]

이면

\[
\psi(R)\le(1/2-\xi)d
\]

인 동안 distortion이 상수로 남는다.

즉 family capacity가 \(d\)에 비례하여 커져야 tail을 표현할 수 있다.

## 19.6 Gaussian polynomial monitor

\[
V_r=\operatorname{span}\{1,X,\dots,X^r\}.
\]

Gaussian hypercontractivity로

\[
\beta_{V_r}(\{X\ge t\})
\le
3^r\sqrt{\bar\Phi(t)}.
\]

Gaussian tail에서

\[
\sqrt{\bar\Phi(t)}
\approx
e^{-d/2+o(d)}.
\]

따라서

\[
\beta\lesssim e^{r\log3-d/2+o(d)}.
\]

tail leakage를 상수로 만들려면 \(r=\Omega(d)\)가 필요하다.

## 19.7 정리 7

고정 distortion threshold \(\tau\)에 대해 필요한 polynomial degree를 \(r^*_{L^1}(d,\tau)\)라 하면

\[
cd
\le
r^*_{L^1}(d,\tau)
\le
Cd
\]

이다.

즉

\[
r^*=\Theta(d).
\]

현재 증명 상수는 대략 lower \(1/(2\log3)\), upper \(2e\) 수준이며 정확한 threshold constant는 주장하지 않는다.

## 19.8 Upper bound와 Hermite expansion

Gaussian에서 \(g_t\)를 Hermite polynomial basis로 전개할 수 있다.

degree \(r\)에서 자른 residual의 \(L^2\) 질량은

\[
\sum_{k>r}\frac{t^{2k}}{k!}
\]

형태다.

Poisson tail과 비슷한 구조 때문에 \(r\)이 \(d=t^2/2\)의 상수배가 되면 residual이 작아진다.

---

# 20. 실험 전체 설계

## 20.1 왜 training-free tilt인가?

KL-regularized optimum이 정확히 exponential tilt이므로, 이미 생성한 trace pool에서

\[
w_i\propto e^{tS_i}
\]

로 reweight하면 유한 empirical pool 위의 최적분포를 정확히 구현할 수 있다.

장점:

- RL training instability가 없다.
- \(d(t)\), \(L\), \(g\), tail set을 정확히 계산할 수 있다.
- 하나의 generation pool로 여러 \(d\)를 연속적으로 탐색할 수 있다.

## 20.2 데이터

- Base model: Qwen2.5-7B-Instruct
- MATH-500 중 level 4–5
- 정답이 명확한 숫자 또는 단일 분수인 203개 문제
- prompt당 1,024개 trace
- 총 207,872 traces
- generation: temperature 1.0, top-p 1.0

## 20.3 Visible score

같은 base model을 LLM judge로 사용하되 정답지는 주지 않는다.

judge가 Yes/No를 답할 때 첫 토큰 log-odds를

\[
S
=
\log P(\mathrm{Yes})
-
\log P(\mathrm{No})
\]

로 사용하고 \(\pm16\)에 clip한다.

## 20.4 Hidden fidelity

추출한 final answer를 ground-truth와 programmatic하게 비교한 correctness다.

pool 통계:

- baseline accuracy: 0.571
- score AUC for correctness: 0.851
- 최고 score decile의 error rate: 0.062
- level-5 only에서는 0.136

즉 judge는 꽤 좋지만 완벽하지 않아 실제 proxy–truth gap이 있다.

## 20.5 Pool ceiling

prompt당 \(N=1024\)개 finite pool이므로 한 trace에 모든 질량이 몰리면 최대 KL은 대략

\[
d_{\max}\approx\log N\approx6.93.
\]

이 이상은 pool이 tilt를 표현할 수 없다.

---

# 21. 실험 E1: Coverage barrier

## 21.1 목표

각

\[
d\in\{0.5,\dots,6\}
\]

에서 true fidelity shift를 full labeled pool로 계산하고, 각 auditor가 오차

\[
\epsilon=0.025,\qquad\delta=0.2
\]

를 달성하는 최소 label count를 측정한다.

## 21.2 Baseline-only auditor

- prompt-conditional score bins 사용
- tilt와 bin mass는 알고 있음
- \(\mu\)-label로 각 bin의 fidelity curve 추정
- label이 없는 bin에는 prior \(1/2\) 사용

중요한 점: global score bin은 문제 난이도와 score를 섞어 asymptotic bias가 \(+0.11\sim+0.30\) 생겼다. prompt-conditional bin은 bias를 0.002 이하로 유지했다.

## 21.3 Free-active auditor

- \(q_t\)에서 \(n\) labels
- \(\mu\)에서 \(n\) labels
- 차이의 표본평균
- 논문은 \(n\)을 arm당 비용으로 보고하므로 총 labels는 \(2n\)

## 21.4 결과

- baseline-only:
  - \(d=0.5\): 1,024 labels
  - \(d=6\): 370,728 labels
- free-active:
  - 전체 \(d\)에서 약 1,024–1,448 labels

\(d\ge2\)에서 baseline \(\log n\) slope는 약 1.4로 이론의 slope-one phase와 질적으로 일치한다.

## 21.5 Judge hacking curve

judge score:

\[
8.2\to12.3
\]

로 계속 상승한다.

true accuracy는:

- baseline 0.571
- \(d\approx1\)에서 0.631로 최고
- 이후 0.619로 하락

즉 더 강한 judge optimization이 실제 correctness를 계속 개선하지는 않는다.

best-of-\(n\)도 \(n=16\) 부근에서 peak 후 \(n=1024\)까지 하락한다.

## 21.6 핵심 해석

가장 감사가 필요한 overoptimization 구간에서 baseline 감사가 가장 비싸진다.

---

# 22. 실험 E2: Audit pressure

## 22.1 실용 sampler 7개

Temperature:

\[
\tau\in\{0.7,0.85,1.15,1.3\}
\]

Prompt-tilt suffix:

- confident
- verified
- persuasive

각 sampler의 per-draw KL을 generation-time log probability와 baseline teacher-forcing으로 측정한다.

## 22.2 Tail

\[
d\in\{2,4,6\}
\]

에서 prompt별 \(q_t\)-median score 이상을 \(A\)로 둔다.

따라서 \(q_t(A)\approx1/2\).

## 22.3 샘플 비용

sampler가 \(A\)를 hit할 확률 \(\nu(A)\)를 측정하고, 원하는 confidence로 한 번 이상 hit하는 데 필요한 \(n\)을 계산한다.

각 방법을 \((K,\log n)\) 평면에 그린다.

## 22.4 결과

1. 모든 측정점이

\[
K+\log n
\ge
\log\frac1{\mu(A)}
\]

frontier 위에 있었다.

2. size-biased oracle은 \(d=6\)에서 frontier에 0.02 nat 이내로 도달했다.

3. practical samplers는 frontier보다 상당히 위에 있었다.

4. 높은 temperature는 큰 sequence-level KL을 쓰지만 tail 접근 개선은 제한적이었다.

5. prompt suffix는 분포를 크게 바꿔도 원하는 optimized score tail로 안정적으로 이동하지 않았다.

## 22.5 인사이트

“분포를 많이 바꾸는 것”과 “감사에 필요한 방향으로 바꾸는 것”은 다르다.

KL budget은 비용이고, tail hit probability가 효용이다. 좋은 active auditor는 KL당 tail 접근 효율이 높아야 한다.

---

# 23. 실험 E3: Monitor complexity

## 23.1 질문

judge score \(S\)를 볼 수 없는 trace-only monitor가 density shift \(g\)를 얼마나 잘 근사하는가?

## 23.2 Monitor ladder

### Surface features

11개 표면 특징의 linear, degree-2, degree-3 polynomial model.

### Frozen embedding

Qwen2.5-1.5B embedding 위의:

- linear probe
- MLP width 64–1024

### Fine-tuned LM

Qwen2.5-1.5B LoRA fine-tuning + scalar regression head.

모두 raw trace만 보고, \(S\)나 judge에는 접근하지 않는다.

학습 target은 full pool에서 정확히 계산된 \(g\), loss는 \(L^1\).

## 23.3 평가량

\[
\frac{
\operatorname{dist}_{L^1}(g,V_R)
}{
\mathbb E_\mu|g|
}
\]

에 대응하는 held-out relative \(L^1\) error를 측정한다.

- 0: 완벽
- 1에 가까움: density shift를 거의 못 설명

## 23.4 결과

trace-only ladder는 \(d\in[1,6]\)에서 약 0.47–0.88의 큰 distortion을 유지했다.

LoRA fine-tuned LM도 큰 blind spot을 남겼다.

\(S\)를 추가 feature로 준 learned control은:

- \(d=1\): 0.25
- \(d=2\): 0.37

까지 크게 개선되었다.

analytic S-oracle은 이론적으로 0이다.

## 23.5 고-\(d\)에서 control도 나빠지는 이유

\(d\ge4\)에서는 empirical tilt가 prompt당 극소수 trace에 집중하고 median ESS가 8 이하가 된다.

\(g\)가 극단적인 spike를 가지므로 finite-sample regression이 어려워진다. 이는 정리 6과 모순이 아니다. 정리 6은 population-level sufficiency이고, 실험은 finite-sample learning이다.

## 23.6 Worst-case residual 검증

최고 monitor \(v^\star\)에 대해

\[
h=\operatorname{sign}(g-v^\star)
\]

와 비슷한 residual을 구성하면 predicted fidelity shift와 실제 계산 shift가 거의 일치했다.

예:

- \(d=2\), \(\rho=1/2\)
- 0.458 vs. 0.456

이는 duality의 magnitude가 실제로 checkable하다는 것을 보여준다.

## 23.7 주의

이 실험은 correctness label이 monitor에게 안 보인다고 직접 보인 것이 아니다.

“이 monitor가 못 보는 bounded residual 중 큰 fidelity shift를 만드는 것이 존재한다”는 worst-case certification이다.

---

# 24. 실험 E4/E5: 실제 GRPO 정책

## 24.1 왜 필요한가?

training-free exact tilt는 이상화된 KL-optimal policy다.

실제 RL인 GRPO가 같은 geometry를 보이는지 확인해야 한다.

## 24.2 설정

- 같은 judge score로 GRPO fine-tuning
- LoRA adapters
- 여러 KL coefficient와 learning rate
- 22 checkpoints + base anchor
- measured policy KL:

\[
\hat d
=
\mathbb E_\pi[\log\pi-\log\mu]
\in[0.16,256].
\]

## 24.3 결과 1: Goodhart curve 재현

accuracy:

- base anchor: 0.493
- \(\hat d\approx30\): 약 0.579 peak
- \(\hat d\approx90\): 약 0.52로 하락

judge score는 계속 올라 14.7에 도달했다.

즉 exact tilt뿐 아니라 실제 RL에서도 overoptimization이 나타났다.

## 24.4 결과 2: Total KL과 audit-relevant pressure의 분리

각 checkpoint에서 policy의 절반 질량을 차지하는 judge-score tail \(A_\pi\)를 정의하고 baseline rarity를 측정했다.

GRPO points는 matched total KL에서 tilt curve보다 훨씬 낮은 rarity를 보였다.

예:

- GRPO \(\hat d=3.5\)에서 rarity 약 1.16
- tilt는 같은 KL에서 rarity 약 4.4

해석:

GRPO는 KL을 judge-score 방향에만 쓰지 않는다.

- 길이
- 형식
- entropy
- 기타 policy drift

에도 budget을 쓴다.

## 24.5 Tilt가 의미하는 것

실험 범위에서 exact tilt는 matched total KL 좌표에서 audit-relevant tail concentration의 empirical worst-case envelope처럼 행동했다.

하지만 논문은 arbitrary event에 대해 tilt가 모든 분포보다 tail concentration을 최대화한다고 일반적으로 주장하지는 않는다.

## 24.6 실무적 결론

일반 trained policy에서는

\[
D(q\Vert\mu)
\]

만으로 감사 비용을 예측하지 말라.

대신 실제 optimized constant-mass region의

\[
r=\log1/\mu(A)
\]

를 직접 측정하라.

---

# 25. 강건성 실험과 추가 결과

## 25.1 \(\epsilon\) 변화

\[
\epsilon\in\{0.02,0.025,0.03\}
\]

에서도 phase shape는 유지되었다.

\(d=6\) baseline labels:

- 441,000
- 371,000
- 312,000

active auditor는 예상대로 \(\epsilon^{-2}\)에 따라 변했다.

## 25.2 약한 judge

Qwen2.5-1.5B judge:

- top-decile error 0.134
- AUC 0.764
- correctness가 \(d\)와 함께 단조 하락
- \(d\approx3.5\) 이후 fidelity shift가 음수

다만 \(\Delta_t\)가 0을 통과해 target 자체가 trivial한 지점이 많아 main coverage claim은 7B judge pool에 집중한다.

## 25.3 더 강한 base model

Qwen2.5-14B:

- top-decile error 0.025
- baseline accuracy 0.624
- baseline complexity 1,218 \(\to\) 131,072
- active 약 1,200

proxy–truth gap이 작아져도 geometry barrier는 유지되었다.

## 25.4 Mixed GSM8K+MATH pool

- 786 prompts
- 805,000 traces
- accuracy 0.811
- top-decile error 0.009

baseline complexity:

\[
13,777\to2,097,152
\]

active는 900 이하.

작은 proxy gap에서도 coverage phase가 보였다.

## 25.5 대체 hidden label: step-faithfulness

14B step checker가 생성한 model-derived faithfulness label은 judge score와 함께 계속 상승했다.

- correctness shift는 peak 후 감소
- model-derived faithfulness shift는 \(+0.11\to+0.17\)

교훈:

> hidden label 자체가 또 다른 모델 proxy이면 optimized judge와 co-inflate할 수 있다.

그래서 ground truth로 programmatic correctness를 사용한 것이다.

## 25.6 Score dependence

같은 pool을 다음 score로 tilt했다.

- judge score
- per-token log probability
- random noise
- completion length

judge, log probability, random은 rarity–KL slope-one curve에 가깝게 겹쳤다.

length는 token cap saturation 때문에 light-tail mechanism에서 벗어났다.

그러나 fidelity shift는 score마다 완전히 달랐다.

- judge: 양수
- random: 약 0
- length: 큰 음수

---

# 26. 논문의 가장 중요한 인사이트

## 26.1 감사 비용과 실제 효과는 별개다

최적화가 correctness를 올릴 수도, 내릴 수도, 거의 안 바꿀 수도 있다.

하지만 optimized region이 baseline에서 희귀하면 **무슨 일이 일어났는지 알아내는 비용**은 커진다.

## 26.2 “한 희귀 영역, 세 장벽”

같은 영역 \(A\)가:

1. baseline label coverage를 깨뜨리고,
2. active sampler가 지불해야 할 KL을 만들고,
3. monitor가 표현해야 할 density shift를 날카롭게 만든다.

## 26.3 Exponential tilt에서 KL은 감사 가격표다

light-tailed score-directed optimization에서

\[
r=d+o(d).
\]

따라서

\[
\log n_{\text{base}}\approx d.
\]

KL은 단순한 regularization coefficient가 아니라 baseline 감사 비용의 log-scale 예측값이 된다.

## 26.4 일반 정책에서는 rarity를 직접 측정해야 한다

GRPO처럼 KL을 여러 방향에 쓰는 정책에서는 total KL이 audit pressure를 과대평가할 수 있다.

직접 필요한 것은

\[
r=\log1/\mu(A).
\]

## 26.5 Active auditing은 “공짜 q-label”이 아니다

optimized policy에서 직접 샘플을 얻는 것이 항상 가능하거나 저렴하지 않다.

그 sampler를 구성하는 distributional work 자체를 KL로 가격화해야 한다.

## 26.6 Monitor는 correctness predictor보다 density-shift tracker여야 한다

worst-case auditing에서 중요한 것은 baseline에서 fidelity를 잘 예측하는 능력이 아니라

\[
g=dq/d\mu-1
\]

를 얼마나 잘 설명하는가다.

## 26.7 비트 수보다 구조가 중요하다

적은 bit로 likelihood ratio를 quantize할 수 있으므로 단순 output-capacity lower bound는 약하다.

실제 장벽은 monitor family가 희귀 optimized tail에 질량을 집중할 수 있는 구조적 능력이다.

---

# 27. 논문이 주장하지 않는 것과 한계

## 27.1 Visible-score optimization은 항상 해롭지 않다

실험에서도 correctness는 처음에는 상승했다.

논문은 fidelity 변화의 부호를 예측하지 않는다.

## 27.2 모든 monitor가 실패한다고 하지 않는다

정리 5–7은 특정 feature span 또는 structured family와 worst-case residual에 대한 결과다.

좋은 monitor가 \(g\)를 충분히 포착하면 barrier를 줄일 수 있다.

## 27.3 모든 KL이 audit-relevant하지 않다

exponential tilt에서는 \(r\approx d\)지만, 일반 trained policy에서는 분리될 수 있다.

## 27.4 Minimax 결과다

정리들은 최악의 가능한 fidelity residual을 고려한다.

실제 task의 correctness가 그 최악 방향과 얼마나 정렬되는지는 별도 empirical question이다.

## 27.5 실험 범위

- 한 계열의 Qwen 모델
- 한 수학 도메인
- 한 judge 설계
- finite empirical pool
- training-free tilt의 pool ceiling

외부 타당성은 추가 연구가 필요하다.

## 27.6 Sharp constant는 미해결

polynomial monitor threshold는 \(\Theta(d)\)이지만 정확한 상수는 주어지지 않는다.

## 27.7 Active oracle 구현 가능성

event-tilted law와 size-biased oracle은 정보이론적 benchmark다. 실제 시스템에서 likelihood ratio를 정확히 계산하고 sampler를 구현하기 어려울 수 있다.

## 27.8 Score tail assumption

light-tailed, non-saturating score에서 slope-one geometry가 나온다. bounded/capped score나 heavy-tail에서는 다른 phase가 가능하다.

---

# 28. 자주 생기는 오해

## 오해 1: \(t\)는 generation temperature다

아니다. \(t\)는 trace-level visible score tilt 강도다. generation temperature는 보통 \(\tau\)로 쓰며 token logits에 적용된다.

## 오해 2: \(q=\mu e^{tS}\)는 이미 확률분포다

정규화가 필요하다.

\[
q(o)=\frac{\mu(o)e^{tS(o)}}{\mathbb E_\mu e^{tS}}.
\]

## 오해 3: KL이 6이면 샘플이 정확히 \(e^6\)개 필요하다

정확한 등식이 아니라 asymptotic log-scale 관계다.

\[
\log n=d+o(d).
\]

\(\epsilon,\delta\), 함수 class, finite-sample constant가 큰 영향을 준다.

## 오해 4: Random score는 fidelity를 망치지 않으니 감사도 쉽다

fidelity shift가 0에 가깝더라도 그 사실을 모르는 감사자는 여전히 희귀 영역을 확인해야 할 수 있다. 다만 target shift가 \(\epsilon\)보다 작으면 해당 추정 문제는 trivial해질 수 있다.

## 오해 5: Monitor가 correctness를 잘 예측하면 충분하다

worst-case shift auditing에서는 \(g\)를 포착해야 한다. baseline predictive accuracy와 density-shift approximation은 다른 능력이다.

## 오해 6: 큰 total KL이면 반드시 감사가 어렵다

일반 policy에서는 아니다. audit-relevant rarity \(r\)가 직접적인 지표다.

## 오해 7: Active auditing은 distribution shift를 만들므로 무효다

active audit의 목적이 바로 informative region에 query하는 것이다. 다만 query law가 얼마나 멀리 이동했는지 비용으로 계산해야 공정한 비교가 된다.

## 오해 8: E3가 실제 correctness monitor 실패를 증명했다

아니다. 학습된 monitor에 대한 worst-case bounded blind spot을 인증했다.

---

# 29. 정리별 한 페이지 치트시트

## Theorem 1 — No-hit lower bound

**상황**: 두 세계가 희귀 영역 \(A\)에서만 다름  
**핵심 사건**: \(n\)개 샘플이 모두 \(A\)를 miss  
**결과**:

\[
(1-p)^n\le2\delta
\quad\Rightarrow\quad
n=\Omega\left(\frac1p\log\frac1\delta\right).
\]

**한 문장**: 보지 못한 영역의 label 차이는 추정할 수 없다.

## Theorem 2 — Light-tailed phase

**상황**: Weibull light-tail score + exponential tilt + Lipschitz fidelity  
**결과**:

\[
\log n^*_{\text{base}}=d+o(d).
\]

**한 문장**: score-directed KL \(d\)는 baseline label cost의 로그가 된다.

## Proposition 1 — Free active

\[
n=O(\epsilon^{-2}\log1/\delta)
\]

per arm, independent of \(d\).

**한 문장**: optimized distribution에서 직접 보면 희귀도 문제는 사라진다.

## Theorem 3 — Conservation law

\[
K+\log n\gtrsim r.
\]

**한 문장**: 희귀 tail 접근 비용은 KL 이동과 샘플 수 사이에서만 교환될 수 있다.

## Theorem 4 — Size-biased oracle

\[
D(P_m^\star\Vert\mu^m)
\approx[d-\log m]_+.
\]

**한 문장**: block size가 \(\log m\)만큼 KL 부담을 줄여 frontier를 달성한다.

## Theorem 5 — Monitor duality

\[
\text{worst blind spot}
=
\rho\,\operatorname{dist}_1(g,V).
\]

**한 문장**: monitor가 density shift를 못 근사한 만큼 숨은 fidelity loss가 존재한다.

## Theorem 6 — Sufficiency

\[
T\text{ sufficient}
\iff
g\text{ is a function of }T.
\]

**한 문장**: monitor statistic이 likelihood ratio를 결정해야 완전한 bounded-residual auditing이 가능하다.

## Theorem 7 — Polynomial complexity

\[
r^*_{\text{degree}}=\Theta(d).
\]

**한 문장**: Gaussian tilt의 희귀 tail을 표현하려면 polynomial degree가 KL과 선형으로 증가해야 한다.

---

# 30. 연습문제와 해설

## 문제 1

baseline에서 tail \(A\)의 확률이 \(10^{-4}\)다. tail을 한 번 이상 볼 확률을 약 63%로 만들려면 몇 샘플이 필요한가?

### 해설

\[
P(\text{no hit})\approx e^{-np}.
\]

hit 확률 63%이면 no-hit이 약 37% \(=e^{-1}\)이므로 \(np\approx1\).

\[
n\approx10^4.
\]

---

## 문제 2

\[
\mu=(0.8,0.2),\quad S=(0,2),\quad t=1.
\]

tilt \(q\)를 계산하라.

### 해설

unnormalized weight:

\[
(0.8e^0,\;0.2e^2)
=
(0.8,\;1.4778).
\]

합은 2.2778.

\[
q\approx(0.351,\;0.649).
\]

---

## 문제 3

\(q(A)=1/2\), \(\mu(A)=e^{-6}\)이다. \(r\)은?

### 해설

\[
r=\log\frac1{\mu(A)}=6.
\]

baseline에서 tail hit까지 필요한 샘플 scale은 \(e^6\approx403\)이다. 정밀 추정에는 더 큰 상수가 필요할 수 있다.

---

## 문제 4

감사 budget이 \(K=2\), rarity가 \(r=6\)이면 conservation law상 필요한 \(\log n\)의 최소 scale은?

### 해설

\[
K+\log n\gtrsim r
\]

이므로

\[
\log n\gtrsim4,
\qquad
n\gtrsim e^4\approx54.6.
\]

상수와 confidence 항은 생략한 scale 계산이다.

---

## 문제 5

왜 \(S\)-monitor는 exponential tilt에서 population-level로 충분한가?

### 해설

\[
g=e^{tS-\Lambda(t)}-1.
\]

즉 \(g\)가 \(S\)만의 함수다. 따라서 \(S\)를 알면 likelihood ratio를 정확히 알 수 있다.

---

## 문제 6

왜 baseline correctness predictor와 좋은 audit monitor가 다를 수 있는가?

### 해설

baseline predictor는 \(m(O)\)를 잘 맞추는 것이 목표다.

audit monitor는 분포 변화가 결합되는 방향 \(g(O)\)를 포착해야 한다. 실제 shift는

\[
\Delta_q(m)=\mathbb E_\mu[gm].
\]

이므로 \(m\)의 평균적 예측보다 \(g\)와 residual의 정렬을 놓치지 않는 것이 중요하다.

---

## 문제 7

GRPO의 total KL이 30인데 rarity가 3이라면 baseline hit scale은 \(e^{30}\)인가 \(e^3\)인가?

### 해설

감사에 직접 관련된 것은 rarity \(r=3\)이므로 hit scale은 \(e^3\) 쪽이다.

total KL 30의 나머지는 audit score와 무관한 방향에 쓰였을 수 있다.

---

## 문제 8

왜 high-temperature sampler가 큰 KL을 쓰면서도 나쁜 auditor일 수 있는가?

### 해설

temperature는 분포를 넓히지만, 원하는 judge-score tail로 선택적으로 이동하지 않는다. 많은 확률질량을 무관한 낮은 score trace로도 보낼 수 있다. 따라서 KL당 tail hit 증가가 작을 수 있다.

---

# 31. 권장 학습 순서

## 1단계: 직관

다음 세 식의 뜻을 말로 설명할 수 있게 한다.

\[
r=\log1/\mu(A)
\]

\[
n_{\text{base}}\approx e^r
\]

\[
K+\log n\gtrsim r
\]

## 2단계: 분포 변환

다음을 직접 계산한다.

\[
q_t(o)
=
\frac{\mu(o)e^{tS(o)}}{Z(t)}.
\]

작은 2–3 outcome 예제로 \(q_t\), KL, likelihood ratio를 계산해 본다.

## 3단계: Coverage

Theorem 1의 no-hit proof를 스스로 다시 쓴다.

## 4단계: Gaussian special case

\[
\mu=N(0,1),\qquad q_t=N(t,1)
\]

에서

\[
d=t^2/2,\qquad r=d+\frac12\log d+O(1)
\]

를 이해한다.

## 5단계: Active audit

전체 transcript를 hit indicator로 압축하는 데이터 처리 논리를 이해한다.

## 6단계: Monitor duality

\[
\Delta_q(m)=\mathbb E_\mu[gm]
\]

와

\[
\|f\|_1=\sup_{\|h\|_\infty\le1}\mathbb E[fh]
\]

를 연결한다.

## 7단계: 실험

Figure 2–5를 다음 질문으로 읽는다.

- x축이 정확히 무엇인가?
- y축이 비용인가, 성능인가, distortion인가?
- 이론의 어느 예측을 검증하는가?
- finite-pool artifact는 어디서 생기는가?
- 실제 GRPO와 exact tilt는 왜 다른가?

---

# 32. 용어 사전

**Absolute continuity**  
\(\mu(A)=0\)인 곳에서는 \(q(A)=0\)도 성립하는 관계. likelihood ratio \(dq/d\mu\)가 존재하려면 필요하다.

**Adaptive audit**  
이전 query 결과를 보고 다음 query distribution을 바꾸는 감사.

**Audit-relevant rarity**  
optimized policy에서 중요한 영역이 baseline에서 얼마나 희귀한지를 로그로 측정한 값.

**Baseline distribution \(\mu\)**  
최적화 전 모델의 출력분포.

**Best-of-\(n\)**  
baseline에서 \(n\)개 생성 후 score가 가장 높은 것을 선택하는 정책.

**Cumulant generating function**  
\(\Lambda(t)=\log\mathbb E[e^{tS}]\). tilt의 평균과 분산 등을 미분으로 생성한다.

**Data-processing inequality**  
데이터를 요약하거나 변환하면 KL divergence가 커지지 않는다는 성질.

**Density ratio / likelihood ratio**  
\(L=dq/d\mu\). 새 분포가 각 sample을 baseline보다 몇 배 선호하는지 나타낸다.

**Density shift**  
\(g=L-1\). baseline 대비 증가·감소량.

**Effective sample size**  
importance weights가 실제로 몇 개의 균등 샘플과 비슷한 정보를 가지는지 나타내는 지표.

**Exponential tilt**  
\(\mu\)에 \(e^{tS}\)를 곱해 score가 높은 영역을 강조하는 분포 변환.

**Fidelity**  
진짜 correctness, soundness 등 실제 관심 속성.

**Free active auditing**  
baseline과 optimized distribution에서 직접 라벨을 얻는 이상화된 oracle.

**Goodhart’s law / overoptimization**  
proxy score를 강하게 최적화할수록 proxy와 실제 목표의 관계가 깨질 수 있는 현상.

**Hidden label**  
비싸거나 드물게만 관측할 수 있는 실제 target label.

**KL budget**  
분포를 baseline에서 얼마나 멀리 움직일 수 있는지를 KL로 제한한 예산.

**Light-tailed distribution**  
큰 값의 확률이 매우 빠르게 감소하는 분포. Gaussian, Weibull \(\alpha>1\) 등이 예.

**Minimax**  
가능한 데이터 생성 세계 중 최악의 경우를 고려해 최선의 알고리즘 성능을 정의하는 관점.

**Monitor span \(V\)**  
monitor feature들의 선형결합으로 표현 가능한 함수 공간.

**No-hit event**  
샘플이 중요한 희귀 영역을 한 번도 방문하지 않은 사건.

**Prompt-conditional tilt**  
prompt 분포는 유지하면서 각 prompt 내부 출력분포만 score에 따라 tilt하는 방식.

**Radon–Nikodym derivative**  
연속 분포에서 \(q/\mu\) 역할을 하는 엄밀한 density ratio.

**Size-biased law**  
likelihood ratio가 큰 sample 또는 block에 더 높은 sampling probability를 주는 분포.

**Structured complexity**  
단순 bit 수가 아니라 polynomial degree, feature dictionary, network structure 등 monitor가 허용하는 표현 형식의 복잡도.

**Tail leakage**  
monitor family의 함수가 전체 질량 중 희귀 tail에 집중시킬 수 있는 비율.

**Total variation**  
두 분포 차이의 또 다른 거리. \(\frac12\mathbb E_\mu|dq/d\mu-1|\).

**Visible score**  
싸고 자동으로 계산할 수 있어 최적화에 사용하는 proxy score.

**Weibull tail**  
밀도가 \(e^{-s^\alpha/\alpha}\) 꼴로 감소하는 tail family.

---

# 최종 요약

논문의 구조는 사실 다음 다섯 줄로 압축된다.

1. 최적화는
   \[
   q_t\propto\mu e^{tS}
   \]
   로 높은 score 영역에 질량을 옮긴다.

2. optimized policy에서 중요한 영역 \(A\)가 baseline에서
   \[
   \mu(A)=e^{-r}
   \]
   만큼 희귀해진다.

3. baseline-only 감사에는
   \[
   n\approx e^r
   \]
   규모의 라벨이 필요하다.

4. active 감사도
   \[
   K+\log n\gtrsim r
   \]
   라는 가격을 피할 수 없다.

5. monitor는
   \[
   g=dq/d\mu-1
   \]
   를 표현하지 못한 만큼 worst-case blind spot을 가진다.

exponential tilt와 light-tailed score에서는

\[
r=d+o(d),
\qquad
d=D(q_t\Vert\mu).
\]

따라서 **visible-score optimization의 KL pressure가 hidden-fidelity auditing의 비용을 log-scale로 결정한다.**


---

# 부록 A. 수학 배경을 더 깊게 익히기

이 부록은 본문을 읽다가 “왜 이 식이 가능한가?”라는 질문이 생길 때 찾아보는 참고 절이다.

## A.1 지수와 로그

### 지수의 핵심 성질

\[
e^{a+b}=e^ae^b.
\]

그래서 여러 단계의 multiplicative likelihood ratio는 로그를 취하면 additive하게 변한다.

\[
\log(L_1L_2\cdots L_R)
=
\sum_{r=1}^R\log L_r.
\]

KL과 path composition에 로그가 자연스럽게 등장하는 이유다.

### 로그가 rarity를 표현하는 이유

희귀확률이

\[
p=e^{-r}
\]

이면 역수 표본 scale은

\[
1/p=e^r.
\]

로그를 쓰면 지수적으로 큰 비용을 선형 좌표 \(r\)로 나타낼 수 있다.

## A.2 미분

함수의 미분은 입력이 조금 변할 때 출력이 얼마나 변하는지 나타낸다.

exponential tilt의 log-normalizer

\[
\Lambda(t)=\log\mathbb E_\mu[e^{tS}]
\]

에 대해 미분하면

\[
\Lambda'(t)
=
\frac{\mathbb E_\mu[Se^{tS}]}
{\mathbb E_\mu[e^{tS}]}
=
\mathbb E_{q_t}[S].
\]

한 번 더 미분하면

\[
\Lambda''(t)
=
\mathbb E_{q_t}[S^2]
-
(\mathbb E_{q_t}[S])^2
=
\operatorname{Var}_{q_t}(S).
\]

따라서 \(\Lambda\)는 convex하다.

## A.3 Convexity와 Jensen 부등식

함수 \(\phi\)가 convex하면

\[
\phi(\mathbb E[X])
\le
\mathbb E[\phi(X)].
\]

로그는 concave하므로 방향이 반대다.

\[
\mathbb E[\log X]
\le
\log\mathbb E[X].
\]

정리 4의 KL upper bound를 얻을 때 Jensen과 log의 concavity가 사용된다.

## A.4 라그랑주 승수

제약

\[
\sum_oq(o)=1
\]

이 있는 최적화에서는 목적함수에

\[
\lambda\left(\sum_oq(o)-1\right)
\]

을 더한다.

각 \(q(o)\)에 대해 미분값을 0으로 놓으면 stationary point를 얻는다. KL term은 strictly convex이므로 최적해가 유일하게 결정되는 경우가 많다.

## A.5 적분 표기 \(d\mu\)

\[
\int h(o)\,d\mu(o)
\]

는 “분포 \(\mu\)에 따라 \(h\)를 평균낸다”는 뜻이다.

\(\mu\)가 density \(p_\mu(o)\)를 가진다면

\[
\int h(o)\,d\mu(o)
=
\int h(o)p_\mu(o)\,do.
\]

엄밀한 측도론을 모두 몰라도 논문을 읽을 때는 이 해석으로 충분하다.

## A.6 절대연속과 density ratio

\(q\ll\mu\)는

\[
\mu(A)=0\Rightarrow q(A)=0
\]

라는 뜻이다.

baseline이 절대 생성할 수 없는 output을 optimized policy가 생성한다면 \(q/\mu\) ratio가 무한대가 되어 standard importance weighting이 불가능하다.

논문 실험에서 prompt suffix sampler가 같은 support를 유지하도록 설계한 이유와도 관련된다.

## A.7 Total variation

\[
\operatorname{TV}(q,\mu)
=
\sup_A|q(A)-\mu(A)|
=
\frac12\mathbb E_\mu|g|.
\]

\(g=dq/d\mu-1\)의 \(L^1\) 크기는 두 분포의 전체적인 차이와 직접 연결된다.

Figure 4에서 relative distortion의 denominator가 \(\mathbb E_\mu|g|\)인 이유는 전체 density shift 중 monitor가 놓친 비율로 해석하기 위해서다.

## A.8 Gaussian distribution

표준정규분포:

\[
\phi(x)
=
\frac1{\sqrt{2\pi}}e^{-x^2/2}.
\]

upper-tail probability:

\[
\bar\Phi(t)=P(Z\ge t).
\]

큰 \(t\)에서 Mills ratio:

\[
\bar\Phi(t)
\sim
\frac{\phi(t)}{t}
=
\frac1{t\sqrt{2\pi}}e^{-t^2/2}.
\]

로그를 취하면

\[
\log\frac1{\bar\Phi(t)}
=
\frac{t^2}{2}+\log t+O(1).
\]

이 식이 Gaussian rarity와 KL을 연결한다.

## A.9 Weibull-type tail

논문의 density는

\[
p_\alpha(s)\propto e^{-s^\alpha/\alpha}.
\]

\(\alpha>1\)일 때 매우 큰 \(s\)의 확률이 빠르게 줄어든다.

tilt \(e^{ts}\)와 곱하면 exponent는

\[
ts-s^\alpha/\alpha.
\]

최댓값 위치와 curvature를 보면 optimized distribution의 중심과 폭을 근사할 수 있다.

## A.10 Laplace method / saddle-point approximation

큰 parameter가 있을 때

\[
\int e^{M\phi(x)}dx
\]

는 \(\phi(x)\)가 최대인 \(x^\star\) 근처가 거의 전부를 차지한다.

근처에서 2차 Taylor 전개:

\[
\phi(x)
\approx
\phi(x^\star)
+
\frac12\phi''(x^\star)(x-x^\star)^2.
\]

그러면 적분이 Gaussian 적분처럼 된다.

Theorem 2에서 optimized score가 \(s_t\) 근처에 집중하고 baseline mass가 \(e^{-d+o(d)}\)가 됨을 보일 때 쓰는 핵심 도구다.

## A.11 Large deviations

큰 편차 이론은 평균적인 위치에서 멀리 떨어진 희귀 사건의 확률이

\[
P(A_d)\approx e^{-I_d}
\]

꼴로 감소하는 것을 분석한다.

이 논문에서는 optimized policy가 평범하게 보는 영역이 baseline에서는 large-deviation event가 된다.

KL \(d\)가 그 event의 rate와 일치하는 것이 \(r=d+o(d)\)의 배경이다.

## A.12 Minimax

추정기 \(\hat\theta\)의 minimax risk는

\[
\inf_{\hat\theta}
\sup_{f\in\mathcal F}
R_f(\hat\theta).
\]

뜻:

1. 자연이 허용된 class에서 가장 어려운 \(f\)를 고른다.
2. 우리는 그 최악의 경우에도 가장 좋은 추정기를 고른다.

논문의 lower bound는 “어떤 실제 세계에서도 반드시 이만큼 어렵다”가 아니라, “허용 class 안에는 적어도 이만큼 어려운 세계가 있다”는 의미다.

## A.13 Two-point lower bound

전체 class를 분석하는 대신 서로 구별하기 어려운 두 세계 \(f_0,f_1\)만 고른다.

- 관측분포는 가깝다.
- 목표 parameter는 멀다.

그러면 어떤 추정기도 두 세계에서 동시에 정확할 수 없다.

Theorem 1과 Theorem 3의 핵심 proof strategy다.

## A.14 Data-processing inequality의 사용법

전체 transcript \(Z\)가 복잡해도, 우리가 관심 있는 사건

\[
H=h(Z)
\]

로 압축할 수 있다.

\[
D(P_Z\Vert Q_Z)
\ge
D(P_H\Vert Q_H).
\]

따라서 “전체 transcript KL이 \(K\) 이하”이면 hit indicator의 Bernoulli KL도 \(K\) 이하다.

복잡한 adaptive auditor를 단순한 hit probability 문제로 바꾸는 강력한 단계다.

## A.15 선형 span

\[
V=\operatorname{span}\{v_1,\dots,v_k\}
\]

는

\[
a_1v_1+\cdots+a_kv_k
\]

꼴의 모든 함수다.

linear monitor는 feature를 이런 방식으로 결합한다.

## A.16 Orthogonality

확률분포 \(\mu\) 아래에서

\[
\mathbb E_\mu[hv]=0
\]

이면 \(h\)와 \(v\)가 orthogonal하다고 말한다.

논문의 \(h\perp V\)는 모든 monitor feature와의 평균 곱이 0이라는 뜻이다.

## A.17 \(L^p\) norm

\[
\|f\|_p
=
\left(\mathbb E|f|^p\right)^{1/p}.
\]

특히:

\[
\|f\|_1=\mathbb E|f|,
\]

\[
\|f\|_2=(\mathbb E f^2)^{1/2},
\]

\[
\|f\|_\infty=\text{essential supremum of }|f|.
\]

\(L^2\)는 제곱오차, \(L^1\)은 절대오차에 대응한다.

## A.18 왜 monitor theorem은 \(L^1\)인가?

hidden residual \(h\)에

\[
\|h\|_\infty\le1
\]

라는 bounded constraint를 둔다.

\(L^\infty\)의 dual norm이 \(L^1\)이므로 정확한 worst-case 값이 \(L^1\) distance가 된다.

## A.19 Sigma-algebra

\(\sigma(T)\)는 statistic \(T\)를 알 때 구분할 수 있는 사건들의 집합이다.

“\(g\)가 \(\sigma(T)\)-measurable”이라는 말은 \(g\)가 오직 \(T\)만의 함수라는 뜻으로 이해하면 된다.

## A.20 Conditional expectation as projection

\[
\mathbb E[g\mid T]
\]

는 \(T\)만으로 \(g\)를 설명하는 최선의 요약 중 하나다.

\(L^2\)에서는 정확한 orthogonal projection이다. 논문의 정리 6은 \(L^1\) 거리와 bounded invisible residual의 관점에서 statistic sufficiency를 다룬다.

## A.21 Hermite polynomials

Gaussian measure에서 ordinary monomial보다 자연스러운 orthogonal basis가 Hermite polynomial이다.

\[
H_0(x)=1,\quad H_1(x)=x,\quad H_2(x)=x^2-1,\dots
\]

Gaussian exponential tilt의 likelihood ratio는 Hermite series로 전개 가능하다.

degree truncation의 residual tail을 계산해 Theorem 7 upper bound를 얻는다.

## A.22 Hypercontractivity

저차 Gaussian polynomial의 높은 \(L^p\) norm이 \(L^2\) norm에 의해 제어된다는 성질이다.

degree \(r\) polynomial \(p\)에 대해 논문 proof는

\[
\|p\|_4
\le
3^{r/2}\|p\|_2
\]

를 사용한다.

이 성질은 저차 polynomial이 매우 희귀한 tail에 마음대로 질량을 집중하지 못하게 한다.

---

# 부록 B. 원문의 모든 주요 정리와 명제

## B.1 Theorem 1 — No-hit lower bound

본문 Section 3. 두 세계가 baseline 질량 \(p\)인 영역에서만 다르면

\[
(1-p)^n\le2\delta.
\]

## B.2 Definition 1 — Audit coverage modulus

큰 target separation을 작은 baseline 영역에 숨길 수 있는 최소 질량을 정의한다.

## B.3 Theorem 2 — Light-tailed auditability phase

Weibull-type score와 Lipschitz fidelity에서

\[
\log n^*_{\mathrm{base}}=d+o(d).
\]

## B.4 Proposition 1 — Free active contrast

\[
O(\epsilon^{-2}\log1/\delta)
\]

labels per distribution으로 충분하고 \(d\)와 무관하다.

## B.5 Theorem 3 — Budget–sample conservation law

전체 adaptive query transcript KL이 \(K\) 이하이면 exact minimax risk를 Bernoulli hit problem으로 표현할 수 있고

\[
K+\log n\gtrsim r.
\]

## B.6 Corollary 1 — Gaussian audit-pressure phase

Gaussian tilt에서

\[
r=d+\frac12\log d+O(1).
\]

따라서 conservation frontier의 leading term은 \(d\)다.

## B.7 Theorem 4 — Size-biased likelihood-ratio block oracle

block law가 frontier를 \(o(d)\) 오차로 달성한다.

## B.8 Corollary 2 — Weibull budgeted-active phase

Weibull score에서

\[
K+\log N=\Theta_{\epsilon,\delta,\alpha}(d(t)).
\]

## B.9 Proposition 2 — Multi-stage path-KL composition

최종 output KL은 path KL 이하이고, non-canceling stage에서 audit pressure가 단계별 KL 합에 따라 누적된다.

## B.10 Theorem 5 — Exact monitor duality

\[
\inf_m\Delta_q(m)
=
-\rho\operatorname{dist}_{L^1(\mu)}(g,V).
\]

## B.11 Theorem 6 — Statistic sufficiency dichotomy

bounded residual 전부를 감사할 수 있는 필요충분조건은 \(g\)가 statistic의 함수인 것이다.

## B.12 Inequality (5) — Tail leakage lower bound

\[
\operatorname{dist}_1(g,V)
\ge
\frac{q(A)-\mu(A)-2\beta_V(A)}
{1+\beta_V(A)}.
\]

## B.13 Proposition 3 — Structured tail-leakage principle

family capacity가 rarity와 충분히 빠르게 성장하지 않으면 constant distortion이 남는다.

## B.14 Theorem 7 — Linear polynomial-monitor phase

Gaussian polynomial monitor에서 필요한 degree가

\[
\Theta(d).
\]

## B.15 Theorem 8 — Smooth strict separation

Gaussian tilt와 Lipschitz class에서도 adaptive auditor는

\[
\frac{K}{1-2\delta}+\log n
\ge
d+\frac12\log d-O_{\delta,L}(1)
\]

을 만족해야 한다.

특히

\[
K=o(d)
\]

이면

\[
n\ge e^{d-o(d)}.
\]

**중요성**: Theorem 3의 두-world discontinuous construction을 smooth fidelity class로 확장한다.

## B.16 Theorem 9 — Fixed audit sampler exact risk

고정 audit distribution \(\nu\)에서 \(n\)개 noiseless label을 얻을 때 exact minimax absolute risk는

\[
\frac{b\eta}{2}(1-\nu(A))^n.
\]

따라서 density ratio cap

\[
\frac{d\nu}{d\mu}\le R
\]

이면

\[
n=\Omega\left(
\frac{1}{R\mu(A)}
\log\frac{b\eta}{2\epsilon}
\right).
\]

per-draw KL cap만 있을 때 한 번의 hit probability를 상수로 만들려면

\[
\kappa\ge\log(1/p)-O(1).
\]

**중요성**: 단순히 tail probability를 polynomial factor로 enrich하는 sampler는 exponential barrier를 제거하지 못한다.

## B.17 Proposition 4 — Bit-limited monitors are too powerful

\(M\)-value statistic의 최적 distortion은 \(g\)의 \(M\)-level scalar quantization 문제와 같다.

Gaussian tilt에서 fixed distortion에는

\[
M=\Theta_\tau(\sqrt d)
\]

level이면 충분하므로

\[
\frac12\log_2d+O_\tau(1)
\]

bits만 필요하다.

**중요성**: meaningful lower bound는 bit count가 아니라 structured representation에 대해 세워야 한다.

---

# 부록 C. 원문 페이지와 그림 안내

페이지 번호는 PDF의 표지 페이지를 1쪽으로 센 기준이다.

| 위치 | 내용 |
|---|---|
| 1쪽 | 초록, 핵심 문제, 세 장벽의 요약 |
| 2쪽 Figure 1 | “한 희귀 영역, 세 감사 장벽” 전체 개념도 |
| 2쪽 Section 2 | \(O,\mu,S,m,q_t,d,r,\Delta\) 정의 |
| 2–3쪽 Section 3 | Coverage barrier, Theorem 1–2 |
| 3–4쪽 Section 4 | Audit-pressure barrier, Theorem 3–4 |
| 4쪽 Section 5 | Monitor duality와 structured complexity, Theorem 5–7 |
| 4–5쪽 Section 6.1 | 데이터와 exact empirical tilt testbed |
| 5쪽 Figure 2 | E1 label complexity와 judge hacking |
| 5–6쪽 Figure 3 | E2 \(K+\log n\) frontier |
| 6쪽 Figure 4 | E3 monitor distortion |
| 6–7쪽 Figure 5 | exact tilt와 GRPO 비교 |
| 7쪽 Section 8 | 한계와 실무적 함의 |
| 9–10쪽 Appendix A | proof sketches, Theorem 8–9, Proposition 4 |
| 10쪽 Figures 6–8 | synthetic validation과 full monitor ladder |
| 11–12쪽 Appendix C–F | GRPO 세부, universality, experimental details |
| 12쪽 Figure 9 | GRPO score/accuracy vs. measured KL |
| 13쪽 Figure 10 | rarity–KL universality와 score-dependent fidelity shift |

---

# 부록 D. 그림을 읽는 정확한 순서

## D.1 Figure 1

먼저 왼쪽에서 오른쪽으로 읽는다.

1. \(\mu\)가 \(e^{tS}\)에 의해 \(q_t\)로 reweight된다.
2. rarity \(r=\log1/\mu(A)\)가 생긴다.
3. 같은 \(r\)가 coverage, active query, monitor capacity를 통제한다.

## D.2 Figure 2 왼쪽

- x축: optimization KL \(d(t)\)
- y축: \(\log n^*\)
- 빨간색: baseline-only
- 파란색: free-active
- 점선: slope 1

빨간색이 직선이면 실제 label count가 지수적으로 증가한다는 뜻이다.

## D.3 Figure 2 오른쪽

두 y축을 혼동하지 않는다.

- 왼쪽 y축: judge score
- 오른쪽 y축: true accuracy

judge score는 단조 증가하지만 accuracy는 peak 후 감소한다.

## D.4 Figure 3

- x축: 전체 audit KL \(K\), symlog scale
- y축: \(\log n\)
- dashed frontier:
  \[
  K+\log n=r
  \]

점이 오른쪽 위에 있을수록 둘 다 많이 쓰는 비효율적 sampler다.

## D.5 Figure 4

relative distortion이 작을수록 좋다.

analytic S-oracle이 0인 것은 \(g\)가 \(S\)의 정확한 함수이기 때문이다.

고-\(d\)에서 learned S-control이 다시 나빠지는 것은 population information 부족이 아니라 finite-sample spike 문제다.

## D.6 Figure 5 왼쪽

- tilt curves는 대체로 slope-one
- capped length는 예외
- GRPO diamonds는 같은 total KL에서 아래쪽

따라서 total KL과 audit-relevant rarity를 구분해야 한다.

## D.7 Figure 5 오른쪽

x축이 log scale임에 주의한다.

GRPO는 큰 measured KL을 사용한 뒤 accuracy가 peak하고 감소한다.

## D.8 Figure 10 오른쪽

같은 KL에서도 fidelity shift의 부호가 score에 따라 다르다는 가장 직접적인 그림이다.

이 그림이 “geometry controls audit cost; score alignment controls benefit/harm”을 요약한다.

---

# 부록 E. 논문을 다 이해했는지 확인하는 질문

다음 질문에 자기 말로 답할 수 있으면 논문의 핵심을 이해한 것이다.

1. 왜 \(q_t\propto\mu e^{tS}\)가 KL-regularized objective의 해인가?
2. generation temperature \(\tau\)와 tilt parameter \(t\)는 어떻게 다른가?
3. 왜 optimized median region을 baseline에서 찾는 비용이 \(1/\mu(A)\) 규모인가?
4. \(r=d+o(d)\)에서 \(o(d)\)는 무엇을 의미하는가?
5. Theorem 1의 두 세계는 왜 no-hit event에서 구별 불가능한가?
6. free-active auditor가 \(d\)와 무관한 이유는?
7. \(K+\log n\gtrsim r\)에서 \(K\)와 \(\log n\)이 왜 더해지는가?
8. size-biased block oracle에서 block size \(m\)가 왜 KL을 \(\log m\)만큼 줄이는가?
9. 왜 좋은 correctness predictor가 반드시 좋은 audit monitor는 아닌가?
10. Theorem 5에서 \(L^1\) 거리가 등장하는 이유는?
11. 왜 \(S\)를 보는 monitor는 exponential tilt에서 sufficient한가?
12. 왜 bit-limited monitor lower bound는 약한가?
13. polynomial degree가 왜 \(\Theta(d)\)여야 하는가?
14. E3가 실제 correctness blind spot을 직접 증명한 것은 아닌 이유는?
15. GRPO에서 total KL과 rarity가 분리되는 이유는?
16. 왜 random-score tilt도 audit geometry를 만들 수 있는가?
17. length score가 slope-one 관계에서 벗어난 이유는?
18. model-derived faithfulness label이 왜 위험할 수 있는가?
19. 논문의 결과 중 minimax인 것과 empirical한 것을 구분할 수 있는가?
20. 실제 정책을 감사할 때 total KL 대신 무엇을 직접 측정해야 하는가?

---

# 부록 F. 연구 확장 아이디어

1. Heavy-tailed score에서 \(r\)–\(d\) 관계는 어떻게 변하는가?
2. bounded score와 saturation에서 정확한 phase transition은 무엇인가?
3. 실제 sampler를 conservation frontier에 가깝게 만드는 알고리즘은 무엇인가?
4. likelihood ratio를 모를 때 size-biased oracle을 근사할 수 있는가?
5. nonlinear monitor family에 대한 rigorous structured complexity는 어떻게 정의할 것인가?
6. correctness가 worst-case residual과 얼마나 정렬되는지를 데이터로 측정할 수 있는가?
7. prompt distribution까지 바뀌는 RL에서 mixture rarity는 어떻게 정의해야 하는가?
8. sequential deployment에서 audit pressure를 online하게 추적할 수 있는가?
9. 여러 visible score를 동시에 최적화할 때 희귀영역 geometry는 어떻게 합성되는가?
10. 인간 감사자의 인지적 monitor capacity를 \(V\)로 모델링할 수 있는가?

---

# 문서 끝

이 가이드를 한 번에 모두 이해할 필요는 없다. 가장 중요한 학습 루프는 다음과 같다.

1. 작은 유한분포에서 \(q_t\)와 KL을 직접 계산한다.
2. no-hit lower bound를 직접 증명한다.
3. Gaussian special case로 \(r\approx d\)를 확인한다.
4. monitor duality를 \(\Delta=\mathbb E_\mu[gm]\)에서 다시 유도한다.
5. Figure 2–5를 각 정리와 연결한다.

이 과정을 반복하면 처음에는 서로 무관해 보이던 coverage, active sampling, monitor complexity가 모두 **같은 density shift와 같은 희귀영역 geometry**에서 나온다는 점이 보이기 시작한다.
