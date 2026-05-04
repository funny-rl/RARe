
## Question

0. on-policy와 off-policy의 재정의: 정책은 같은데 거기에 대해 \max, expectation을 취하면 target이 달라 off-policy가 되는 것인가? 아님 이 두 가지는 같은 정책의 행동 확률을 공유하므로 on-policy인가. 

1. In hazardous domains such as CliffWalking, frequent decision points increase the34 likelihood of catastrophic failure. Notably, this vulnerability persists not only under stochastic training conditions but also during evaluation, particularly when the learned greedy action is sub-optimal.

이렇게 표현해도 괜찮을까?

2. Background의 수식 1, 2번에서 상태전이확률을 결정론적으로 표현하는 것 같은데 맞을까? 그럼 MDP의 상태전이확률 함술를 확률 분포가 아닌 카르테시안 곱으로 표현해야 할까. 

3. Background에서 수식 2번에 Reward 함수에 expectation을 넣는 게 맞을까. 

4. Overestimation bias 그림 2개 버전으로 넣을까요 4개 버전으로 넣을까요.

5. 수렴성 증명 내용을 넣어야 할까. 

6. Overestimation 실험에서 (\max, base agent)의 편향을 각각 나눠서 분석하는 법: 지금은 그냥 합쳐서 보여줌. 



## ablation:
1. 샘플링 수가 줄어들 때마다 성능이 낮아지는가?

2. \max operator와 action policy 각각으로부터 bias가 발생하는가? 후자는 확인 완료, 전자는 내 생각엔 RARe와 TempoRL의 skip value function을 학습하기 위한 TD-Target을 가우시안 랜덤 노이즈로 만들어 실험하기. 