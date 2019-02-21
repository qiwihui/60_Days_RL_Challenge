
![](images/logo5.png)

![](https://img.shields.io/badge/Uploaded-100%25-green.svg)  

### 我为你和我设计了这个挑战：在60天内深入学习深度强化学习!!

您肯定听说过 [Deepmind 使用 AlphaGo Zero](https://www.youtube.com/watch?time_continue=24&v=tXlM99xPQC8) 和 [OpenAI 在 Dota 2 中](https://www.youtube.com/watch?v=l92J1UvHf6M)取得的惊人成果！你不想知道它们是如何工作的吗？现在正是你我最终学会“深度强化学习”，并将其应用到已有项目的合适时机。

> 最终目标是使用这些通用技术并将其应用于各种重要的现实世界问题。
> **Demis Hassabis**

这个项目希望引导你完成深度强化学习算法，从最基本的到高度先进的 AlphaGo Zero。你可以发现**按周组织的主题**和**建议学习资源**。同时，每周我都会提供用 Python 实现的**应用实例**，以帮助你更好地消化理论。我们强烈建议你修改并使用它们！

<br>

**敬请关注** [![Twitter Follow](https://img.shields.io/twitter/follow/espadrine.svg?style=social&label=Follow)](https://twitter.com/andri27_it) [![GitHub followers](https://img.shields.io/github/followers/espadrine.svg?style=social&label=Follow)](https://github.com/andri27-ts)

#### #60天强化学习

现在我们还有 [** Slack 频道**](https://60daysrlchallenge.slack.com/)。要获得邀请，请给我发电子邮件 andrea.lonza@gmail.com。

这是我的第一个此类项目，所以，如果您有任何想法，建议或改进，请联系我 andrea.lonza@gmail.com。

学习深度学习，计算机视觉或自然语言处理请访问我的[一年机器学习之旅](https://github.com/andri27-ts/1-Year-ML-Journey)。

### 必备知识

* 了解 Python 和 PyTorch
* [机器学习](https://github.com/andri27-ts/1-Year-ML-Journey)
* [了解深度学习的基础知识（MLP，CNN 和 RNN）](https://github.com/andri27-ts/1-Year-ML-Journey)

## 目录

 - **[第一周 - 强化学习介绍](#第一周---强化学习介绍)**
 - **[第二周 - 强化学习基础](#第二周---强化学习基础-马尔科夫决策过程-动态规划和不基于模型的控制)**
 - **[第三周 - 值函数近似和 DQN](#第三周---值函数近似和-DQN)**
 - **[第四周 - 策略梯度方法和 A2C](#第四周---策略梯度方法和-A2C)**
 - **[第五周 - 高级策略梯度 - TRPO 和 PPO](#第五周---高级策略梯度---TRPO-和-PPO)**
 - **[第六周 - 进化策略和遗传算法](#第六周---进化策略和遗传算法)**
 - **[第七周 - 基于模型的强化学习](#第七周---基于模型的强化学习)**
 - **[第八周 - 高级概念和你选择的项目](第八周---高级概念和你选择的项目)**
 - **[最后四天 - 评论 + 分享](#最后四天---评论--分享)**
 - **[最好的资源](#最好的资源)**
 - **[额外的资源](#额外的资源)**

<br>

## 第一周 - 强化学习介绍

- **[强化学习简介](https://www.youtube.com/watch?v=JgvyzIkgxF0)，Arxiv Insights**
- **[介绍和课程概述](https://www.youtube.com/watch?v=Q4kF8sfggoI&index=1&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3) - CS294，Levine**
- **[深度强化学习：像素乒乓](http://karpathy.github.io/2016/05/31/rl/)，Karpathy**

---

#### 建议

- 很棒的入门论文：[深度强化学习：概述](https://www.groundai.com/project/deep-reinforcement-learning-an-overview/)
- 开始编码：[从头开始：50行 Python 实现人工智能平衡技术](https://towardsdatascience.com/from-scratch-ai-balancing-act-in-50-lines-of-python-7ea67ef717)

<br>

## 第二周 - 强化学习基础：*马尔科夫决策过程，动态规划和不基于模型的控制*

> 忘记过去的人，终将重蹈覆辙。 - **George Santayana**

本周，我们将了解强化学习的基本内容，从问题的定义一直到用于表达策略或状态质量的函数的估计和优化。

----

### 理论材料

* **[马尔科夫决策过程（Markov Decision Process）](https://www.youtube.com/watch?v=lfHX2hHRMVQ&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-&index=2)**  David Silver 强化学习
  * 马尔科夫过程（Markov Processes）
  * 马尔科夫决策过程（Markov Decision Processes）

- **[动态规划设计（Planning by Dynamic Programming）](https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-&index=3)**  David Silver 强化学习
  * 策略迭代（Policy iteration）
  * 价值迭代（Value iteration）

* **[不基于模型的预测（Model-Free Prediction）](https://www.youtube.com/watch?v=PnHCvfgC_ZA&index=4&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-)**  David Silver 强化学习
  * 蒙特卡罗学习（Monte-Carlo Learning）
  * 时序差分学习（Temporal Difference Learning）
  * TD(λ)

- **[不基于模型的控制（Model-Free Control）](https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-&index=5)**  David Silver 强化学习
  * Ɛ-贪婪策略迭代（Ɛ-greedy policy iteration）
  * GLIE 蒙特卡洛搜索（GLIE Monte Carlo Search）
  * SARSA
  * 重要性采样（Importance Sampling）

----

### 本周项目

[Q-learning applied to FrozenLake](Week2/frozenlake_Qlearning.ipynb). For exercise, you can solve the game using SARSA or implement Q-learning by yourself. In the former case, only few changes are needed.
[Q-learning 解决冰冻湖问题](Week2/frozenlake_Qlearning.ipynb)。作为练习，您可以使用 SARSA 解决游戏问题或自行实施 Q-learning。在前一种情况下，只需要进行少量更改即可。

----

#### 了解更多

- :books: 阅读 [强化学习导论 - Sutton, Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) 的第3、4、5、6、7章节。
- :tv: [价值函数介绍](https://www.youtube.com/watch?v=k1vNh4rNYec&index=6&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3) - DRL UC Berkley，Sergey Levine

<br>

## 第三周 - 值函数近似和 DQN

本周我们将学习更多高级概念，并将深度神经网络应用于 Q-learning 算法。

----

### 理论材料

#### Lectures

- **[价值函数近似（Value functions approximation）](https://www.youtube.com/watch?v=UoPei5o4fps&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=6) - RL by David Silver**
  - 差分近似函数（Differentiable function approximators）
  - 递增方法（Incremental Methods）
  - 批方法（Batch methods）（用于 DQN 网络）

* **[高级 Q-learning 算法](https://www.youtube.com/watch?v=nZXC5OdDfs4&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&index=7) - DRL UC Berkley by Sergey Levine**
  - Replay Buffer
  - Double Q-learning
  - Continous actions (NAF,DDPG)
  - 实用技巧

#### 论文

##### 必读

 - [利用深度强化学习机器可以成为 Atari 游戏达人](https://arxiv.org/pdf/1312.5602.pdf) - 2013
 - [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) - 2015
 - [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/pdf/1710.02298.pdf) - 2017

##### DQN 变型

 - [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) - 2015
 - [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) - 2015
 - [Dueling Network Architectures for Deep Reinforcement Learning](http://proceedings.mlr.press/v48/wangf16.pdf) - 2016
 - [Noisy networks for exploration](https://arxiv.org/pdf/1706.10295.pdf) - 2017
 - [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/pdf/1710.10044.pdf) - 2017

----

### 本周项目

[**DQN 和一些应用于 Pong 的变体**](Week3)

本周的目标是开发一个 DQN 算法玩 Atari 游戏。为了使项目更有趣，我开发了3个 DQN 变型：**Double Q-learning**，**Multi-step learning**，**Dueling networks** 和 **Noisy Nets**。你可以使用它们玩游戏，如果你有信心，你可以实现 Prioritized replay， Dueling networks 或者 Distributional RL。阅读论文以了解更多改进。

-----

#### 建议
  - :tv: [企业深度强化学习：缩小从游戏到行业的差距](https://www.youtube.com/watch?v=GOsUHlr4DKE)

<br>

## 第四周 - 策略梯度方法和 A2C

Week 4 introduce Policy Gradient methods, a class of algorithms that optimize directly the policy. Also, you'll learn about Actor-Critic algorithms. These algorithms combine both policy gradient (the actor) and value function (the critic).

----

### Theoretical material

#### Lectures

* **[Policy gradient Methods](https://www.youtube.com/watch?v=KHZVXao4qXs&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=7) - RL by David Silver**
  - Finite Difference Policy Gradient
  - Monte-Carlo Policy Gradient
  - Actor-Critic Policy Gradient

- **[Policy gradient intro](https://www.youtube.com/watch?v=XGmd3wcyDg8&t=0s&list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37&index=3) - CS294-112 by Sergey Levine (RECAP, optional)**
  - Policy Gradient (REINFORCE and Vanilla PG)
  - Variance reduction

* **[Actor-Critic](https://www.youtube.com/watch?v=Tol_jw5hWnI&list=PLkFD6_40KJIxJMR-j5A1mkxK26gh_qg37&index=4) - CS294-112 by Sergey Levine (More in depth)**
  - Actor-Critic
  - Discout factor
  - Actor-Critic algorithm design (batch mode or online)
  - state-dependent baseline

#### Papers

- [Policy Gradient methods for reinforcement learning with function approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)

---

### Project of the Week

[**Vanilla PG and A2C**](Week4)
The exercise of this week is to implement a policy gradient method or a more sophisticated actor-critic. In the repository you can find an implemented version of PG and A2C. Pay attention that A2C give me strange result. You can try to make it works or implement an [asynchronous version of A2C (A3C)](https://arxiv.org/pdf/1602.01783.pdf).

---

#### Suggested
  - :books: [Intuitive RL: Intro to Advantage-Actor-Critic (A2C)](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)
  - :books: [Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

<br>

## 第五周 - 高级策略梯度 - TRPO 和 PPO

This week is about advanced policy gradient methods that improve the stability and the convergence of the "Vanilla" policy gradient methods. You'll learn and implement PPO, a RL algorithm developed by OpenAI and adopted in [OpenAI Five](https://blog.openai.com/openai-five/).

----

### Theoretical material

#### Lectures

- **[Advanced policy gradients](https://www.youtube.com/watch?v=ycCtmp4hcUs&t=0s&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&index=15) - CS294-112 by Sergey Levine**
  - Problems with "Vanilla" Policy Gradient Methods
  - Policy Performance Bounds
  - Monotonic Improvement Theory
  - Algorithms: NPO, TRPO, PPO

* **[Natural Policy Gradients, TRPO, PPO](https://www.youtube.com/watch?v=xvRrgxcpaHY) - John Schulman, Berkey DRL Bootcamp - (RECAP, optional)**
  * Limitations of "Vanilla" Policy Gradient Methods
  * Natural Policy Gradient
  * Trust Region Policy Optimization, TRPO
  * Proximal Policy Optimization, PPO

#### Papers

- [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) - 2015
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) - 2017

---

### Project of the Week

This week, you have to implement PPO or TRPO. I suggest PPO given its simplicity (compared to TRPO). In the project folder [**Week5**](Week5) you can find an implementation of [**PPO that learn to play BipedalWalker**](Week5).
Furthermore, in the folder you can find other resources that will help you in the development of the project. Have fun!

To learn more about PPO read the [paper](https://arxiv.org/pdf/1707.06347.pdf) and take a look at the [Arxiv Insights's video](https://www.youtube.com/watch?v=5P7I-xPq8u8)

NB: the hyperparameters of the PPO implementation I released, can be tuned to improve the convergence.

---

#### Suggested
  - :books: To better understand PPO and TRPO: [The Pursuit of (Robotic) Happiness](https://towardsdatascience.com/the-pursuit-of-robotic-happiness-how-trpo-and-ppo-stabilize-policy-gradient-methods-545784094e3b)
  - :tv: [Nuts and Bolts of Deep RL](https://www.youtube.com/watch?v=8EcdaCk9KaQ&)
  - :books: PPO best practice: [Training with Proximal Policy Optimization](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md)
  - :tv: [Explanation of the PPO algorithm by Arxiv Insights](https://www.youtube.com/watch?v=5P7I-xPq8u8)

<br>

## 第六周 - 进化策略和遗传算法

In the last year, Evolution strategies (ES) and Genetic Algorithms (GA) has been shown to achieve comparable results to RL methods. They are derivate-free black-box algorithms that require more data than RL to learn but are able to scale up across thousands of CPUs. This week we'll look at this black-box algorithms.

----

### Material

- **Evolution Strategies**
  - [Intro to ES: A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)
  - [ES for RL: Evolving Stable Strategies](http://blog.otoro.net/2017/11/12/evolving-stable-strategies/)
  - [Derivative-free Methods - Lecture](https://www.youtube.com/watch?v=SQtOI9jsrJ0&feature=youtu.be)
  - [Evolution Strategies (paper discussion)](https://blog.openai.com/evolution-strategies/)
- **Genetic Algorithms**
  - [Introduction to Genetic Algorithms — Including Example Code](https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3)


#### Papers

 - [Deep Neuroevolution: Genetic Algorithms are a Competitive Alternative for Training Deep Neural Networks for Reinforcement Learning](https://arxiv.org/pdf/1712.06567.pdf)
 - [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf)

---

### Project of the Week
The project is to implement a ES or GA.
In the [**Week6 repository**](Week6) you can find a basic implementation of the paper [Evolution Strategies as a
Scalable Alternative to Reinforcement Learning](https://arxiv.org/pdf/1703.03864.pdf) to solve LunarLanderContinuous. You can modify it to play more difficult environments or add your ideas.

<br>

## 第七周 - 基于模型的强化学习

The algorithms studied up to now are model-free, meaning that they only choose the better action given a state. These algorithms achieve very good performance but require a lot of training data. Instead, model-based algorithms, learn the environment and plan the next actions accordingly to the model learned. These methods are more sample efficient than model-free but overall achieve worst performance. In this week you'll learn the theory behind these methods and implement one of the last algorithms.

----

### Material

- **Model-Based RL by Davide Silver (Deepmind) (concise version)**
  - [Integrating Learning and Planning](https://www.youtube.com/watch?v=ItMutbeOHtc&index=8&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
    - Model-Based RL Overview
    - Integrated architectures
    - Simulation-Based search
- **Model-Based RL by Sergey Levine (Berkley) (in depth version)**
  - [Learning dynamical systems from data](https://www.youtube.com/watch?v=yap_g0d7iBQ&index=9&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)
    - Overview of model-based RL
    - Global and local models
    - Learning with local models and trust regions
  - [Learning policies by imitating optimal controllers](https://www.youtube.com/watch?v=AwdauFLan7M&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&index=10)
    - Backpropagation into a policy with learned models
    - Guided policy search algorithm
    - Imitating optimal control with DAgger
  - [Advanced model learning and images](https://www.youtube.com/watch?v=vRkIwM4GktE&index=11&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)
    - Models in latent space
    - Models directly in image space
    - Inverse models


#### Papers

 - [Imagination-Augmented Agents for Deep Reinforcement Learning - 2017](https://arxiv.org/pdf/1707.06203.pdf)
 - [Reinforcement learning with unsupervised auxiliary tasks - 2016](https://arxiv.org/pdf/1611.05397.pdf)
 - [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning - 2018](https://arxiv.org/pdf/1708.02596.pdf)

---

### Project of the Week

As a [project](Week7), I chose to implement the model-based algorithm described in this [paper](https://arxiv.org/pdf/1708.02596.pdf).
You can find my implementation [here](Week7).
NB: Instead of implementing it on Mujoco as in the paper, I used [RoboSchool](https://github.com/openai/roboschool), an open-source simulator for robot, integrated with OpenAI Gym.

---

#### Suggested
  - :books: [World Models - Can agents learn inside of their own dreams?](https://worldmodels.github.io/)

<br>

## 第八周 - 高级概念和你选择的项目

This last week is about advanced RL concepts and a project of your choice.

----

### Material

- Sergey Levine (Berkley)
  - [Connection between inference and control](https://www.youtube.com/watch?v=iOYiPhu5GEk&index=13&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=0s)
  - [Inverse reinforcement learning](https://www.youtube.com/watch?v=-3BcZwgmZLk&index=14&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=0s)
  - [Exploration (part 1)](https://www.youtube.com/watch?v=npi6B4VQ-7s&index=16&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=0s)
  - [Exploration (part 2) and transfer learning](https://www.youtube.com/watch?v=0WbVUvKJpg4&index=17&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=0s)
  - [Multi-task learning and transfer](https://www.youtube.com/watch?v=UqSx23W9RYE&index=18&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=0s)
  - [Meta-learning and parallelism](https://www.youtube.com/watch?v=Xe9bktyYB34&index=18&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3)
  - [Advanced imitation learning and open problems](https://www.youtube.com/watch?v=mc-DtbhhiKA&index=20&list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3&t=0s)
- David Silver (Deepmind)
  - [Classic Games](https://www.youtube.com/watch?v=N1LKLc6ufGY&feature=youtu.be)


 ---

### The final project
Here you can find some project ideas.
 - [Pommerman](https://www.pommerman.com/) (Multiplayer)
 - [AI for Prosthetics Challenge](https://www.crowdai.org/challenges/nips-2018-ai-for-prosthetics-challenge) (Challenge)
 - [Word Models](https://worldmodels.github.io/) (Paper implementation)
 - [Request for research OpenAI](https://blog.openai.com/requests-for-research-2/) (Research)
 - [Retro Contest](https://blog.openai.com/retro-contest/) (Transfer learning)


---

#### Suggested
* AlphaGo Zero
  - [Paper](https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ)
  - DeepMind blog post: [AlphaGo Zero: Learning from scratch](https://deepmind.com/blog/alphago-zero-learning-scratch/)
  - Arxiv Insights video: [How AlphaGo Zero works - Google DeepMind](https://www.youtube.com/watch?v=MgowR4pq3e8)
* OpenAI Five
  - OpenAI blog post: [OpenAI Five](https://blog.openai.com/openai-five/)
  - Arxiv Insights video: [OpenAI Five: Facing Human Pro's in Dota II](https://www.youtube.com/watch?v=0eO2TSVVP1Y)

<br>

## 最后四天 - 评论 + 分享

Congratulation for completing the 60 Days RL Challenge!! Let me know if you enjoyed it and share it!

See you!

## 最好的资源

:tv: [Deep Reinforcement Learning](https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3) - UC Berkeley class by Levine, check [here](http://rail.eecs.berkeley.edu/deeprlcourse/) their site.

:tv: [Reinforcement Learning course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) - by David Silver, DeepMind. Great introductory lectures by Silver, a lead researcher on AlphaGo. They follow the book Reinforcement Learning by Sutton & Barto.

:notebook: [Reinforcement Learning: An Introduction](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262193981/ref=sr_1_2?s=books&ie=UTF8&qid=1535898372&sr=1-2&keywords=reinforcement+learning+sutton) - by Sutton & Barto. The "Bible" of reinforcement learning. [Here](https://drive.google.com/file/d/1opPSz5AZ_kVa1uWOdOiveNiBFiEOHjkG/view) you can find the PDF draft of the second version.

## 额外的资源

:books: [Awesome Reinforcement Learning](https://github.com/aikorea/awesome-rl). A curated list of resources dedicated to reinforcement learning

:books: [GroundAI on RL](https://www.groundai.com/?text=reinforcement+learning). Papers on reinforcement learning
