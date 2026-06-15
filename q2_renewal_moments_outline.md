# 第 2 题：ordinary renewal process 的 n 阶统计量

设更新间隔为独立同分布随机变量

\[
X_1,X_2,\ldots \overset{\text{i.i.d.}}{\sim} F,
\]

并令

\[
S_k=X_1+\cdots+X_k,
\qquad
N(t)=\max\{k:S_k\le t\}.
\]

其中，\(N(t)\) 表示时刻 \(t\) 之前发生的更新次数。记 \(F^{*k}(t)=\mathbb P(S_k\le t)\) 为更新间隔分布的 \(k\) 次卷积。

## 1. 指示变量表示

由于事件 \(\{N(t)\ge k\}\) 等价于 \(\{S_k\le t\}\)，因此

\[
N(t)=\sum_{k=1}^{\infty}\mathbf 1_{\{S_k\le t\}}.
\]

从而一阶矩为

\[
\mathbb E[N(t)]
=\sum_{k=1}^{\infty}\mathbb P(S_k\le t)
=\sum_{k=1}^{\infty}F^{*k}(t).
\]

这就是 renewal function：

\[
m(t)=\mathbb E[N(t)].
\]

## 2. 阶乘矩

定义下降阶乘

\[
(N(t))_r=N(t)(N(t)-1)\cdots (N(t)-r+1).
\]

由于

\[
\binom{N(t)}{r}
=
\sum_{1\le k_1<\cdots<k_r}
\mathbf 1_{\{S_{k_r}\le t\}},
\]

而

\[
(N(t))_r=r!\binom{N(t)}{r},
\]

所以

\[
\mathbb E[(N(t))_r]
=
r!\sum_{1\le k_1<\cdots<k_r}
\mathbb P(S_{k_r}\le t).
\]

固定最后一个下标 \(k_r=k\)。前面 \(r-1\) 个下标可以从 \(1,\ldots,k-1\) 中选取，因此一共有

\[
\binom{k-1}{r-1}
\]

种选择。由此得到

\[
\boxed{
\mathbb E[(N(t))_r]
=
r!\sum_{k=r}^{\infty}
\binom{k-1}{r-1}F^{*k}(t)
}.
\]

## 3. 普通 n 阶矩

下降阶乘与普通幂之间满足

\[
N(t)^n
=
\sum_{r=1}^{n}
S(n,r)(N(t))_r,
\]

其中 \(S(n,r)\) 是第二类 Stirling 数。因此

\[
\boxed{
\mathbb E[N(t)^n]
=
\sum_{r=1}^{n}
S(n,r)\,r!
\sum_{k=r}^{\infty}
\binom{k-1}{r-1}F^{*k}(t)
}.
\]

这给出了 ordinary renewal process 的任意 n 阶统计量表达式。

## 4. 一阶与二阶情形检查

当 \(n=1\) 时：

\[
\mathbb E[N(t)]
=
\sum_{k=1}^{\infty}F^{*k}(t).
\]

当 \(n=2\) 时，由 \(N^2=(N)_1+(N)_2\)，得到

\[
\mathbb E[N(t)^2]
=
\sum_{k=1}^{\infty}F^{*k}(t)
+
2\sum_{k=2}^{\infty}(k-1)F^{*k}(t).
\]

因此

\[
\boxed{
\operatorname{Var}(N(t))
=
\sum_{k=1}^{\infty}F^{*k}(t)
+
2\sum_{k=2}^{\infty}(k-1)F^{*k}(t)
-
\left(\sum_{k=1}^{\infty}F^{*k}(t)\right)^2
}.
\]

## 5. 指数间隔作为数值验证

若

\[
X_i\sim \operatorname{Exp}(\lambda),
\]

则 renewal process 退化为速率为 \(\lambda\) 的 Poisson process：

\[
N(t)\sim\operatorname{Poisson}(\lambda t).
\]

因此理论上

\[
\mathbb E[N(t)]=\lambda t,
\qquad
\operatorname{Var}(N(t))=\lambda t,
\]

并且

\[
\mathbb E[(N(t))_r]=(\lambda t)^r.
\]

配套脚本 `q2_renewal_moments_validate.py` 会用 Monte Carlo 模拟验证上述公式。
