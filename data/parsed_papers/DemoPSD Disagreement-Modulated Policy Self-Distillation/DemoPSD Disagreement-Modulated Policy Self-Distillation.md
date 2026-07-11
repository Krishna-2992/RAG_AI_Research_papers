<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

## DemoPSD:Disagreement-Modulated Policy Self-Distillation

Yunhe Li *,1 , HaoShi *,2 , WenhaoLiu 2 , Mengzhe Ruan 1 , Hanxu Hou 3 Zhongxiang Dai 4 , Shuang Qiu †,1 , Linqi Song †,1

1 City University of Hong Kong 2 Tsinghua University 3 Shenzhen University of Advanced Technology 4 Chinese University of Hong Kong, Shenzhen

uuen.li@my.cityu.edu.hk shih22@mails.tsinghua.edu.cn {shuanqiu,linqi.song}@cityu.edu.hk

## Abstract

On-policy self-distillation (OPSD) has emerged as a practical method for training large language models (LLMs) to reason, where a single model acts as both the teacher and the student with different levels of information access. However, recent studies have found that the teacher's dense token-level supervision, conditioned on privileged information, can lead to overfitting to in-domain patterns, suppress exploration, and hurt cross-domain generalization, while also introducing a more fundamental issue: privileged information leakage , where the student encodes answer-dependent shortcuts that are unavailable at test time. We introduce DemoPSD , a novel framework that resolves such problems through the idea of selective adoption of teacher guidance : the student adopts the teacher's guidance when their distributions remain reasonably consistent, and relies more on its own reasoning when their distributions substantially diverge, indicating that the teacher's output is overly influenced by privileged information. Instead of fitting the full teacher distribution, DemoPSD steers the student toward a reverse-KL barycenter target , a weighted geometric combination of the teacher and student distributions, that naturally balances learning from the teacher with preserving the student's own reasoning capacity. We measure the difference between their distributions and use such a discrepancy to adaptively control the blending at each token position. We provably show that DemoPSD achieves (1) leakage attenuation , i.e., effective mitigation of privileged information leakage; and (2) exploration preservation , i.e., preservation of exploration capacity under dense token-level distillation. Extensive experiments on SciKnowEval across four scientific fields show that DemoPSD outperforms both GRPO and SDPO while maintaining higher training entropy and robustly generalizing to out-of-distribution GPQA benchmarks.

## 1. Introduction

Reinforcement learning with verifiable rewards (RLVR) has become a central paradigm for post-training large language models on reasoning tasks (Shao et al., 2024, DeepSeek-AI, 2025, Yu et al., 2026a). Methods such as Group Relative Policy Optimization (GRPO) train models by sampling multiple rollouts per question and using

* Equal contribution.

† Corresponding author.

<!-- image -->

- (a) Policy entropy over training steps. DemoPSD maintains 33-98% higher entropy than SDPO across all domains, avoiding policy entropy collapse.
- (b) Best-of-16 Validation accuracy for each SciKnowEval domain over training steps.

Figure 1: DemoPSD preserves higher entropy (left), which translates into better best@16 performance (right).

outcome correctness as a reward signal. While effective, RLVR suffers from a fundamental credit assignment bottleneck : standard RLVR methods distribute a rollout-level reward uniformly among all tokens in a rollout, offering coarse token-level credit signals that fail to distinguish individual token contributions (Hübotter et al., 2026).

On-policy distillation (OPD) addresses this bottleneck by introducing dense, token-level supervision from a teacher model on the student's self-generated trajectories (Agarwal et al., 2024, Gu et al., 2024, Lu and Thinking Machines Lab, 2025). Unlike off-policy distillation, which trains on teacher-generated texts and suffers from compounding exposure bias (Ross et al., 2011), OPD allows the student to learn from its own distribution while receiving rich feedback. This paradigm has been widely adopted in industry, including Qwen3 (Qwen Team, 2025) and DeepSeek-V4 (DeepSeek-AI, 2026), establishing OPD as a practical complement to RLVR.

Aparticularly appealing variant is on-policy self-distillation (OPSD) (e.g., Zhao et al., 2026, Hübotter et al., 2026), where a single model serves as both teacher and student. The teacher is the same model conditioned on privileged information, such as a verified reasoning trace or ground-truth answer, while the student receives only the question. OPSD eliminates the need for an external teacher and has demonstrated severalfold improvements in token efficiency over GRPO (Zhao et al., 2026, Shenfeld et al., 2026). However, recent theoretical and empirical analysis has revealed a critical failure mode: privileged information leakage (Yang et al., 2026). Because the teacher conditions on privileged information y ∗ that the student never observes at test time, the OPSD objective contains an irreducible mutual information gap I ( yt ; y ∗ | x , y &lt; t ) &gt; 0 , which is a conditional mutual information with x the input question, y &lt; t the generated prefix, yt the next token, and y ∗ the privileged information available only to the teacher. A positive value indicates that, even after conditioning on the question and generated prefix, the privileged signal still provides additional information about the next token, drving the student to encode answer-dependent shortcuts. This manifests as early performance gains followed by gradual degradation. As a result, the student may internalize cues tied to the privileged information instead of acquiring transferable reasoning strategies (Yang et al., 2026).

This failure mode reflects a broader tension between benefiting from the teacher's guidance and pre- serving the student's ability to reason independently. Privileged information leakage is a symptom of a more fundamental design choice: standard OPSD optimizes the student to imitate the teacher's privileged conditional distribution at every token. This objective is problematic for two major reasons: first, the teacher's distribution at certain positions (e.g., numerical answers, solution-revealing steps) is shaped by privileged information rather than transferable reasoning; second, directly matching the teacher's distribution at every token can suppress the student's own reasoning capacity when privileged information is unavailable.

Several recent studies have proposed different mechanisms to address this challenge. RLSD (Yang et al., 2026) avoids leakage entirely by using self-distillation only for magnitude estimation. HDPO (Ding, 2026) restricts privileged distillation to 'cliff prompts.' EGRSD (Ke et al., 2026) gates distillation by teacher entropy. SRPO (Li et al., 2026) routes entire samples between GRPO and SDPO based on rollout correctness. DASD (Zhang et al., 2026) uses entropy-routed directional supervision, pulling low-entropy tokens toward the privileged teacher while pushing high-entropy tokens away to preserve exploration. GATES (Stein et al., 2026) uses consensus among multiple teachers. The aforementioned methods share a common intuition: not all tokens are equally trustworthy . Yet they all rely on indirect proxies such as the teacher's entropy, sample correctness, student entropy, or multi-teacher consensus rather than directly measuring how much the teacher's prediction is influenced by privileged information . How to design the distillation target itself to balance teacher-guided learning with the student's own reasoning, however, has received relatively little attention.

Our work introduces DemoPSD, a novel framework that addresses this challenge in standard OPSD through the principle of selective adoption of teacher guidance : the student adopts the teacher's guidance when their distributions are reasonably consistent, and relies more on its own reasoning when the teacher's distribution substantially diverges from the student's, indicating that the teacher's output is overly influenced by privileged information. Rather than fitting the full teacher distribution, DemoPSD trains the student toward a reverse-KL barycenter target , which is a weighted geometric combination of the teacher's and student's distribution:

<!-- formula-not-decoded -->

where α t is a per-token leakage attenuation coefficient determined by the disagreement between the distributions of the teacher and the student, controlling how far the target is interpolated from the teacher's distribution toward the student's distribution. When α t is sufficiently small, the privileged information y ∗ does not substantially shift the teacher's distribution, the target therefore remains close to the teacher. As α t increases, the teacher's distribution becomes more strongly shaped by y ∗ . Forcing the student to directly match the teacher would encode answer-dependent shortcuts into the student, which is precisely the privileged information leakage. The target in (1) is therefore interpolated further toward the student's distribution to attenuate leakage while preserving the student's unprivileged reasoning capacity. Figure 1 previews our main empirical results based on the principle of selective adoption of teacher guidance. DemoPSD preserves substantially higher training entropy than SDPO across all domains, which further translates into improved best@16 results.

Our contribution. Specifically, our main contributions are three-fold:

1. We propose a novel on-policy self-distillation algorithm DemoPSD that effectively prevents the student model from overfitting the teacher's distribution, thereby improving both in-domain and cross-domain reasoning capabilities and reducing privileged information leakage during self-distillation.
2. We theoretically prove two key properties of DemoPSD: (1) leakage attenuation , i.e., the disagreementweighted reverse-KL barycenter target reduces the rate of privileged information leakage; and (2) ex-

- ploration preservation , i.e., DemoPSD retains the student's exploration capacity under dense token-level distillation.
3. We conduct extensive experiments on SciKnowEval across four scientific domains. Our empirical results show that DemoPSD consistently outperforms SDPO and GRPO, achieving up to a 4.2% improvement in @16 accuracy while maintaining 35-97% higher training entropy. On the out-of-distribution GPQA benchmark, DemoPSD maintains strong overall accuracy and demonstrates robust generalization, whereas SDPO exhibits a gradual performance decline.

## 2. Related Work

On-Policy Distillation and Self-Distillation. Recent OPD methods such as GKD (Agarwal et al., 2024) and MiniLLM (Gu et al., 2024) train the student on model-generated trajectories while using teacher distributions as dense supervision. This on-policy design is motivated by the classic imitation-learning observation that training only on expert-generated states can suffer from compounding errors under distribution shift (Ross et al., 2011). Subsequent work further studies OPD from different perspectives: REOPOLD (Ko et al., 2026) relaxes on-policy distillation for more efficient reasoning, Veto (Jang et al., 2026) reformulates the distillation target to improve training stability, and Song and Zheng (2026) provide a broader survey of OPD methods for large language models. Another line of research focuses on the on-policy self-distillation problem. SelfDistilled Reasoner (Zhao et al., 2026) studies the setting where a single model provides its own on-policy distillation signal for reasoning. SDPO (Hübotter et al., 2026) further frames reinforcement learning through self-distillation, converting sparse outcome feedback into dense training signals. Related variants explore complementary design choices: SD-Zero (He et al., 2026) uses self-revision to transform binary rewards into dense supervision, UniSD (Jin et al., 2026) proposes a unified framework for self-distillation in LLMs, and CRISP (Sang et al., 2026) applies iterative self-policy distillation to compressed reasoning. As shown in §4.3, DemoPSD instead uses a disagreement-dependent geometric target that preserves dense supervision on low-disagreement tokens while attenuating teacher-induced signals on high-disagreement tokens.

Addressing Privileged Information Leakage Recent work has begun to analyze and mitigate failure modes in on-policy self-distillation. Yang et al. (2026) study self-distilled RLVR and identify privileged information leakage as a key concern. HDPO (Ding, 2026) focuses privileged self-distillation on cliff prompts, while PBSD (Yu et al., 2026b) moves beyond direct KL matching through preference-based self-distillation and reward regularization. Other methods adjust when or how self-distillation is applied. SRPO (Li et al., 2026) unifies group-relative optimization and self-distillation through sample routing, DASD (Zhang et al., 2026) adapts supervision according to the direction of the self-distillation signal, and PAINT (Tan and Hong, 2026) interpolates between partial- and full-solution prompts. Kim et al. (2026) analyzes why self-distillation can degrade reasoning ability. In contrast, DemoPSD keeps the token-level distillation setting but changes the distributional target itself: the reverse-KL barycenter adaptively interpolates between the privileged teacher and the unprivileged student according to teacher-student disagreement.

Mixture Distributions and Entropy Dynamics AMiD (Shin et al., 2026) introduces α -mixture assistant distributions for knowledge distillation, making it conceptually related to our reverse-KL barycenter target, although AMiD does not address privileged self-distillation. Entropy stability has also emerged as an important issue in large-scale RL training systems such as DAPO (Yu et al., 2026a) and in explicit entropy-control methods such as EntroPIC (Yang et al., 2025), motivating our focus on preserving exploration during dense distillation. PACED (Xu et al., 2026) studies distillation and on-policy self-distillation at the frontier of student competence, which is complementary to our token-level disagreement-based target adaptation.

## 3. Background and Problem Setting

## 3.1. Reinforcement Learning with Verifiable Rewards

We consider the standard RLVR setup for post-training LLMs. Given a dataset of questions D = { ( xi , a ∗ i ) } N i = 1 where a ∗ i is the verifiable answer, the model πθ ( · | x ) generates rollouts y ∼ πθ ( · | x ) and receives a binary reward r ( y , a ∗ ) ∈ { 0, 1 } based on outcome correctness. GRPO (Shao et al., 2024) estimates advantages from these rewards within each rollout group and optimizes:

<!-- formula-not-decoded -->

where ˆ A ( y ) is the group-relative advantage. For a group of G rollouts { yj } G j = 1 sampled for the same question x , GRPO computes

<!-- formula-not-decoded -->

with a small constant ϵ &gt; 0 for numerical stability. The KL regularizer is defined as

<!-- formula-not-decoded -->

One of the fundamental limitations is that r provides only an outcome reward per response, offering no guidance on which tokens contributed more to success or failure.

## 3.2. On-Policy Self-Distillation

Reinforcement learning via self-distillation (SDPO) (Hübotter et al., 2026) addresses the credit assignment bottleneck by introducing dense, token-level supervision from a privileged version of the same model. The teacher πθ ( · | x , y ∗ ) is the current model conditioned on both the question x and privileged information y ∗ (e.g., a verified reasoning trace or ground truth), while the student πθ ( · | x ) receives only the question. Given a student-generated rollout ˆ y ∼ πθ ( · | x ) , the SDPO objective minimizes per-token divergence:

<!-- formula-not-decoded -->

The key insight is that the teacher leverages its access to the privileged y ∗ to provide richer feedback than an outcome reward. The stopgrad operator prevents gradients from flowing into the teacher, which keeps the teacher from shifting toward the student and ignoring y ∗ .

## 3.3. The Privileged Information Leakage Problem

While SDPO achieves impressive token efficiency, Yang et al. (2026) proved that the setting is fundamentally ill-posed. Since the teacher conditions on privileged information y ∗ that the student cannot observe, the SDPO objective contains an irreducible mutual information gap:

<!-- formula-not-decoded -->

This gap implies the student can never perfectly achieve the teacher's conditional distribution, regardless of capacity. At the gradient level, per-sample gradients include an y ∗ -specific deviation whose variance is proportional to this mutual information. At the early stage of training, the beneficial gradient component dominates, producing rapid training reward improvement. However, as the student approaches the teacher's marginal distribution, the deviation takes over, driving the student to encode x → y ∗ correlations, which is exactly the privileged information leakage explained in Yang et al. (2026). Empirically, SDPO performance peaks early and then gradually degrades during the remaining training stage.

The leakage problem points to a deeper issue: the teacher's distribution is not always an appropriate target for direct fitting. Even if leakage could be eliminated, a student who exactly replicates the teacher has lost its own capacity for reasoning. What we need instead is a training target that adaptively incorporates the teacher's guidance while preserving the student's own reasoning ability.

## 4. The Proposed Method: DemoPSD

This section presents the proposed method DemoPSD, built on the principle of selective adoption of teacher guidance, i.e., the student follows the teacher's guidance when privileged information does not heavily distort the teacher's distribution so that it diverges substantially from the student's distribution. Below, we first describe how to measure teacher-student disagreement (§4.1), then introduce the reverse-KL barycenter target that implements selective adoption (§4.2), derive its loss and gradient (§4.3), and describe the full training procedure (§4.4).

## 4.1. Measuring Teacher-Student Disagreement

The key ingredient of DemoPSD is measuring the disagreement between the teacher's and student's predictions at each token position: one made with privileged information, and one made without . Token positions where these predictions remain reasonably consistent are likely to reflect transferrable knowledge that the student can safely adopt, while positions where they substantially disagree indicate that the teacher's output has been overly influenced by privileged information.

Disagreement and Leakage Attenuation Coefficient. At each token position t , the privileged teacher's prediction is obtained by conditioning the model on the question x , the privileged information y ∗ , and the student's rollout prefix ˆ y &lt; t . For notational convenience, we write this distribution as π t T ( v , y ∗ ) as shorthand for πθ ( v | x , y ∗ , ˆ y &lt; t ) . The corresponding student's prediction conditions only on x and ˆ y &lt; t , and we write it as π t S ( v ) as shorthand for πθ ( v | x , ˆ y &lt; t ) . We use these shorthand notations when no ambiguity arises and revert to the full conditional form when the conditioning context needs to be made explicit. The privileged prediction provides a rich teacher signal because it receives y ∗ , while the student's prediction serves as the reference distribution for evaluating disagreement. We describe how y ∗ is incorporated into the model's context in §4.4. In practice, for training stability , we use a separate exponential moving average (EMA) copy of the student when computing the disagreement in (7) and the target distribution in (9); implementation details are summarized in Algorithm 1. We measure the disagreement dt between π T and π S by using the Jensen-Shannon divergence (JSD):

<!-- formula-not-decoded -->

From dt , we derive a leakage attenuation coefficient α t = f ( dt ) that controls how much the target shifts away from the privileged teacher and toward the student's own prediction. We require f to be monotonically increasing so that larger teacher-student disagreement leads to stronger leakage attenuation, and to satisfy f ( 0 ) = 0 so that the target reduces to the teacher distribution when the two predictions match. We also use a saturating form with lim d → ∞ f ( d ) = α max , which prevents extreme disagreement from completely discarding the teacher signal. The cap α max is an empirical hyperparameter: setting it too large assigns too little weight to the teacher distribution and can weaken useful distillation signals. We realize f via a rescaled sigmoid:

<!-- formula-not-decoded -->

where β controls the sensitivity of the gate to teacher-student disagreement: a larger β makes α t increase more sharply with small changes in dt , causing the target to move away from the privileged teacher more aggressively, whereas a smaller β yields a smoother transition and retains more teacher signal under moderate disagreement. This realization has two key properties: (1) When α t is sufficiently small, i.e., the two distributions are reasonably consistent, it is safe to distill; (2) As α t increases to α max , i.e., they strongly disagree, distillation becomes increasingly risky.

## 4.2. Reverse-KL Barycenter Target

Given the coefficient α t , we define the distillation target as a geometric mixture of the two distributions. The target at token position t is:

<!-- formula-not-decoded -->

This distribution is the reverse-KL barycenter of the privileged teacher and the student distributions under the weight α t , defined by

<!-- formula-not-decoded -->

where ∆ ( V ) denotes the probability simplex over the vocabulary V . The reverse-KL barycenter in (10) defines the weighted centroid of a collection of probability distributions, i.e., π t T and π t S in this problem, under the reverse KL divergence. Equivalently, this target interpolates between the teacher and student distributions in log-probability space,

<!-- formula-not-decoded -->

where Z α t is the normalization term for (9) defined as:

<!-- formula-not-decoded -->

Geometric Mixture vs Arithmetic Mixture. The geometric mixture is chosen over the arithmetic alternative (( 1 -α t ) π t T + α t π t S ) for two reasons: (1) Because probabilities are multiplied, a token receives substantial target mass only when it is supported by both the privileged teacher and the student. Thus, tokens endorsed primarily by the teacher but assigned very low probabilities by the student are naturally suppressed, whereas an arithmetic mixture would still allocate them non-trivial mass. (2) When the teacher and student distributions have different modes, an arithmetic mixture can average the modes into a diffuse target with inflated entropy. The geometric mixture avoids this mode-averaging effect, yielding a sharper and more coherent training signal. This is consistent with AMiD's (Shin et al., 2026) observation that mixture geometry controls mode-covering versus mode-seeking behavior.

## 4.3. Loss Function

The student is trained to minimize the reverse KL divergence objective toward the reverse-KL barycenter target:

<!-- formula-not-decoded -->

Directly computing and differentiating through the normalization term Z α t would make the optimization complicated. However, the full target distribution is wrapped with stop-gradient: the teacher π t T , the reference student π t S , the weight α t are all treated as fixed during the backward pass. Consequently, Z α t becomes constant and the optimization hence avoids directly backpropagating through it. Then the gradient of L DemoPSD takes the following form:

<!-- formula-not-decoded -->

The DemoPSD gradient keeps the same reverse-KL score-function form while scaling the teacher-induced log-ratio signal by the disagreement-based factor ( 1 -α t ) . As illustrated in (13), positions with larger teacher-student disagreement contribute a weaker distillation signal, reducing the tendency to backpropagate privileged information dependent guidance from the teacher.

## 4.4. Privileged Information Injection and Training Procedure

Algorithm 1 summarizes the full DemoPSD algorithm.

Privileged Information Injection. Generally, for each training prompt x with privileged information y ∗ , and a relevant student-generated rollout ˆ y , we construct the teacher's input by prepending y ∗ to the prompt context:

<!-- formula-not-decoded -->

The student model receives only:

<!-- formula-not-decoded -->

both of which share the same model. The only difference is whether the privileged information y ∗ is included in the conditioning context.

Reprompting Mechanism. For a correct rollout, the generated response itself contains rich solution information and can therefore serve as privileged information for the teacher model. Following Hübotter et al. (2026), we use a reprompting mechanism to construct this privileged context: for each prompt group, if at least one rollout is correct, we randomly select one correct rollout as y ∗ and insert it into the teacher context above; if no rollout is correct, no reliable privileged teacher context can be formed, so the prompt is skipped for distillation. As explained in Hübotter et al. (2026), model performance is not sensitive to syntactic variations of the reprompting template, so we use a similar template to instantiate the privileged information for the teacher model.

## Algorithm 1 DemoPSD

- 1: for each training step do

Require: Dataset D , student model πθ , sensitivity β , max leakage attenuation coefficient α max

- 2: Sample batch { xi } from D ; generate rollouts ˆ yi ∼ πθ ( · | xi )
- 4: for each filtered prompt ( x , y ∗ , ˆ y ) do
- 3: Filter: keep only prompts with at least one correct rollout ( r ( ˆ yi , a ∗ i ) = 1 )
- 5: Obtain the teacher policy: π t T ← πθ ( · | x , y ∗ , ˆ y &lt; t ) for all t
- 7: Compute the distributional disagreement dt via (7)
- 6: Obtain the student policy: π t S ← πθ ( · | x , ˆ y &lt; t ) for all t
- 8: Compute the leakage attenuation coefficient α t via (8)
- 10: end for
- 9: Compute the reverse-KL barycenter target via (9)
- 11: Update θ via gradient descent on L DemoPSD ( θ )
- 12: end for

## 5. Theoretical Analysis

This work aims to solve a central question that how we preserve the token-level distributional supervision while suppressing privileged information leakage caused by conditioning the teacher on y ∗ ? Standard OPSD exploits dense teacher distributions but is vulnerable to leakage. In this section, we provide a detailed theoretical analysis of DemoPSD from the perspectives of leakage attenuation and exploration preservation.

As we have analyzed in §4.1, in practice, we maintain a separate EMA copy of the student as the unprivileged reference, and construct the teacher based on the EMA copy for stability. Let π ¯ θ denote this EMA copy of the current student πθ . Throughout this section, both the privileged teacher distribution π T and the student distribution π S in the target are obtained by conditioning π ¯ θ on the corresponding privileged or unprivileged contexts. Following Yang et al. (2026), we define the per-step leakage rate as the expected squared magnitude of the privileged deviation:

<!-- formula-not-decoded -->

In this definition, ∆ t ∈ R |V| is a vector indexed by tokens in the vocabulary V , and ∥ ∆ t ∥ 2 = ∑︀ v ∈V ∆ t ( v ) 2 is its squared ℓ 2 norm, measuring the total squared log-probability shift induced by y ∗ at position t . Throughout this section, we write π t θ ( v ) : = πθ ( v | x , ˆ y &lt; t ) for the student distribution at position t , and π α t target : = π α t target for the reverse-KL barycenter target defined in (9) with normalization constant Z α t in (11). Consequently, ∆ t directly measures the influence of y ∗ on the model's own prediction, rather than a discrepancy between two independent models.

Theorem 1 (Leakage Attenuation) . The effective leakage rate induced by DemoPSD satisfies:

<!-- formula-not-decoded -->

where the strict inequality holds whenever Pr ( α t &gt; 0 ) &gt; 0 . Moreover, the attenuation is strongest where leakage risk is greatest: since α t is monotonically increasing in dt and dt correlates positively with ∥ ∆ t ∥ (both measure the divergence between π t T and π t S ), positions with larger privileged deviation tend to receive larger α t and hence stronger suppression.

The key message is that DemoPSD does not merely reduce the average leakage, it selectively attenuates positions that contribute most to leakage. The factor ( 1 -α t ) scales down the privileged deviation ∆ t in the gradient, and since α t correlates positively with ∥ ∆ t ∥ , the suppression is strongest precisely where it is most needed. Full proof is in Appendix A.1.

Theorem 2 (Exploration Preservation) . Let π α t target ( v ) ∝ (︀ π t T ( v , y ∗ ) )︀ 1 -α t (︀ π t S ( v ) )︀ α t be the reverse-KL barycenter target with the leakage attenuation coefficient α t ∈ [ 0, α max ] , and write ∆ t ( v ) = log π t T ( v , y ∗ ) -log π t S ( v ) for the log-ratio. The full-teacher target minimized by SDPO is the special case α t = 0 , namely π t T . Assume the privileged signal is positively aligned with the model's own unprivileged prediction, i.e.,

<!-- formula-not-decoded -->

under every geometric interpolation q t γ ∝ ( π t T ) γ ( π t S ) 1 -γ , γ ∈ [ 0, 1 ] , between the student and teacher's distributions. Then the DemoPSD target preserves strictly more entropy than the full-teacher target, with the ordering

<!-- formula-not-decoded -->

̸

holding with strict inequalities whenever 0 &lt; α t and π t T = π t S . Moreover, the entropy gain H ( π α t target ) -H ( π t T ) ≥ 0 over the full-teacher target is non-decreasing in α t : the more the teacher's prediction depends on the privileged y ∗ (the larger α t ), the more exploration capacity DemoPSD retains relative to SDPO.

Proof sketch. We write the family q t γ ( v ) as q t γ ( v ) ∝ π t S ( v ) e γ ∆ t ( v ) , which is an exponential family with parameter γ and sufficient statistic ∆ t . Differentiating H ( q t γ ) = -E q t γ [ log q t γ ] and using d d γ E q t γ [ f ] = Cov q t γ ( f , ∆ t ) yields d H d γ = -Cov q t γ ( ∆ t , log q t γ ) . Substituting log q t γ = log π t S + γ ∆ t -log Z γ and expanding gives (18). Under condition (16), both terms are non-positive for γ &gt; 0 , so H ( q t γ ) is decreasing. Since 1 -α t &lt; 1 , the DemoPSD target q 1 -α t has strictly higher entropy than the OPSD target q 1 . Full proof is in Appendix A.2.

The result follows by tracking the entropy along the geometric path q t γ ∝ ( π t T ) γ ( π t S ) 1 -γ that connects the unprivileged distribution ( q 0 = π t S ) to the full teacher ( q 1 = π t T ); the DemoPSD target sits at q 1 -α t = π α t target , strictly short of the teacher. Along this path the entropy obeys

<!-- formula-not-decoded -->

which separates the entropy change into two terms. The first term -γ Var q t γ [ ∆ t ] is the intrinsic entropy cost of incorporating the privileged signal: any nonconstant multiplicative shift reduces entropy, and this cost grows with the γ . The second term -Cov q t γ ( ∆ t , log π t S ) captures the interaction with the model's existing predictions, and condition (16) requires their positive correlation: tokens to which the model already assigns high probability receive a larger boost from y ∗ . This is the natural regime for self-distillation, where the teacher is the same model with additional answer information and hence predominantly sharpens existing predictions rather than contradicting them. Because DemoPSD halts the interpolation at γ = 1 -α t &lt; 1 rather than at the full teacher γ = 1 , it never pays the final, steepest portion of this entropy cost; the entropy it saves grows with α t , consistent with the 33-98% entropy improvements over SDPO observed in Table 3.

These two results characterize why the reverse-KL barycenter target is suitable for privileged selfdistillation. Theorem 1 shows that disagreement-dependent weighting attenuates the contribution of privileged deviations, thereby reducing the pressure to imitate privileged information dependent teacher signals. Theorem 2 complements this result by showing that, under the stated covariance condition, the barycenter target remains strictly more entropic than the full privileged-teacher target, preserving exploration where the teacher distribution is strongly shaped by y ∗ . Combined with the gradient analysis in §4.3, these results indicate that DemoPSD retains dense token-level supervision when the teacher and student agree, while down-weighting teacher signals that are likely to reflect privileged information on high-disagreement positions.

Table 1: Main results on SciKnowEval. Bold indicates the best method per metric. DemoPSD consistently outperforms both GRPO and SDPO across all four domains and all metrics.

|           | mean@16   | mean@16   | mean@16   | maj@16   | maj@16   | maj@16   | best@16   | best@16   | best@16   |
|-----------|-----------|-----------|-----------|----------|----------|----------|-----------|-----------|-----------|
| Domain    | GRPO      | SDPO      | DemoPSD   | GRPO     | SDPO     | DemoPSD  | GRPO      | SDPO      | DemoPSD   |
| Biology   | 33.51     | 36.88     | 39.25     | 34.84    | 38.07    | 40.64    | 58.36     | 64.04     | 68.51     |
| Chemistry | 65.83     | 71.70     | 72.98     | 66.72    | 72.41    | 73.71    | 80.47     | 85.94     | 90.05     |
| Material  | 76.32     | 76.13     | 76.53     | 76.50    | 76.24    | 76.71    | 80.24     | 81.69     | 81.79     |
| Physics   | 66.31     | 68.98     | 71.64     | 70.52    | 71.88    | 74.24    | 82.59     | 85.51     | 88.13     |
| Average   | 60.49     | 63.42     | 65.10     | 62.14    | 64.65    | 66.33    | 75.42     | 79.30     | 82.12     |

Table 2: Out-of-distribution generalization on GPQA Extended. Material science has no GPQA counterpart. Values are taken at the final training stage (mean over the last three evaluations). DemoPSD remains stable and improves slightly across all three GPQA domains, whereas SDPO degrades substantially over training (Figure 3).

| Method   |   Biology |   Chemistry |   Physics |   Average |
|----------|-----------|-------------|-----------|-----------|
| SDPO     |     57.81 |       28.62 |     52.99 |     46.47 |
| DemoPSD  |     61.42 |       41.75 |     59.98 |     54.38 |

## 6. Experiments

We evaluate DemoPSD on scientific reasoning benchmarks, comparing against SDPO and GRPO as the primary baselines. The experiments focus on three aspects: in-domain accuracy, training entropy as an empirical indicator of exploration preservation, and out-of-distribution generalization as a proxy for reduced privileged information leakage.

## 6.1. Experimental Setup

Base Model. We use Qwen3-4B-Instruct (Qwen Team, 2025) as the base model for all experiments.

Training Data. We train on SciKnowEval (Feng et al., 2024), a multi-domain scientific reasoning benchmark formulated as 4-choice multiple-choice questions. We train and evaluate separately on four domains: biology, chemistry, material science, and physics.

Evaluation Benchmarks. Weevaluate the performance on the following benchmarks to assess both in-domain accuracy and out-of-domain generalization:

- SciKnowEval (in-domain): Domain-matched test sets for each of the four scientific domains.
- GPQA Extended (Rein et al., 2023) (out-of-domain): Graduate-level science questions in biology, chemistry, and physics. It is used to assess generalization beyond the training distribution.

(a) Validation accuracy (mean@16) on SciKnowEval over training steps. DemoPSD maintains higher accuracy than SDPO across training, with the largest margins observed in biology and physics.

<!-- image -->

- (b) Validation mean@16 vs. the sensitivity parameter β per domain. The dashed line is the SDPO baseline. DemoPSD remains competitive with or above the SDPO baseline across β ∈ [ 25, 100 ] , while the optimal choice of β varies by domain.

Figure 2: (a) Validation accuracy curves across four domains of SciKnowEval. (b) Sensitivity to β .

Evaluation Metrics. For each test prompt, we sample 16 rollouts and report three complementary metrics that capture different aspects of model quality:

- mean@16 : Average accuracy across 16 sampled rollouts.
- best@16 : Best accuracy among 16 rollouts.
- maj@16 : Accuracy of the majority-voted answer across 16 rollouts.

Baselines. We compare DemoPSD against two baselines:

- GRPO (Shao et al., 2024): The standard RLVR baseline that estimates group-relative advantages from binary outcome rewards.
- SDPO (Hübotter et al., 2026): The on-policy self-distillation baseline.

All three methods use the same codebase, training infrastructure, base model, and training data, differing only in their optimization objectives: GRPO uses outcome-level reward, SDPO uses dense teacher supervision, and DemoPSD uses disagreement-modulated reverse-KL barycenter targets.

Hyperparameters. All methods share the following settings: learning rate 1 × 10 -6 , batch size 64, 8 rollouts per prompt for training, max prompt length 2048, max response length 16384, 10 warmup steps, 3 training epochs. For distillation-based methods (SDPO and DemoPSD), we additionally use topk = 100 for distillation, EMA rate η = 0.05 , and training temperature = 1.0 with validation temperature = 0.7 . DemoPSD-specific parameters: α max = 0.15 . The sensitivity parameter β is tuned per domain (see §6.5). GRPO uses a KL penalty coefficient β KL = 0.04 and clips importance sampling ratios at 2.0. Training uses 8 NVIDIA H20 GPUs with FSDP, vLLM for rollout generation, and flash attention.

## Out-of-Distribution Generalization on GPQA Extended (mean@16)

Figure 3: Out-of-distribution generalization on GPQA Extended. Each panel tracks GPQA accuracy over training for one domain (material science has no GPQA counterpart). SDPO reaches its best OOD accuracy early and then degrades as training progresses, consistent with accumulating in-domain overfitting and privileged information leakage. In contrast, DemoPSD maintains stable OOD performance and achieves an improvement over training.

<!-- image -->

## 6.2. Main Results

Table 1 reports the accuracy results across all four scientific domains. On average, DemoPSD improves over SDPO by 1.68 on mean@16, 1.68 on maj@16, and 2.82 on best@16. The best@16 improvement is notably larger, indicating that DemoPSD's preserved exploration entropy surfaces higher-quality reasoning paths during sampling. Compared to GRPO, the total gain from DemoPSD is 5.21 on mean@16, demonstrating that the combination of dense supervision and selective adoption leads to substantial improvement.

Figure 2a shows how the validation accuracy mean@16 changes with training steps. DemoPSD matches or outperforms SDPO throughout training, and the difference grows in later epochs. This agrees with our theoretical prediction that reducing leakage becomes more helpful as the student moves closer to the teacher's distribution. Figure 1b reports the corresponding best@16 accuracy. The improvement is especially clear under best@16 and grows across training, indicating that the higher-entropy policy maintains broader solution coverage.

## 6.3. Out-of-Distribution Generalization

We evaluate the model's out-of-distribution generalization capability on GPQA Extended dataset, which contains graduate-level science questions that differ substantially from SciKnowEval in format, difficulty, and question style. Table 2 reports the accuracy at convergence, and Figure 3 traces the full GPQA learning curves.

Although SDPO and DemoPSD start from comparable OOD accuracy, their performance evolves in substantially different directions over training (Figure 3). SDPO reaches its best OOD performance early, but subsequently degrades across all three GPQA domains; the largest drop occurs in chemistry, where accuracy decreases from 40.45 to 28.62 . This mirrors the in-domain leakage-degradation pattern (§3.3): by collapsing onto the teacher, SDPO overfits the training distribution and loses the exploratory capacity necessary to transfer to novel questions. In contrast, DemoPSD maintains stable OOD accuracy throughout training and achieves a measurable improvement, ending 7.91 above SDPO on average.

Table 3: Training dynamics at the final step. DemoPSD maintains substantially higher entropy than SDPO across all domains.

| Domain    | Method       | Entropy     | ∆ Ent.   | mean ¯ α t   | mean ¯ d t   | Active%   |
|-----------|--------------|-------------|----------|--------------|--------------|-----------|
| Biology   | SDPO DemoPSD | 0.602 0.816 | - +35.5% | - 0.055      | - 0.046      | - 64.8    |
| Chemistry | SDPO DemoPSD | 0.322 0.555 | - +72.4% | - 0.036      | - 0.037      | - 84.0    |
| Material  | SDPO DemoPSD | 0.150 0.297 | - +98.0% | - 0.033      | - 0.031      | - 68.8    |
| Physics   | SDPO DemoPSD | 0.385 0.511 | - +32.7% | - 0.040      | - 0.026      | - 90.6    |

Table 4: Sensitivity to β (mean@16). All configurations use α max = 0.15 .

| β    | Biology   | Chemistry   | Material   | Physics   |
|------|-----------|-------------|------------|-----------|
| 15   | -         | 71.93       | -          | -         |
| 25   | -         | 72.98       | 76.46      | -         |
| 50   | -         | 71.90       | 76.53      | 70.55     |
| 70   | 39.25     | -           | -          | -         |
| 100  | 36.88     | -           | 76.06      | 71.64     |
| SDPO | 36.88     | 71.70       | 76.13      | 68.98     |

## 6.4. Training Dynamics

To understand how DemoPSD achieves its accuracy gains, we examine training dynamics, including entropy, disagreement, and hedging behavior at the final training step (Table 3).

Entropy Preservation. DemoPSD maintains 33-98% higher final entropy than SDPO across all domains (Figure 1a). The largest entropy gap appears in material science (+98.0%), where SDPO's entropy drops to 0.150, close to entropy collapse.

Disagreement Sparsity. The average leakage attenuation coefficient ¯ α t stays consistently low (0.0330.055), while the mean disagreement ¯ dt ranges from 0.026 to 0.046. These values indicate that the target remains close to the teacher distribution for most tokens, with strong attenuation applied only to a small subset of positions exhibiting substantial teacher-student disagreement. This pattern is consistent with the selective adoption principle: DemoPSD preserves the teacher signal on most tokens and applies disagreementmodulated attenuation only at positions where teacher and student's predictions diverge.

Active Sample Fraction. The 'Active %' column in Table 3 denotes the fraction of training samples for which a valid privileged teacher context is available, namely samples whose prompt group contains at least one correct rollout. The active fraction correlates with domain difficulty.

Table 5: Statistics of disagreement-based attenuation at the final training step. For most tokens, the target remains close to the teacher distribution, while only a small subset of high-disagreement tokens is interpolated more strongly toward the student distribution.

| Statistic                                | Biology   | Chemistry   | Material   | Physics   |
|------------------------------------------|-----------|-------------|------------|-----------|
| Mean leakage attenuation coefficient α t | 0.055     | 0.036       | 0.033      | 0.040     |
| Mean disagreement d t                    | 0.046     | 0.037       | 0.031      | 0.026     |
| Active sample fraction                   | 64.8%     | 84.0%       | 68.8%      | 90.6%     |

(a) Distribution of per-token JSD disagreement dt . Each panel shows one domain at the final training step. The distribution is heavily right-skewed: the vast majority of tokens have near-zero disagreement, and only 2%-5% exceed 0.25 .

<!-- image -->

- (b) DemoPSD dynamics over training. Mean leakage attenuation coefficient α t (blue, left axis) and mean JSD disagreement dt (pink, right axis) over training batch per domain. Both quantities remain small and relatively stable.

Figure 4: Disagreement analysis of DemoPSD across four scientific domains.

## 6.5. Hyperparameter Sensitivity

The key hyperparameter of DemoPSD is β , which controls how sharply the leakage attenuation coefficient α t responds to disagreement. Table 4 shows the three best-performing β configurations for each domain.

A general pattern emerges that domains where the disagreement is smaller (e.g., physics with mean ¯ dt = 0.026 ) benefit from a higher β to amplify the weak disagreement signal, while domains with greater disagreement (e.g., biology with mean ¯ dt = 0.046 ) benefit from a lower β to avoid over-aggressive hedging. Across the range β ∈ [ 25, 100 ] , DemoPSD consistently matches or outperforms SDPO, demonstrating moderate robustness.

Remap vs. Threshold Mode. All top-performing configurations adopt the remapped alpha schedule in (8), which constrains α t to [ 0, α max ] and guarantees that the privileged teacher retains at least ( 1 -α max ) of the mixture weight. Figure 2b further illustrates how accuracy varies with β across domains.

## 6.6. Disagreement Analysis

To characterize how disagreement is distributed across tokens, we summarize statistics of the per-token disagreement dt and leakage attenuation coefficient α t for the best-performing DemoPSD run in each domain (Table 5).

Across all domains, the disagreement distribution is strongly right-skewed: most tokens exhibit negligible disagreement, allowing the student to remain closely aligned with the teacher, while only a small subset of tokens (typically around 5) shows substantial divergence and triggers stronger attenuation of the privileged teacher signal. This sparsity is beneficial because DemoPSD preserves the dense token-level supervision of OPSD for the vast majority of positions, while selectively attenuating potential leakage only where the teacher-student mismatch is pronounced. Figure 4a visualizes this sparsity directly. Figure 4b shows the evolution of the mean attenuation coefficient α t and disagreement dt over training steps.

## 7. Conclusion

We introduced DemoPSD, a self-distillation framework based on selective adoption of teacher guidance: instead of forcing the student to imitate the privileged teacher at every token, DemoPSD constructs a reverseKL barycenter target that adaptively balances teacher guidance with the student's own reasoning capacity. Our analysis shows that the disagreement-dependent barycenter weight directly modulates the teacherinduced signal in the training gradient: low-disagreement tokens retain dense teacher supervision, whereas high-disagreement tokens receive attenuated privileged guidance. We formalized this behavior through leakage attenuation and exploration preservation, showing how the proposed learning target reduces pressure to imitate privileged information while maintaining higher-entropy supervision. Empirically, DemoPSD improves over SDPO and GRPO across four scientific domains, maintains 35-98% higher training entropy, and generalizes robustly to out-of-distribution benchmarks.

## References

- Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations , volume 2024, pages 21246-21263, 2024.
- DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 , 2025.
- DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.
- Ken Ding. Hdpo: Hybrid distillation policy optimization via privileged self-distillation. arXiv preprint arXiv:2603.23871 , 2026.
- Kehua Feng, Xinyi Shen, Weijie Wang, Xiang Zhuang, Yuqi Tang, Qiang Zhang, and Keyan Ding. Sciknoweval: Evaluating multi-level scientific knowledge of large language models. arXiv preprint arXiv:2406.09098 , 2024.
- Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. Minillm: Knowledge distillation of large language models. In International Conference on Learning Representations , volume 2024, pages 32694-32717, 2024.
- Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam Fowl, Abhishek Panigrahi, Danqi Chen, and Sanjeev Arora. Self-distillation zero: Self-revision turns binary rewards into dense supervision. arXiv preprint arXiv:2604.12002 , 2026.
- Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802 , 2026.

- Ijun Jang, Jewon Yeom, Juan Yeo, Hyunggu Lim, and Taesup Kim. Stable on-policy distillation through adaptive target reformulation. arXiv preprint arXiv:2601.07155 , 2026.
- Yiqiao Jin, Yiyang Wang, Lucheng Fu, Yijia Xiao, Yinyi Luo, Haoxin Liu, B. Aditya Prakash, Josiah Hester, Jindong Wang, and Srijan Kumar. Unisd: Towards a unified self-distillation framework for large language models. arXiv preprint arXiv:2605.06597 , 2026.
- Junlong Ke, Zichen Wen, Weijia Li, Conghui He, and Linfeng Zhang. Respecting self-uncertainty in on-policy self-distillation for efficient llm reasoning. arXiv preprint arXiv:2605.13255 , 2026.
- Jeonghye Kim, Xufang Luo, Minbeom Kim, Sangmook Lee, Dohyung Kim, Jiwon Jeon, Dongsheng Li, and Yuqing Yang. Why does self-distillation (sometimes) degrade the reasoning capability of llms? arXiv preprint arXiv:2603.24472 , 2026.
- Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation. arXiv preprint arXiv:2603.11137 , 2026.
- Gengsheng Li, Tianyu Yang, Junfeng Fang, Mingyang Song, Mao Zheng, Haiyun Guo, Dan Zhang, Jinqiao Wang, and Tat-Seng Chua. Unifying group-relative and self-distillation policy optimization via sample routing. arXiv preprint arXiv:2604.02288 , 2026.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025.

Qwen Team. Qwen3 technical report. arXiv preprint arXiv:2505.09388 , 2025.

- David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&amp;a benchmark. arXiv preprint arXiv:2311.12022 , 2023.
- Stéphane Ross, Geoffrey Gordon, and Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics , 2011.
- Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. Crisp: Compressed reasoning via iterative self-policy distillation. arXiv preprint arXiv:2603.05433 , 2026.
- Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 , 2024.
- Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897 , 2026.
- Donghyeok Shin, Yeongmin Kim, Suhyeon Jo, Byeonghu Na, and Il-Chul Moon. Amid: Knowledge distillation for llms with α -mixture assistant distribution. In The Fourteenth International Conference on Learning Representations , 2026.
- Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626 , 2026.
- Alex Stein, Furong Huang, and Tom Goldstein. Gates: Self-distillation under privileged context with consensus gating. arXiv preprint arXiv:2602.20574 , 2026.

- Zhiquan Tan and Yinrong Hong. Paint: Partial-solution adaptive interpolated training for self-distilled reasoners. arXiv preprint arXiv:2604.26573 , 2026.
- Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, and Zhipeng Wang. Paced: Distillation and on-policy self-distillation at the frontier of student competence. arXiv preprint arXiv:2603.11178 , 2026.
- Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled rlvr. arXiv preprint arXiv:2604.03128 , 2026.

Kai Yang, Xin Xu, Yangkun Chen, Weijie Liu, Jiafei Lyu, Zichuan Lin, Deheng Ye, and Saiyong Yang. Entropic: Towards stable long-term training of llms via entropy stabilization with proportional-integral control. arXiv preprint arXiv:2511.15248 , 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems , 38:113222-113244, 2026a.

Xin Yu, Liuchen Liao, Yiwen Zhang, Yingchen Yu, Lingzhou Xue, and Qinzhen Guo. Preference-based self-distillation: Beyond kl matching via reward regularization. arXiv preprint arXiv:2605.05040 , 2026b.

Hongbin Zhang, Chaozheng Wang, Kehai Chen, Youcheng Pan, Yang Xiang, Jinpeng Wang, and Min Zhang. Tailoring teaching to aptitude: Direction-adaptive self-distillation for llm reasoning. arXiv preprint arXiv:2605.22263 , 2026.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734 , 2026.

## A. Detailed Proofs

## A.1. Complete Proof of Theorem 1

Proof. We fix a token position t and suppress the expectations over x and ˆ y for clarity . Recall the notation: π t θ ( v ) = πθ ( v | x , ˆ y &lt; t ) , π t T ( v , y ∗ ) = π ¯ θ ( v | x , y ∗ , ˆ y &lt; t ) , π t S ( v ) = π ¯ θ ( v | x , ˆ y &lt; t ) , and ∆ t ( v ) = log π t T ( v , y ∗ ) -log π t S ( v ) .

Following Yang et al. (2026), the leakage at position t is driven by the ∇ θ E π t θ [ ∆ t ] term, which is the component that carries y ∗ -dependent information and pushes the student to encode privileged correlations. In standard OPSD, this term enters with coefficient 1 , yielding a per-position leakage contribution proportional to ∥ ∆ t ∥ 2 . In DemoPSD, the same term enters with coefficient ( 1 -α t ) , yielding per-position contribution ( 1 -α t ) 2 ∥ ∆ t ∥ 2 .

Since α t = ( σ ( β · dt ) -0.5 ) · 2 · α max &gt; 0 whenever dt &gt; 0 , we have ( 1 -α t ) &lt; 1 on all positions with nonzero disagreement. Taking expectations over positions:

<!-- formula-not-decoded -->

where the strict inequality uses ( 1 -α t ) 2 &lt; 1 on { dt &gt; 0 } , which has positive probability by assumption. Since α t is monotonically increasing in dt and dt = JSD ( π t S ∥ π t T ) correlates with ∥ ∆ t ∥ , the attenuation factor ( 1 -α t ) 2 is smallest at positions with the largest ∥ ∆ t ∥ , concentrating the suppression where it is most needed.

## A.2. Complete Proof of Theorem 2

Proof. We fix a token position t , using the same notation as in Appendix A.1.

Step 1: Exponential family structure. Write q t γ ( v ) = π t S ( v ) e γ ∆ t ( v ) / Z γ where Z γ = ∑︀ v π t S ( v ) e γ ∆ t ( v ) is the partition function and ∆ t ( v ) = log π t T ( v , y ∗ ) -log π t S ( v ) . This is a one-parameter exponential family with parameter γ , sufficient statistic ∆ t ( v ) , and base measure π t S .

At the boundary values: q 0 ( v ) = π t S ( v ) and q 1 ( v ) = π t S ( v ) e ∆ t ( v ) / Z 1 = π t T ( v , y ∗ ) . The DemoPSD target corresponds to γ = 1 -α t , so q 1 -α t = π α t target .

Standard exponential family identities give:

<!-- formula-not-decoded -->

A key property we will use: for any function f : V → R ,

<!-- formula-not-decoded -->

Step 2: Entropy derivative. Since log q t γ ( v ) = log π t S ( v ) + γ ∆ t ( v ) -log Z γ , the entropy is:

<!-- formula-not-decoded -->

Differentiating each term with respect to γ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(23) uses (21) with f = log π t S and (24) uses the product rule and (21) with f = ∆ t . Summing (23)-(25), the E q t γ [ ∆ t ] terms cancel such that

<!-- formula-not-decoded -->

Step 3: Monotonicity under the covariance condition. Under condition (16), both terms in (26) are non-positive for γ &gt; 0 :

̸

- -γ Var q t γ [ ∆ t ] ≤ 0 , with strict inequality when γ &gt; 0 and ∆ t is nonconstant (i.e., π t T = π t S );

̸

- -Cov q t γ ( ∆ t , log π t S ) ≤ 0 by the condition.

Hence d H ( q t γ ) d γ ≤ 0 for all γ ∈ [ 0, 1 ] , with strict inequality on ( 0, 1 ] when π t T = π t S .

̸

Step 4: Entropy ordering. Since H ( q t γ ) is strictly decreasing on [ 0, 1 ] when π t T = π t S :

<!-- formula-not-decoded -->

̸

where the strict inequalities require 0 &lt; α t &lt; 1 (so that 0 &lt; 1 -α t &lt; 1 , placing the DemoPSD target strictly between the two endpoints) and π t T = π t S .

## B. Implementation Details

Topk Distillation. Weextract topk = 100 tokens from the student's logits, compute both teacher probabilities on this same subset, and aggregate remaining mass into a tail bucket. This reduces memory from O ( |V| ) to O ( k ) per position. The student's topk indices are shared with both teacher forwards, ensuring all three distributions are index-aligned.

Probability Floor. All teacher log-probabilities are clamped: log p ( v ) ← max ( log p ( v ) , log 10 -8 ) to prevent numerical issues in the geometric mixture computation.

Importance Sampling Clip. To stabilize training across PPO minibatches, we clip the importance sampling ratio: ρ = min ( exp ( log πθ ( yt ) -log πθ old ( yt )) , 2.0 ) .

EMA Schedule. The unprivileged reference uses EMA rate η = 0.05 , updated once after all minibatches complete within a training step.

Masking. Only response tokens are included in the loss ( T excludes prompt tokens). Samples without a valid reprompt (demopsd\_mask = 0) have their loss contribution zeroed.

Privileged Context Truncation. When the privileged prompt (question + correct solution + student response) exceeds the maximum reprompt length (10,240 tokens), the demonstration is truncated from the right, preserving the system/user prefix. This is a deliberate departure from SDPO's error-on-overflow behavior, ensuring training proceeds even with long demonstrations.