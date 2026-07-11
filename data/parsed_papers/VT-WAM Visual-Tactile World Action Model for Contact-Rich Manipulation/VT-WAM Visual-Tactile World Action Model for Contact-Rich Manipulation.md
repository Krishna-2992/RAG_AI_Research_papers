## VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation

Shuai Tian 1 , 2 , Yupeng Zheng 1 , 2 , 3 ∗ , Yuhang Zheng 4 , Songen Gu 5 , Yujie Zang 3 , 4 ,

Yuxing Qin 1 , 2 , Weize Li 3 , Haoran Li 1 , 2 , † , Wenchao Ding 3 , † , Dongbin Zhao 1 , 2

1 SKL-MAIS, Institute of Automation, Chinese Academy of Sciences

2 School of Artificial Intelligence, University of Chinese Academy of Sciences

3 TARS Robotics 4 National University of Singapore 5 Fudan University ∗ †

Project Leader Corresponding Author

Abstract -Contact-rich manipulation requires policies to react to local deformation, pressure, slip, and friction, yet these cues are temporally sparse and often invisible in visual observations. Existing visual-tactile policies usually feed tactile observations directly into action prediction, but rarely model tactile deformation dynamics during action generation. In this paper, we introduce VT-WAM, a Visual-Tactile World Action Model that jointly learns future visual prediction, tactile deformation prediction, and action prediction within a unified flow matching framework. In particular, VT-WAM introduces (1) Asymmetric Mixture-of-Transformers (MoT) Attention to bridge a first-frame visual anchor with temporal tactile dynamics, and (2) contact-gated Action-Visual-Tactile Attention Guidance (AVTAG) to encourage action queries to rely on tactile evidence during contact phases. Across six real-world contact-rich manipulation tasks, VT-WAM achieves a 71.67% average success rate, outperforming Fast-WAM by 26.67% and OmniVTLA by 35.84%. Ablations demonstrate that modeling tactile deformation dynamics and guiding contact-phase tactile attention are both important for contact-rich tasks. Project Website: https://vt-wam.github.io/.

## I. INTRODUCTION

Contact-rich manipulation poses a core and persistent challenge in robotic manipulation, essential for practical deployment. Unlike free-space manipulation, these tasks rely on local interaction states, including deformation, pressure, slip, and friction. These states are often weakly visible, transient, or occluded in visual observations, making visioncentric policies [1], [2] unreliable when execution demands tactile-informed adjustments.

Recent visual-tactile policies [3], [4], [5] have introduced tactile sensing into action prediction and have made progress on contact-rich tasks. However, these policies often fail to exploit tactile information fully [6]. The reason is that contact-rich manipulation primarily depends on local contact evolution rather than global scene variations. Specifically, tactile deformation evolves and provides force feedback only during brief contact phases, as shown in Fig. 1(a). In contrast, visual observations provide dense scene-level information in most frames. This temporal imbalance in information availability causes neural networks to favor visual evidence during joint training, while tactile signals remain underutilized.

To address this issue, our key insight is to couple action prediction with tactile evolution, enabling the policy to leverage tactile changes during contact phases. Recently, World Action Models (WAMs) have provided the ability to predict world dynamics by coupling action prediction and video prediction [7], [8], [9]. Building on world action models, we propose VT-WAM, a visual-tactile world action model that jointly learns future visual prediction, tactile deformation prediction, and action prediction within a unified flow matching framework.

Fig. 1. Sparse tactile dynamics provide decisive evidence for contactrich manipulation. (a) Across six real-world tasks, tactile responses appear mainly around short contact events, making the informative signal temporally sparse. (b) By coupling action prediction with tactile deformation dynamics, VT-WAM improves the average success rate from 45.00% with Fast-WAM to 71.67%, with consistent gains on both surface-interaction and constrained insertion tasks.

<!-- image -->

In particular, VT-WAM has two core modules that make tactile dynamics useful for action prediction. Asymmetric MoT Attention routes action tokens to a first-frame visual anchor for scene context and to the full tactile sequence for contact evolution. This enables visual-cache inference mode without discarding tactile dynamics needed for contact phases. Contact-gated AVTAG further reduces visualdominance bias by applying a training-only hinge ranking loss that encourages action queries to attend to tactile evidence during contact phases. This auxiliary guidance makes the model rely more on tactile dynamics when contact information is physically informative, without changing the inference-time architecture.

We evaluate VT-WAM on six real-world contact-rich tasks, covering surface-interaction and constrained insertion regimes. As shown in Fig. 1(b), VT-WAM achieves a 71.67% success rate and outperforms the baseline Fast-WAM [9] by 26.67%. Detailed ablation studies of tactile dynamics modeling methods and attention guidance demonstrate the effectiveness of our core designs in contact-rich tasks.

Our main contributions are summarized as follows:

- We formulate VT-WAM to couple tactile deformation dynamics with action prediction through joint visualtactile-action flow matching.
- We introduce Asymmetric MoT Attention and AVTAG to enable a visual anchor, temporal tactile dynamics, and contact-phase tactile guidance.
- We validate VT-WAM on six real-world tasks, reaching 71.67% average success, 26.67% above Fast-WAM; ablations confirm both designs.

## II. RELATED WORK

## A. Tactile Policy for Contact-Rich Manipulation

Tactile sensing is useful for contact-rich manipulation because it provides local interaction information. Existing tactile policies commonly use tactile signals in three ways. One line of work augments diffusion policies [1] with tactile conditioning, including FARM [10], TacDiffusion [11], PolyTouch [12], and KineDex [13]. Another line develops reactive or dual-system policies that use tactile or force feedback for online contact correction, including RDP [3], Force Policy [14], and M2-ResPolicy [15]. Recent VisionLanguage-Action (VLA) models further incorporate tactile feedback into large-scale policy architectures such as BiTLA [16], OmniVTLA [5], VTLA [17], Tactile-VLA [18], TaF-VLA [19], and VLA-Touch [4]. Beyond using tactile observations as policy inputs, recent visual-tactile world models explicitly predict contact evolution: VT-WM jointly models visual and tactile observations in a latent recurrent state space and uses the learned dynamics for planning [20], while OmniVTA predicts visual-tactile evolution and incorporates the predicted tactile into adaptive policy fusion and reflex control [21]. These methods show that tactile deformations are valuable for contact-rich tasks, but tactile prediction is still used indirectly through planning or downstream action modules. In contrast, our VT-WAM couples tactile prediction with action prediction in a single flow matching objective with Asymmetric MoT Attention and contact-gated guidance, enabling tactile dynamics-aware action prediction.

## B. World Action Models for Manipulation

Compared with Vision-Language-Action (VLA) models that directly predict actions from current observations and language instructions, World Action Models [22] incorporate future-state prediction into action prediction. Existing WAMs are commonly organized into cascaded and joint architectures according to how future-state prediction is coupled with action prediction. Cascaded WAMs first synthesize future visual states or intermediate plans, and then derive executable actions from the predicted futures, with representative methods including UniPi [23], VLP [24], RoboEnvision [25], and Dream4manip [26]. Joint WAMs instead learn future dynamics and action prediction within a shared generative objective, as in Fast-WAM [9], DreamZero [27], Motus [28], LingBot-VA [8], GigaWorld-Policy [29], and UWM [7]. However, existing WAMs mainly model visual dynamics for action prediction. VT-WAM extends WAMs to tactile deformation dynamics, allowing contact evolution to directly inform action prediction.

## III. METHOD

VT-WAM is a visual-tactile World Action Model for contact-rich manipulation. Given wrist camera observations O v , tactile deformation observations O t , proprioceptive state s , language instruction c , and action chunk A , VT-WAM jointly learns future visual prediction, tactile deformation prediction, and action prediction within a unified flow matching framework. Fig. 2 illustrates the overall VT-WAM architecture, including the visual-tactile-action expert backbone, Asymmetric MoT Attention, contact-gated AVTAG, and the training and inference procedures.

## A. The Architecture of VT-WAM

As shown in Fig. 2(a), VT-WAM uses a visual-tactileaction expert architecture. The visual expert encodes wrist camera tokens as global scene context, the tactile expert models local contact evolution from tactile deformation tokens, and the action expert predicts the action chunk from visual and tactile evidence. Asymmetric MoT Attention connects the three experts, enabling joint visual, tactile, and action prediction within one backbone.

VT-WAM first maps each modality into a token sequence. The wrist camera sequence O v ∈ R T v × 3 × H × W is encoded by the Wan2.2 video VAE [30] and patchified into visual tokens X v ∈ R N v × d . The tactile deformation sequence O t ∈ R T t × 6 × H t × W t contains 3D deformation fields from two tactile surfaces. Following OmniVTA [21], a pretrained tactile VAE encodes this sequence into tactile tokens X t ∈ R N t × d . Each action chunk A ∈ R S a × D a is linearly projected into action tokens X a ∈ R S a × d . The language instruction and proprioceptive state are provided to each expert through cross-attention [31].

At the l -th Asymmetric MoT Attention layer, each expert computes query, key, and value tensors from its own token stream. These tensors are then concatenated in the order of visual, tactile, and action tokens:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Asymmetric MoT Attention performs masked attention over this concatenated token sequence:

Fig. 2. Overview of VT-WAM. (a) Joint visual-tactile-action flow matching with three modality-specific experts connected by Asymmetric MoT Attention. (b) Attention masks in Asymmetric MoT Attention during training and inference. (c) Contact-gated AVTAG applies a training-only hinge ranking loss that encourages action queries to prioritize tactile evidence during contact phases.

<!-- image -->

<!-- formula-not-decoded -->

Here P ( l ) denotes the attention map over visual, tactile, and action tokens, and M determines which query tokens can attend to which key tokens. The resulting output Y ( l ) gives the updated token features for the visual, tactile, and action experts. After the Asymmetric MoT Attention layers, modality-specific projection heads predict velocity fields for the visual, tactile, and action tokens under the flow matching objective. The visual and tactile experts supervise future visual prediction and tactile deformation prediction during training, while the action head produces the action chunk for control.

## B. Asymmetric MoT Attention

Asymmetric MoT Attention controls how visual, tactile, and action tokens exchange information in MoT [32] layers. The design is motivated by two requirements of contactrich control. First, wrist camera observations mainly provide global scene context, while tactile deformation provides the key evidence for contact interaction. Therefore, action prediction should access the tactile sequence to capture contact evolution. Second, while tactile dynamics are essential for contact-rich control, denoising future visual tokens introduces unnecessary latency during deployment. VT-WAM therefore uses an asymmetric readout: action tokens attend to the tactile sequence for contact dynamics, but attend only to the first-frame visual tokens for global context.

Fig. 2(b) illustrates how this readout is implemented in training and inference. During training, VT-WAM keeps the visual, tactile, and action branches in one joint flow matching model, so future visual, tactile, and action predictions are optimized together. During inference, future visual tokens are removed. The tactile and action branches use the first-frame visual anchor, and action tokens attend to the tactile latent sequence being denoised. This preserves contact-dynamics modeling while avoiding the cost of future visual prediction.

We formalize the cross-modal mask used by the asymmetric readout below. The visual tokens are divided into the firstframe visual anchor and future visual tokens. Let F v denote the number of first-frame visual tokens, N v the number of all visual tokens, N t the number of tactile tokens, and S a the number of action tokens. VT-WAM packs these tokens in the order [ X v ; X t ; X a ] and applies a blockwise attention mask M , where rows correspond to query tokens being updated and columns correspond to key tokens available as information sources. A zero entry allows attention, while -∞ blocks attention. Same-modality attention is retained within each expert, and the following rules specify the crossmodal information flow.

For the visual expert, tactile and action keys are masked out so that local contact deformation and future action tokens do not modify the visual representation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For the tactile expert, only the first-frame visual anchor is visible among visual tokens. This grounds tactile dynamics in the global scene context while preventing dependence on future visual tokens. Action keys are also masked out in this readout mask:

<!-- formula-not-decoded -->

For the action expert, the mask exposes the first-frame visual anchor and the full tactile sequence, matching the visual-cache inference mode used for control:

<!-- formula-not-decoded -->

Thus, Asymmetric MoT Attention keeps visual representations stable, grounds tactile dynamics in visual context, and provides action prediction with both visual anchor and contact-evolution information.

## C. Contact-Gated Action-Visual-Tactile Attention Guidance

Although Asymmetric MoT Attention allows action tokens to attend to tactile tokens, joint training can still favor visual evidence over tactile evidence. This is because visual and tactile signals are imbalanced in contact-rich tasks. Visual observations provide dense scene-level information across most frames. In contrast, tactile deformation is local and temporally sparse: it becomes informative mainly during short contact intervals and remains weak or inactive outside contact [21]. As a result, the joint flow matching objective can reduce training loss by relying primarily on visual context, while underusing tactile dynamics that are critical during contact phases.

To mitigate this imbalance, VT-WAM introduces ActionVisual-Tactile Attention Guidance (AVTAG), illustrated in Fig. 2(c). AVTAG adds a training-only auxiliary attention objective that computes the relative attention from action queries to visual and tactile evidence. During contact phases, it applies a contact-gated hinge ranking loss that penalizes cases where relative tactile attention is lower than relative visual attention. This guides action queries to increase tactile attention when local physical interaction is informative.

AVTAG constructs an auxiliary attention distribution from action queries to visual and tactile keys. For clarity, we omit the layer index and use Q a , K v , and K t to denote the action queries and visual-tactile keys from MoT Attention layers. Let K vt = [ K v ; K t ] denote the concatenated visual and tactile keys. To guide action queries without directly changing the visual and tactile key representations, we apply stop-gradient to K vt and define

<!-- formula-not-decoded -->

where sg( · ) denotes stop-gradient. This auxiliary attention is used only for the AVTAG loss, so its gradients guide the action queries while leaving the visual and tactile keys optimized by the main flow matching objective.

For each action token r ∈ { 1 , . . . , S a } , AVTAG sums the auxiliary attention assigned to visual keys and tactile keys:

<!-- formula-not-decoded -->

Fig. 3. Real-world experimental platform. The setup uses a 7-DoF xArm7 robot with a Robotiq 2F-85 gripper, a wrist camera, and paired gripper-mounted Xense tactile sensors. The scene includes the representative objects used in our experiments.

<!-- image -->

These two quantities are then normalized into relative visual and tactile attention weights:

<!-- formula-not-decoded -->

AVTAG applies this guidance only to action tokens in contact phases. Let C denote contact-phase action tokens, identified by pronounced tactile deformation. The auxiliary loss is defined as

<!-- formula-not-decoded -->

This hinge ranking loss penalizes visual-dominant attention during contact phases, and incurs no penalty once p t ( r ) ≥ p v ( r ) .

## D. Training Objective and Efficient Inference

1) Flow Matching Training Objective: VT-WAM is trained with a joint flow matching objective over visual, tactile, and action tokens. The visual, tactile, and action experts each predict the velocity field of their corresponding modality, yielding

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here ˆ f v , ˆ f t , and ˆ f a denote the predicted velocity fields for visual, tactile, and action tokens, and f ∗ v , f ∗ t , and f ∗ a denote the corresponding flow matching targets. When AVTAG is enabled, the full training objective is

<!-- formula-not-decoded -->

Wipe whiteboard clean.

<!-- image -->

Insert the 2-prong plug into the wall socket.

Fig. 4. Overview of real-world contact-rich manipulation tasks. We evaluate VT-WAM on six real-world tasks covering two interaction regimes: surface-interaction tasks and constrained insertion tasks.

TABLE I SUCCESS RATES ON REAL-WORLD CONTACT-RICH TASKS.

| Method            | Surface-Interaction Tasks   | Surface-Interaction Tasks   | Surface-Interaction Tasks   | Surface-Interaction Tasks   | Constrained Insertion Tasks   | Constrained Insertion Tasks   | Constrained Insertion Tasks   | Constrained Insertion Tasks   | Average   |
|-------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|-----------|
| Method            | Wipe Board                  | Wipe Vase                   | Peel Cucumber               | Avg.                        | Insert Plug                   | Swipe Card                    | Insert Tube                   | Avg.                          | Average   |
| DP + Tactile [10] | 30%                         | 20%                         | 25%                         | 25.00%                      | 5%                            | 35%                           | 15%                           | 18.33%                        | 21.67%    |
| RDP [3]           | 45%                         | 60%                         | 40%                         | 48.33%                      | 15%                           | 35%                           | 10%                           | 20.00%                        | 34.17%    |
| π 0 . 5 [2]       | 40%                         | 35%                         | 35%                         | 36.67%                      | 30%                           | 45%                           | 10%                           | 28.33%                        | 32.50%    |
| OmniVTLA [5]      | 45%                         | 30%                         | 25%                         | 33.33%                      | 40%                           | 35%                           | 40%                           | 38.33%                        | 35.83%    |
| Fast-WAM [9]      | 70%                         | 55%                         | 45%                         | 56.67%                      | 20%                           | 55%                           | 25%                           | 33.33%                        | 45.00%    |
| VT-WAM            | 90%                         | 85%                         | 70%                         | 81.67%                      | 60%                           | 70%                           | 55%                           | 61.67%                        | 71.67%    |

- 2) Efficient Visual-Cache Inference: VT-WAM supports two inference modes: joint inference mode and visual-cache inference mode. For visual-tactile prediction analysis, we use joint inference mode, where the model denoises visual, tactile, and action tokens together to evaluate its predictive ability. For real-world control, we use visual-cache inference mode, where the current visual observation is kept as a first-frame anchor and future visual prediction is removed. In this mode, VT-WAM denoises only tactile and action latents through Asymmetric MoT Attention: the tactile expert predicts future tactile deformation as contact evolution, and the action expert predicts the action chunk by attending to both the visual anchor and the tactile sequence. This avoids the cost of predicting future visual tokens during deployment.

## IV. EXPERIMENTS

In this section, we first describe the experimental setup, including the robotic platform, implementation details, baselines, benchmark tasks, and evaluation metrics. We then evaluate VT-WAM on six contact-rich manipulation tasks, analyze visual-tactile prediction quality, and conduct ablation studies to quantify the contribution of key components.

## A. Experimental Setup

- 1) Robotic Platform: To evaluate VT-WAM on real-world contact-rich manipulation tasks, we use the physical robotic

platform shown in Fig. 3. The platform consists of a 7-DoF xArm7 robot equipped with a Robotiq 2F-85 parallel gripper, a wrist camera, and two Xense tactile sensors mounted on the inner surfaces of the gripper fingers. The wrist camera captures 128 × 128 RGB observations at 30 Hz. Each tactile sensor records a 35 × 20 three-dimensional deformation field over the contact surface at 30 Hz.

- 2) Implementation Details: Training data are collected through human kinesthetic teaching, with 100 expert trajectories for each task. The visual, tactile, proprioceptive, and action streams are synchronized and resampled to 30 Hz before training. VT-WAM uses pretrained Wan2.2-5B [30] as the visual backbone and uses 1B-scale DiT models for the tactile and action experts. The loss weights are set to λ v = λ t = λ a = 1 and λ AVTAG = 0 . 05 in all experiments. We optimize models with AdamW using a learning rate of 1 × 10 -4 , weight decay of 1 × 10 -2 , bf16 mixed precision, gradient clipping at 1.0, and cosine learning-rate decay after a 5% warmup. Training is conducted on NVIDIA A100 (80GB) GPUs. During inference evaluation, VT-WAM runs on a remote NVIDIA A100 inference server and uses 10 denoising steps for action prediction.
- 3) Baselines: We compare VT-WAM with representative baselines that cover visuomotor, VLA, and WAM policies:
- DP + Tactile [10]: a tactile-conditioned diffusion policy that predicts action chunks from robot states, wrist camera, and tactile observations.
- RDP [3]: a reactive visual-tactile policy that uses tactile feedback for online action refinement.
- π 0 . 5 [2]: a general vision-language-action policy without tactile input.
- OmniVTLA [5]: a tactile-augmented VLA model that uses tactile observations for action prediction.
- Fast-WAM [9]: a world action model that models visual dynamics and predicts actions without tactile input.

Fig. 5. Visual-Tactile Prediction Results across Six Tasks. For visualization, VT-WAM predicts wrist camera observations together with tactile deformation fields. Blue denotes ground truth, and orange indicates prediction.

<!-- image -->

All methods are trained separately on each task with the same demonstrations and evaluated on the same robot platform, task definitions, and metrics.

## B. Benchmark Tasks and Evaluation Metrics

1) Benchmark Tasks: As shown in Fig. 4, we evaluate VTWAM on six contact-rich tasks [3], [21] grouped into two regimes: surface-interaction tasks and constrained insertion tasks. Surface-interaction tasks include wipe board , wipe vase , and peel cucumber , which require sustained motion over planar, curved, or deformable surfaces. Constrained insertion tasks include insert plug , swipe card , and insert tube , which require fine alignment under tight geometric constraints or visual occlusion.

2) Evaluation Metrics: For each method and each task, we conduct 20 independent trials and report the success rate. For surface-interaction tasks, the score is defined in { 0 , 0 . 5 , 1 } : 0 indicates failure, 0.5 indicates completing more than half of the target region, and 1 indicates completing the full target region. For constrained insertion tasks, the score is binary in { 0 , 1 } , where 1 indicates that the object reaches the target position and 0 otherwise.

## C. Main Results

1) Task Performance: Table I reports performance across the six contact-rich tasks. VT-WAM achieves the highest success rate among all evaluated methods. Compared with the strongest baseline Fast-WAM [9], VT-WAM improves the success rate from 45.00% to 71.67%, corresponding to an absolute gain of 26.67%.

In surface-interaction tasks, the wrist camera often changes only subtly during execution, while contact changes occur at the local interaction surface. Therefore, vision-based policies remain limited on these tasks: π 0 . 5 achieves 36.67% success rate, while OmniVTLA achieves only 33.33% despite using tactile input. This comparison suggests that using tactile observations only as policy inputs is insufficient to model tactile interaction dynamics, which may explain why it does not improve the success rate. Fast-WAM improves the success rate to 56.67% by modeling action-conditioned visual dynamics, highlighting the benefit of action-conditioned dynamics modeling. However, Fast-WAM remains visiononly and cannot directly capture local tactile interaction. VT-WAM improves the success rate to 81.67% by modeling tactile deformation as interaction dynamics.

In constrained insertion tasks, success depends on fine alignment rather than sustained surface coverage. Across insert plug, swipe card, and insert tube, VT-WAM achieves 61.67% success rate, compared with 38.33% for OmniVTLA and 33.33% for Fast-WAM. These results suggest that tactile dynamics are also useful when the robot must correct small pose errors under tight geometric constraints. The improvement is especially clear on the insert tube, where the transparent tube makes visual alignment unreliable and successful execution requires contact-informed correction. Together with the surface-interaction results, this shows that coupling tactile deformation dynamics with action prediction improves success rates on contact-rich tasks.

- 2) Visual-Tactile Prediction Results: We analyze visualtactile prediction results to evaluate the predictive modeling ability of VT-WAM. For this analysis, we run VT-WAM in joint inference mode to predict wrist camera observations and tactile deformation fields together; real-world control instead uses the visual-cache inference mode described in Sec. III. Fig. 5 shows that VT-WAM predicts temporally coherent wrist camera observations and tactile deformation trajectories that capture local contact patterns, including pressure concentration and contact migration. Following OmniVTA [21], we quantify tactile prediction quality using deformation magnitude error and directional consistency. The l 2 distance is computed over the full 3D deformation field,

TABLE II

## TACTILE DEFORMATION PREDICTION QUALITY.

| Method     |   l 2 ↓ |   cos ↑ |
|------------|---------|---------|
| exUMI [33] |   0.091 |   0.618 |
| UVA [34]   |   0.083 |   0.667 |
| VT-WAM     |   0.077 |   0.749 |

TABLE III ABLATION STUDY ON TACTILE DYNAMICS MODELING AND ATTENTION GUIDANCE.

| Models   | Description           | Wipe Vase   | Insert Tube   |
|----------|-----------------------|-------------|---------------|
| M 0      | Fast-WAM [9]          | 55%         | 25%           |
| M 1      | M 0 + Sym. ( T Seq.)  | 65%         | 40%           |
| M 2      | M 0 + Asym. ( T 0 )   | 40%         | 30%           |
| M 3      | M 0 + Asym. ( T Seq.) | 70%         | 50%           |
| M 4      | VT-WAM: M 3 + AVTAG   | 85%         | 55%           |

Notes: All variants are built from Fast-WAM by adding different tactile modeling or attention designs. + Sym. ( T Seq.) adds tactile sequence prediction with symmetric MoT attention. + Asym. ( T 0 ) uses Asymmetric MoT Attention but restricts action queries to the first tactile frame. + Asym. ( T Seq.) allows action queries to attend to the full tactile sequence without A VTAG. VT-WAM is the full model.

while cosine similarity is computed over non-zero deformation regions. All methods are evaluated on the same task demonstrations. As reported in Table II, VT-WAM achieves lower deformation error and higher directional consistency than the baseline models, indicating that the tactile expert learns meaningful contact deformation dynamics.

## D. Ablation Studies

We conduct ablations on the wipe vase and insert tube, which represent the two benchmark regimes. Table III evaluates two design questions: how to incorporate tactile dynamics into action prediction, and whether contact-gated AVTAG improves the real-world success rate by guiding action queries to attend to tactile dynamics.

Ablation 1: How to incorporate tactile dynamics into action prediction? Compared with the visual-only FastWAM baseline, + Sym. ( T Seq.) adds tactile sequence prediction with symmetric MoT attention and improves the success rate from 55% to 65% on wipe vase and from 25% to 40% on insert tube. This result shows that introducing tactile dynamics provides additional information beyond visual dynamics. However, symmetric fusion requires future visual and tactile prediction during inference, which increases the computational cost. To avoid future visual prediction, VTWAM adopts Asymmetric MoT Attention, where action queries attend to the first-frame visual anchor and the full tactile sequence. The importance of temporal tactile information is shown by the comparison between + Asym. ( T 0 ) and + Asym. ( T Seq.). When action queries are restricted to the first tactile frame, + Asym. ( T 0 ) achieves only 40% on wipe vase and 30% on insert tube. Allowing action queries to attend to the full tactile sequence improves the success rate to 70% and 50%, respectively. This comparison indicates that

(a) VT-WAM w/o AVTAG Action Attention to Visual and Tactile Modalities

Fig. 6. AVTAG promotes tactile attention for contact recovery during vase wiping. The red and blue curves denote relative tactile and visual attention weights p t and p v from the action expert, and the dashed curve denotes the contact force | F z | for visualization only. The wrist camera view is the only visual input available to the policy, while the side view is shown only for visualization. When the supporting plane moves downward, the wrist camera view changes only subtly and provides limited evidence about the contact phase. AVTAG encourages the action expert to attend more strongly to tactile evidence during this contact phase, enabling the model to re-establish contact and complete the task.

<!-- image -->

action prediction benefits from tactile evolution over time, rather than only the initial tactile state.

Ablation 2: The effectiveness of the contact-gated attention guidance. The final comparison isolates the contribution of A VTAG. The variant + Asym. ( T Seq.) and the full VT-WAM use the same Asymmetric MoT Attention. The only difference is the contact-gated AVTAG applied during VT-WAM training. This guidance improves the success rate from 70% to 85% on wipe vase and from 50% to 55% on insert tube, indicating that the gain comes from better contactaware tactile attention learned during training. Fig. 6 further explains this effect through a vase-wiping trial with contact disturbance. During execution, the supporting plane of the vase is moved downward, which breaks contact between the wiping board and the vase surface. The wrist camera view is the only visual input available to the policy, whereas the side view is shown only to visualize the interaction process. After contact is lost, the wrist camera view changes only subtly, so the contact phase cannot be reliably identified from the wrist camera view alone. Without AVTAG, the action expert exhibits a nearly static attention pattern and remains dominated by visual tokens throughout the trial. This visual bias prevents the policy from responding to the contact loss. With AVTAG, the action expert increases tactile attention during the contact phase, enabling the policy to use tactile evidence to re-establish contact with the vase surface and complete the wiping task.

## V. CONCLUSION

In this paper, we introduce VT-WAM, a Visual-Tactile World Action Model for contact-rich manipulation. VTWAM extends the world action model by learning tactile deformation as temporal interaction dynamics together with action prediction, rather than using tactile observations only as auxiliary policy inputs. Real-world experiments across surface-interaction and constrained insertion tasks show that VT-WAM consistently improves over visual-only and tactileinput baselines, while tactile prediction analysis and ablations further support the importance of tactile dynamics modeling and contact-phase tactile use. These results indicate that modeling tactile deformation as interaction dynamics provides an effective way to improve action prediction in contact-rich tasks. While this work focuses on individualtask specialization for precise tactile modeling, multi-task training remains unexplored. Future research on multi-task training and scaling laws is a promising direction.

## REFERENCES

- [1] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, 'Diffusion Policy: Visuomotor Policy Learning via Action Diffusion,' The International Journal of Robotics Research , vol. 44, no. 10-11, pp. 1684-1704, 2025.
- [2] K. Black, N. Brown, J. Darpinian, K. Dhabalia, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai et al. , ' π 0 . 5 : A Vision-LanguageAction Model with Open-World Generalization,' arXiv preprint arXiv:2504.16054 , 2025.
- [3] H. Xue, J. Ren, W. Chen, G. Zhang, Y. Fang, G. Gu, H. Xu, and C. Lu, 'Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation,' in Proceedings of Robotics: Science and Systems (RSS) , 2025.
- [4] J. Bi, K. Y. Ma, C. Hao, M. S. Zheng, and H. Soh, 'VLA-Touch: Enhancing Vision-Language-Action Model with Dual-Level Tactile Feedback,' IEEE Robotics and Automation Letters , 2026.
- [5] Z. Cheng, Y. Zhang, W. Zhang, H. Li, K. Wang, L. Song, and H. Zhang, 'OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing,' arXiv preprint arXiv:2508.08706 , 2025.
- [6] J. Hansen, F. Hogan, D. Rivkin, D. Meger, M. Jenkin, and G. Dudek, 'VisuoTactile-RL: Learning multimodal manipulation policies with deep reinforcement learning,' in 2022 International Conference on Robotics and Automation (ICRA) . IEEE, 2022, pp. 8298-8304.
- [7] C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta, 'Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets,' in Proceedings of Robotics: Science and Systems (RSS) , 2025.
- [8] L. Li, Q. Zhang, Y. Luo, S. Yang, R. Wang, F. Han, M. Yu, Z. Gao, N. Xue, X. Zhu et al. , 'Causal World Modeling for Robot Control,' in Proceedings of Robotics: Science and Systems (RSS) , 2026.
- [9] T. Yuan, Z. Dong, Y. Liu, and H. Zhao, 'Fast-WAM: Do World Action Models Need Test-Time Future Imagination?' arXiv preprint arXiv:2603.16666 , 2026.
- [10] E. Helmut, N. Funk, T. Schneider, C. de Farias, and J. Peters, 'TactileConditioned Diffusion Policy for Force-Aware Robotic Manipulation,' arXiv preprint arXiv:2510.13324 , 2025.
- [11] Y. Wu, Z. Chen, F. Wu, L. Chen, L. Zhang et al. , 'TacDiffusion: Force-Domain Diffusion Policy for Precise Tactile Manipulation,' in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) , 2025, pp. 11 831-11 837.
- [12] J. Zhao, N. Kuppuswamy, S. Feng, B. Burchfiel, and E. Adelson, 'PolyTouch: A Robust Multi-Modal Tactile Sensor for Contact-Rich Manipulation Using Tactile-Diffusion Policies,' in Proceedings of the IEEE International Conference on Robotics and Automation (ICRA) , 2025, pp. 104-110.
- [13] D. Zhang, C. Yuan, C. Wen, H. Zhang, J. Zhao, and Y. Gao, 'KineDex: Learning Tactile-Informed Visuomotor Policies via Kinesthetic Teaching for Dexterous Manipulation,' arXiv preprint arXiv:2505.01974 , 2025.
- [14] H. Fang, S. Tang, M. Mei, H. Qin, Z. He, J. Chen, Y. Feng, C. Wang, W. Liu, Z. He et al. , 'Force Policy: Learning hybrid force-position control policy under interaction frame for contact-rich manipulation,' in Proceedings of Robotics: Science and Systems (RSS) , 2026.
- [15] X. Li, Y. Xie, H. Liu, W. Hou, G. Chen, S. Li, and W. Ding, 'Master Micro Residual Correction with Adaptive Tactile Fusion and Force-Mixed Control for Contact-Rich Manipulation,' arXiv preprint arXiv:2603.15152 , 2026.
- [16] S. Yang, H. Li, J. Hu, S. Zhang, G. Yao, Z. Ni, and B. Fang, 'BiTLA: A Bimanual Tactile-Language-Action Model for Contact-Rich Robotic Manipulation,' in Proceedings of the 1st International Workshop on Multi-Sensorial Media and Applications , 2025, pp. 12-17.
- [17] C. Zhang, P. Hao, X. Cao, X. Hao, S. Cui, and S. Wang, 'VTLA: Vision-Tactile-Language-Action Model with Preference Learning for Insertion Manipulation,' arXiv preprint arXiv:2505.09577 , 2025.
- [18] J. Huang, S. Wang, F. Lin, Y. Hu, C. Wen, and Y. Gao, 'Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization,' arXiv preprint arXiv:2507.09160 , 2025.
- [19] Y. Huang, P. Lin, W. Li, D. Li, J. Li, J. Jiang, C. Xiao, and Z. Jiao, 'Tactile-Force Alignment in Vision-Language-Action Models for Force-Aware Manipulation,' arXiv preprint arXiv:2601.20321 , 2026.
- [20] C. Higuera, S. Arnaud, B. Boots, M. Mukadam, F. R. Hogan, and F. Meier, 'Visuo-Tactile World Models,' arXiv preprint arXiv:2602.06001 , 2026.
- [21] Y. Zheng, S. Gu, W. Li, Y. Zheng, Y. Zang, S. Tian, X. Li, C. Hao, C. Gao, S. Liu et al. , 'OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation,' arXiv preprint arXiv:2603.19201 , 2026.
- [22] S. Wang, J. Shi, Z. Fu, X. He, F. Liu, C. Yang, Y. Zhou, Z. Fei, J. Gong, J. Fu et al. , 'World Action Models: The Next Frontier in Embodied AI,' arXiv preprint arXiv:2605.12090 , 2026.
- [23] Y. Du, S. Yang, B. Dai, H. Dai, O. Nachum, J. Tenenbaum, D. Schuurmans, and P. Abbeel, 'Learning Universal Policies via Text-Guided Video Generation,' Advances in Neural Information Processing Systems , vol. 36, pp. 9156-9172, 2023.
- [24] Y. Du, S. Yang, P. Florence, F. Xia, A. Wahid, P. Sermanet, T. Yu, P. Abbeel, J. B. Tenenbaum, L. Kaelbling et al. , 'Video Language Planning,' in Proceedings of the International Conference on Learning Representations (ICLR) , vol. 2024, 2024, pp. 31 138-31 155.
- [25] L. Yang, Y. Bai, G. Eskandar, F. Shen, M. Altillawi, D. Chen, S. Majumder, Z. Liu, G. Kutyniok, and A. Valada, 'RoboEnvision: A Long-Horizon Video Generation Model for Multi-Task Robot Manipulation,' in Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2025, pp. 21 281-21 288.
- [26] S. Gu, Y. Cai, T. Wang, S. Wu, and Y. Fu, 'Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation,' arXiv preprint arXiv:2602.10717 , 2026.
- [27] S. Ye, Y. Ge, K. Zheng, S. Gao, S. Yu, G. Kurian, S. Indupuru, Y. L. Tan, C. Zhu, J. Xiang et al. , 'World Action Models Are Zero-Shot Policies,' arXiv preprint arXiv:2602.15922 , 2026.
- [28] H. Bi, H. Tan, S. Xie, Z. Wang, S. Huang, H. Liu, R. Zhao, Y. Feng, C. Xiang, Y. Rong et al. , 'Motus: A Unified Latent Action World Model,' arXiv preprint arXiv:2512.13030 , 2025.
- [29] A. Ye, B. Wang, C. Ni, G. Huang, G. Zhao, H. Li, H. Li, J. Li, J. Lv, J. Liu et al. , 'GigaWorld-Policy: An Efficient Action-Centered World-Action Model,' arXiv preprint arXiv:2603.17240 , 2026.
- [30] A. Wang, B. Ai, B. Wen, C. Mao, C.-W. Xie, D. Chen, F. Yu, H. Zhao, J. Yang et al. , 'Wan: Open and Advanced Large-Scale Video Generative Models,' arXiv preprint arXiv:2503.20314 , 2025.
- [31] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, 'Attention Is All You Need,' Advances in Neural Information Processing Systems (NeurIPS) , vol. 30, 2017.
- [32] W. Liang, L. Yu, L. Luo, S. Iyer, N. Dong, C. Zhou, G. Ghosh, M. Lewis, W.-t. Yih, L. Zettlemoyer, and X. V. Lin, 'Mixture-ofTransformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models,' Transactions on Machine Learning Research , 2025.
- [33] Y. Xu, L. Wei, P. An, Q. Zhang, and Y.-L. Li, 'exUMI: Extensible Robot Teaching System with Action-Aware Task-Agnostic Tactile Representation,' in Proceedings of the Conference on Robot Learning (CoRL) , 2025.
- [34] S. Li, Y. Gao, D. Sadigh, and S. Song, 'Unified Video Action Model,' arXiv preprint arXiv:2503.00200 , 2025.