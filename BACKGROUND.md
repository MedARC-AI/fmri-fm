### Motivation

The human brain is the most complex system we know of. Accurately diagnosing and treating its range of dysfunctions is largely beyond the limit of modern medicine. Functional neuroimaging data such as fMRI and EEG provide a possible window into better understanding how the brain works, why it sometimes breaks down, and what to do about it. After decades of effort, however, it feels as though we are still far from applying these data to real clinical purpose. One challenge is that the data are a mess. They are extremely noisy, blurry, and difficult to visually interpret. But on the positive side, we have collected a *lot* of it. In many other data domains, we have seen that by combining large amounts of data together with flexible neural network models and large-scale compute, one can "unlock" the signal in complex noisy data. Our goal is to answer a simple question: can we do the same with functional neuroimaging?

### Related work

Building clinically predictive models using functional neuroimaging data has of course been a longstanding goal[^1] [^2]. The current dominant approach combines expert-crafted features, for example resting-state functional connectivity matrices, together with classic shallow machine learning models. See for example[^3], and many other references cited within and since. Unlike other domains like images and text, brain data do not seem to immediately yield to deep learning based approaches[^4] [^5]. Although there are some success cases of shallow nonlinear networks applied to functional brain imaging data, e.g.[^6] [^7] [^8].

One challenge to training high capacity deep learning models for functional neuroimaging is that the amount of *labeled* data available for any particular prediction task is limited. Outside of neuroimaging, this challenge has largely been addressed through the development of self-supervised learning (SSL) methods, where models are first trained to solve some pretext task on unlabeled data. SSL is a natural fit for functional neuroimaging, thanks to the large amounts of publicly available data without clinically specific labels (e.g. HCP [^9], UKBB[^10], ABCD[^11], OpenNeuro[^12]).


In the context of computer vision, some example SSL methods include:

- contrastive learning methods trained to learn invariant representations across multiple image views (e.g. SimCLR[^9], MoCo[^10])
- self-distillation methods which remove the need for negative examples (e.g. BYOL[^11], DINO[^12], SimSiam[^13])
- masked image modeling (MIM) methods which make predictions for unobserved parts of the image rather than unobserved views (e.g. iBOT[^14], MAE[^15], BEIT[^16]).
- methods which combine elements across these groups (e.g. DINOv2[^17], CAPI[^18])

### Approach


### References

[^1]: Gabrieli JD, Ghosh SS, Whitfield-Gabrieli S. [Prediction as a humanitarian and pragmatic contribution from human cognitive neuroscience](https://doi.org/10.1016/j.neuron.2014.10.047). Neuron. 2015.

[^2]: Woo CW, Chang LJ, Lindquist MA, Wager TD. [Building better biomarkers: brain models in translational neuroimaging](https://doi.org/10.1038/nn.4478). Nature neuroscience. 2017.

[^3]: He T, An L, Chen P, Chen J, Feng J, Bzdok D, Holmes AJ, Eickhoff SB, Yeo BT. [Meta-matching as a simple framework to translate phenotypic predictive models from big to small data](https://doi.org/10.1038/s41593-022-01059-9). Nature neuroscience. 2022.

[^4]: He T, Kong R, Holmes AJ, Nguyen M, Sabuncu MR, Eickhoff SB, Bzdok D, Feng J, Yeo BT. [Deep neural networks and kernel regression achieve comparable accuracies for functional connectivity prediction of behavior and demographics](https://doi.org/10.1016/j.neuroimage.2019.116276). NeuroImage. 2020.

[^5]: Schulz MA, Yeo BT, Vogelstein JT, Mourao-Miranada J, Kather JN, Kording K, Richards B, Bzdok D. [Different scaling of linear models and deep learning in UKBiobank brain images versus machine-learning datasets](https://doi.org/10.1038/s41467-020-18037-z). Nature communications. 2020.

[^6]: Heinsfeld AS, Franco AR, Craddock RC, Buchweitz A, Meneguzzi F. [Identification of autism spectrum disorder using deep learning and the ABIDE dataset](https://doi.org/10.1016/j.nicl.2017.08.017). NeuroImage: clinical. 2018.

[^7]: Peng H, Gong W, Beckmann CF, Vedaldi A, Smith SM. [Accurate brain age prediction with lightweight deep neural networks](https://doi.org/10.1016/j.media.2020.101871). Medical image analysis. 2021.

[^8]: Popov P, Mahmood U, Fu Z, Yang C, Calhoun V, Plis S. [A simple but tough-to-beat baseline for fMRI time-series classification](https://doi.org/10.1016/j.neuroimage.2024.120909). NeuroImage. 2024 Dec 1;303:120909.

[^9]: Van Essen DC, et al. [The WU-Minn human connectome project: an overview. Neuroimage](https://www.humanconnectome.org/). 2013.

[^10]: Bycroft C, et al. [The UK Biobank resource with deep phenotyping and genomic data](https://www.ukbiobank.ac.uk/). Nature. 2018.

[^11]: Casey BJ, et al. [The adolescent brain cognitive development (ABCD) study: imaging acquisition across 21 sites](https://abcdstudy.org/). Developmental cognitive neuroscience. 2018.

[^12]: Markiewicz CJ, et al. [The OpenNeuro resource for sharing of neuroscience data](https://doi.org/10.7554/eLife.71774). Elife. 2021.

[^9]: Chen T, Kornblith S, Norouzi M, Hinton G. [A simple framework for contrastive learning of visual representations](https://proceedings.mlr.press/v119/chen20j.html). International conference on machine learning. 2020.

[^10]: He K, Fan H, Wu Y, Xie S, Girshick R. [Momentum contrast for unsupervised visual representation learning](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html). CVPR. 2020.

[^11]: Grill JB, Strub F, Altché F, Tallec C, Richemond P, Buchatskaya E, Doersch C, Avila Pires B, Guo Z, Gheshlaghi Azar M, Piot B. [Bootstrap your own latent-a new approach to self-supervised learning](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html). NeurIPS. 2020.

[^12]: Chen X, He K. [Exploring simple siamese representation learning](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html). CVPR. 2021

[^13]: Caron M, Touvron H, Misra I, Jégou H, Mairal J, Bojanowski P, Joulin A. [Emerging properties in self-supervised vision transformers](https://arxiv.org/abs/2104.14294). ICCV. 2021.

[^14]: Zhou J, Wei C, Wang H, Shen W, Xie C, Yuille A, Kong T. [ibot: Image bert pre-training with online tokenizer](https://arxiv.org/abs/2111.07832). ICLR. 2022.

[^15]: He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. [Masked autoencoders are scalable vision learners](https://arxiv.org/abs/2111.06377). CVPR. 2022.

[^16]: Bao H, Dong L, Piao S, Wei F. [Beit: Bert pre-training of image transformers](https://openreview.net/forum?id=p-BhZSz59o4). ICLR. 2022.

[^17]: Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, Fernandez P, Haziza D, Massa F, El-Nouby A, Assran M. [Dinov2: Learning robust visual features without supervision](https://openreview.net/forum?id=a68SUt6zFt). TMLR. 2024.

[^18]: Darcet T, Baldassarre F, Oquab M, Mairal J, Bojanowski P. [Cluster and predict latent patches for improved masked image modeling](https://arxiv.org/abs/2502.08769). TMLR. 2025.
