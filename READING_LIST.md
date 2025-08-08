## Reading list

### fMRI foundation models

- Thomas A, Ré C, Poldrack R. [Self-supervised learning of brain dynamics from broad neuroimaging data](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8600a9df1a087a9a66900cc8c948c3f0-Abstract-Conference.html). NeurIPS. 2022.

- [`SwiFT`] Kim P, et al. [Swift: Swin 4d fmri transformer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8313b1920ee9c78d846c5798c1ce48be-Abstract-Conference.html). NeurIPS. 2023.

- [`BrainLM`] Caro JO, et al. [BrainLM: A foundation model for brain activity recordings](https://openreview.net/forum?id=RwI7ZEfR27). ICLR. 2024.

- [`Brain-JEPA`] Dong Z, et al. Brain-jepa: Brain dynamics foundation model with gradient positioning and spatiotemporal masking. NeurIPS. 2024.

- [`NeuroSTORM`] Wang C, et al. [Towards a general-purpose foundation model for fMRI analysis](https://arxiv.org/abs/2506.11167). arXiv. 2025.

### Brain behavior prediction

Useful perspective papers on brain-behavior prediction

- Gabrieli JD, Ghosh SS, Whitfield-Gabrieli S. [Prediction as a humanitarian and pragmatic contribution from human cognitive neuroscience](https://doi.org/10.1016/j.neuron.2014.10.047). Neuron. 2015.

- Woo CW, Chang LJ, Lindquist MA, Wager TD. [Building better biomarkers: brain models in translational neuroimaging](https://doi.org/10.1038/nn.4478). Nature neuroscience. 2017.

Representative brain-behavior prediction paper with a good list of references

- He T, An L, Chen P, Chen J, Feng J, Bzdok D, Holmes AJ, Eickhoff SB, Yeo BT. [Meta-matching as a simple framework to translate phenotypic predictive models from big to small data](https://doi.org/10.1038/s41593-022-01059-9). Nature neuroscience. 2022.

Important papers showing negative result that deep learning doesn't immediately work for fMRI

- He T, Kong R, Holmes AJ, Nguyen M, Sabuncu MR, Eickhoff SB, Bzdok D, Feng J, Yeo BT. [Deep neural networks and kernel regression achieve comparable accuracies for functional connectivity prediction of behavior and demographics](https://doi.org/10.1016/j.neuroimage.2019.116276). NeuroImage. 2020.

- Schulz MA, Yeo BT, Vogelstein JT, Mourao-Miranada J, Kather JN, Kording K, Richards B, Bzdok D. [Different scaling of linear models and deep learning in UKBiobank brain images versus machine-learning datasets](https://doi.org/10.1038/s41467-020-18037-z). Nature communications. 2020.

Representative papers demonstrating good performance with shallow task-specific neural network models

- Heinsfeld AS, Franco AR, Craddock RC, Buchweitz A, Meneguzzi F. [Identification of autism spectrum disorder using deep learning and the ABIDE dataset](https://doi.org/10.1016/j.nicl.2017.08.017). NeuroImage: clinical. 2018.

- Peng H, Gong W, Beckmann CF, Vedaldi A, Smith SM. [Accurate brain age prediction with lightweight deep neural networks](https://doi.org/10.1016/j.media.2020.101871). Medical image analysis. 2021.

- Popov P, Mahmood U, Fu Z, Yang C, Calhoun V, Plis S. [A simple but tough-to-beat baseline for fMRI time-series classification](https://doi.org/10.1016/j.neuroimage.2024.120909). NeuroImage. 2024 Dec 1;303:120909.

Deep learning models for brain behavior prediction

- Kawahara J, et al. [BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment](https://doi.org/10.1016/j.neuroimage.2016.09.046). NeuroImage. 2017.

- Li X, et al. [Braingnn: Interpretable brain graph neural network for fmri analysis](https://doi.org/10.1016/j.media.2021.102233). Medical Image Analysis. 2021.

- Kan X, Dai W, Cui H, Zhang Z, Guo Y, Yang C. [Brain network transformer](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a408234a9b80604a9cf6ca518e474550-Abstract-Conference.html). NeurIPS. 2022.

### fMRI decoding and sensory reconstruction

- Beliy R, et al. [From voxels to pixels and back: Self-supervision in natural-image reconstruction from fMRI](https://proceedings.neurips.cc/paper_files/paper/2019/hash/7d2be41b1bde6ff8fe45150c37488ebb-Abstract.html). NeurIPS. 2019.

- Takagi Y, Nishimoto S. [High-resolution image reconstruction with latent diffusion models from human brain activity](https://openaccess.thecvf.com/content/CVPR2023/html/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.html). CVPR. 2023.

- Chen Z, Qing J, Xiang T, Yue WL, Zhou JH. [Seeing beyond the brain: Conditional diffusion model with sparse masked modeling for vision decoding](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Seeing_Beyond_the_Brain_Conditional_Diffusion_Model_With_Sparse_Masked_CVPR_2023_paper.html). CVPR. 2023.

- Scotti P, et al. [Reconstructing the mind's eye: fmri-to-image with contrastive learning and diffusion priors](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ddab70bf41ffe5d423840644d3357f4-Abstract-Conference.html). NeurIPS. 2023.

- Wang S, Liu S, Tan Z, Wang X. [Mindbridge: A cross-subject brain decoding framework](http://openaccess.thecvf.com/content/CVPR2024/html/Wang_MindBridge_A_Cross-Subject_Brain_Decoding_Framework_CVPR_2024_paper.html). CVPR. 2024.

- Benchetrit Y, Banville H, King JR. [Brain decoding: toward real-time reconstruction of visual perception](https://openreview.net/forum?id=3y1K6buO8c). ICLR. 2024.

- Scotti PS, et al. [Mindeye2: Shared-subject models enable fmri-to-image with 1 hour of data](https://openreview.net/forum?id=65XKBGH5PO). ICML. 2024.


### Visual self-supervised learning

Influential papers (though not the first) developing the idea of contrastive learning

- [`SimCLR`] Chen T, Kornblith S, Norouzi M, Hinton G. [A simple framework for contrastive learning of visual representations](https://proceedings.mlr.press/v119/chen20j.html). International conference on machine learning. 2020.

- [`MoCo`] He K, Fan H, Wu Y, Xie S, Girshick R. [Momentum contrast for unsupervised visual representation learning](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.html). CVPR. 2020.

Self-distillation methods that remove the need for negative examples

- [`BYOL`] Grill JB, Strub F, Altché F, Tallec C, Richemond P, Buchatskaya E, Doersch C, Avila Pires B, Guo Z, Gheshlaghi Azar M, Piot B. [Bootstrap your own latent-a new approach to self-supervised learning](https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html). NeurIPS. 2020.

- [`SimSiam`] Chen X, He K. [Exploring simple siamese representation learning](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.html). CVPR. 2021

- [`DINO`] Caron M, Touvron H, Misra I, Jégou H, Mairal J, Bojanowski P, Joulin A. [Emerging properties in self-supervised vision transformers](https://arxiv.org/abs/2104.14294). ICCV. 2021.

Representative masked image modeling methods

- [`iBOT`] Zhou J, Wei C, Wang H, Shen W, Xie C, Yuille A, Kong T. [ibot: Image bert pre-training with online tokenizer](https://arxiv.org/abs/2111.07832). ICLR. 2022.

- [`MAE`] He K, Chen X, Xie S, Li Y, Dollár P, Girshick R. [Masked autoencoders are scalable vision learners](https://arxiv.org/abs/2111.06377). CVPR. 2022.

- [`BEIT`] Bao H, Dong L, Piao S, Wei F. [Beit: Bert pre-training of image transformers](https://openreview.net/forum?id=p-BhZSz59o4). ICLR. 2022.

Methods multiple combining ideas from different methods

- [`DINOv2`] Oquab M, Darcet T, Moutakanni T, Vo H, Szafraniec M, Khalidov V, Fernandez P, Haziza D, Massa F, El-Nouby A, Assran M. [Dinov2: Learning robust visual features without supervision](https://openreview.net/forum?id=a68SUt6zFt). TMLR. 2024.

- [`CAPI`] Darcet T, Baldassarre F, Oquab M, Mairal J, Bojanowski P. [Cluster and predict latent patches for improved masked image modeling](https://arxiv.org/abs/2502.08769). TMLR. 2025.

Video self-supervised learning

- [`MAE-st`] Feichtenhofer C, Li Y, He K. [Masked autoencoders as spatiotemporal learners](https://proceedings.neurips.cc/paper_files/paper/2022/hash/e97d1081481a4017df96b51be31001d3-Abstract-Conference.html). NeurIPS. 2022.

- [`VideoMAE`] Tong Z, Song Y, Wang J, Wang L. [Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html). NeurIPS. 2022.

- [`VideoMAE v2`] Wang L, et al. [Videomae v2: Scaling video masked autoencoders with dual masking](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html). CVPR. 2023.

- [`V-JEPA2`] Assran M, et al. [V-jepa 2: Self-supervised video models enable understanding, prediction and planning](https://arxiv.org/abs/2506.09985). arXiv. 2025.

- [`DINO-world`] Baldassarre F, et al. [Back to the Features: DINO as a Foundation for Video World Models](https://arxiv.org/abs/2507.19468). arXiv. 2025.
