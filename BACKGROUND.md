### Motivation

The human brain is the most complex system we know of. Accurately diagnosing and treating its range of dysfunctions is largely beyond the limit of modern medicine. Functional neuroimaging data such as fMRI and EEG provide a possible window into better understanding how the brain works, why it sometimes breaks down, and what to do about it. After decades of effort, however, it feels as though we are still far from applying these data to real clinical purpose. One challenge is that the data are a mess. They are extremely noisy, blurry, and impossible to visually interpret. But on the positive side, we have collected a *lot* of it. In many other data domains, we have seen that by combining large amounts of data together with flexible neural network models and large-scale compute, one can "unlock" the signal in complex noisy data. Our goal is to answer a simple question: can we do the same with functional neuroimaging?

### Related work

Building clinically predictive models using functional neuroimaging data has of course been a longstanding goal[^1] [^2]. The current dominant approach combines expert-crafted features, for example resting-state functional connectivity matrices, together with classic shallow machine learning models. See for example[^3], and many other references cited within and since. Unlike other domains like images and text, brain data do not seem to immediately yield to deep learning based approaches[^4] [^5]. Although there are some success cases of shallow nonlinear networks applied to functional brain imaging data, e.g. [^6] [^7] [^8].

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
