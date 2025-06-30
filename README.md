# Diffusion-based Adversarial Purification for Intrusion Detection
*Mohamed Amine Merzouk<sup>1,2,3</sup>;
Erwan Beurier<sup>2,3</sup>;
Reda Yaich<sup>3</sup>;
Nora Boulahia-Cuppens<sup>2</sup>;
Frédéric Cuppens<sup>2</sup>;
Foutse Khomh<sup>1,2</sup>*

<sup>1</sup>Mila - Québec AI Institute
<sup>2</sup>Polytechnique Montréal
<sup>3</sup>IRT SystemX

Paper published in the [Proceedings of the 39th Conference on Data and Applications Security and Privacy (DBSec 2025)](https://link.springer.com/chapter/10.1007/978-3-031-96590-6_19).

## Overview
This repository contains the implementation of our research on utilizing diffusion models for adversarial purification in intrusion detection systems. With the increasing sophistication of cyberattacks, conventional machine learning (ML) techniques in intrusion detection have faced significant challenges due to adversarial examples that can mislead models, causing undetected attacks or false alerts. Our work leverages diffusion models to effectively purify adversarial examples, enhancing intrusion detection robustness.

## Background
Adversarial purification is a defense mechanism aimed at removing perturbations from adversarial examples to ensure accurate classification. In this study, we investigate the application of diffusion models, which are generative models inspired by physical diffusion processes. These models gradually add noise to data in a forward process and reconstruct it using a reverse process, effectively mitigating adversarial attacks.

## Key Contributions
- Demonstrated the effectiveness of diffusion models in purifying adversarial examples specifically within the context of network intrusion detection.
- Identified optimal diffusion parameters that maximize adversarial robustness while minimizing impact on normal performance.
- Provided insights into the relationship between diffusion noise and diffusion steps, establishing new benchmarks in the field.

## Datasets
Our experiments are conducted using two widely recognized datasets:
- **NSL-KDD**: A traditional dataset in the cybersecurity domain used for benchmarking.
- **UNSW-NB15**: A contemporary dataset that represents more modern network traffic.

## Adversarial Attacks Evaluated
The performance of our diffusion-based purification method was tested against five prevalent adversarial attack methods:
1. DeepFool
2. Jacobian-based Saliency Map Attack (JSMA)
3. Fast Gradient Sign Method (FGSM)
4. Basic Iterative Method (BIM)
5. Carlini & Wagner's L2 attack.

## Acknowledgments
This work was supported by [Mitacs](https://www.mitacs.ca/) through the Mitacs Accelerate International program and the [CRITiCAL](https://www.critical.polymtl.ca/) chair.
It was enabled in part by support provided by [Calcul Québec](https://calculquebec.ca/), [Compute Ontario](https://computeontario.ca/), the BC DRI Group, and the [Digital Research Alliance of Canada](https://alliancecan.ca/).
This work has also received support from the French government through the "France 2030" program, as part of the IRT SystemX Program CYBELIA.

## License
This project is licensed under the MIT License.
