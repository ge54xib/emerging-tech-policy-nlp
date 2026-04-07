---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'QuantERA is a leading European network of 41 public research funding organizations
    from 31 countries with a mission to support excellence in QT research and innovation.
    SI has recognized the scientific and technological potential of this field and
    joined the QuantERA network in 2017. The QuantERA network has received co-funding
    from the Horizon 2020 program twice, first in 2016 for the QuantERA I ERA-NET
    Cofund partnership and second in 2020 for the QuantERA II

    ERA-NET Cofund partnership.'
- text: 'The Max Planck Society (MPG) began funding basic research on quantum science
    and technology (QST) at an early stage in the spirit of the second quantum revolution.
    In 1981, the Max Planck Institute (MPI) for Quantum Optics was founded in Garching,
    in 1993 the experimental branch of the MPI for Gravitational Physics was set up
    in Hanover, in 1994 it was the turn of the MPI for the Physics of Complex Systems
    in Dresden, and in 2009 the MPI for the Physics of Light in Erlangen. In addition,
    the Max Planck Institutes for Solid State Research in Stuttgart, for Computer
    Science in Saarbrücken, for Nuclear Physics in Heidelberg, for Multidisciplinary
    Natural Sciences in Göttin-gen, for Chemical Solid-state Physics in Dresden, for
    the Structure and Dynamics of Matter in Hamburg, for Microstructure Physics in
    Halle, for Mathematics in Science in Leipzig, for Intelligent Systems in Tübin-gen,
    for Security and Privacy in Bochum as well as the Fritz Haber Institute in Berlin
    also work in part on

    aspects of this topic.'
- text: 'This practice can be sped up significantly by using software to model error
    correction, represent modern algorithmic techniques, and compile into low‑level
    instruction sets. QSI researchers are developing the Bench‑Q software suite for
    this purpose with the University of Southern California, University of Texas at
    Dallas, Aalto University in Finland, and Zapata Computing. The quantum benchmarking
    program is using this software to estimate and optimise the cost of quantum algorithms
    in the processor platforms of Rigetti Computing and IonQ.

    Quantum computing can bring molecular modelling to a new level of accuracy and
    simulating reactions could yield next generation batteries and pharmaceuticals.'
- text: '- • The NQCO should work with Agencies and the broader QIST ecosystem to
    amplify public outreach activities and incorporate clear and realistic descriptions
    of QIST advances, challenges, and opportunities. - • Government sponsored efforts
    that include workforce development activities, such as DOE''s National Quantum
    Information Science Research Centers and NSF’s Quantum Leap Challenge Institutes,
    should strive to create a positive and accurate branding of QIST. They should
    focus attention on realistic possibilities, and highlight ongoing efforts to create
    an environment that encourages, welcomes, and inspires involvement by everyone
    who might wish to participate.'
- text: There are few new start-ups or university or academic spin-offs in the Czech
    Republic that focus on the application of quantum technologies. There are investment
    funds that focus, among other things, on investments in quantum technologies,
    such as Tensor Ventures and Presto Ventures. The first companies are also emerging
    that are attempting to use and create quantum algorithms for their own use.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/all-mpnet-base-v2
---

# SetFit with sentence-transformers/all-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 384 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label            | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:-----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| knowledge_space  | <ul><li>'The Max Planck Society (MPG) was an early supporter of basic research on quantum technologies, in the sense of the second quantum revolution. The Max Planck Institute (MPI) of Quantum Optics in Garching was founded in 1981.'</li><li>'This practice can be sped up significantly by using software to model error correction, represent modern algorithmic techniques, and compile into low‑level instruction sets. QSI researchers are developing the Bench‑Q software suite for this purpose with the University of Southern California, University of Texas at Dallas, Aalto University in Finland, and Zapata Computing. The quantum benchmarking program is using this software to estimate and optimise the cost of quantum algorithms in the processor platforms of Rigetti Computing and IonQ.\nQuantum computing can bring molecular modelling to a new level of accuracy and simulating reactions could yield next generation batteries and pharmaceuticals.'</li><li>'Various partnerships ensure that QuSoft has close relations with the rest of the quantum landscape. Within the Quantum Software Consortium, for example, QuSoft collaborates with QuTech and Leiden University on the development of quantum software and applications. Individual QuSoft researchers also collaborate closely with colleagues at other Dutch institutions.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| innovation_space | <ul><li>'Small Business Research Initiative\nNational Security Strategic Investment Fund\nGovernment can act as an early adopter of emerging technologies to support technology development and demonstrate the value of technologies to other sectors of the economy. There are numerous examples where procurement has been used to drive innovation such as through the (SBRI) and the (NSSIF), providing public service improvements for the government whilst supporting small businesses and stimulating markets. As part of NSSIF’s strategy, and to deliver insight and access to quantum computing for government, NSSIF has deployed £2.6m in R&D contracts with quantum companies, helping to stimulate growth in the UK quantum ecosystem.'</li><li>'To mobilize such private financial resources to a greater extent for venture funds operating in deep tech, it therefore appears essential to further strengthen instruments and incentives that operate as leverage (such as, for example, tax breaks) or introduce forms of de-risking. In order to limit the level of risk and losses on the portfolio, one hypothesis could be to introduce "guarantees" of a public nature (e.g., through SACE), i.e., insurance instruments to guarantee part of the qualified investment, by an institutional fund (Limited Partner), in units or shares of deep-tech Venture Capital Funds residing in the territory of the state (or that of member states of the European Union or states party to the Agreement on the European Economic Area, provided that the manager\'s investment is in startups operating mainly in Italy). A plafond of public guarantees worth 100 million € (covering 20% of each individual VC fund investment) would correspond to significant leverage.'</li><li>'There are few new start-ups or university or academic spin-offs in the Czech Republic that focus on the application of quantum technologies. There are investment funds that focus, among other things, on investments in quantum technologies, such as Tensor Ventures and Presto Ventures. The first companies are also emerging that are attempting to use and create quantum algorithms for their own use.'</li></ul> |
| consensus_space  | <ul><li>'Talent will be supported through programming at NSERC and Mitacs:\n- • ($5.4 million over six years)\nNSERC CREATE grants program\n- • ($40 million over six years)\nMitacs'</li><li>'Promotion campaigns in the sector, marketing of expertise and opportunities nationally and internationally. Responsible parties: InstituteQ\nOther actors: MEC, Business Finland, Technology Industries of Finland, Team Finland network\nEach year, one national (Finnish/ Swedish) and one international awareness campaign targeted at companies, another at young adults and students, the third for UAS and general upper secondary school teachers. Visibility and participation in campaigns of other critical technologies, for example by utilising forums of Technology Industries of Finland.'</li><li>"SI's participation in EuroHPC JU is ensured by the Ministry of Education, Science and Sport. In 2023, SI joined the European Quantum Declaration, the so-called quantum pact, which represents a commitment to cooperate with other Member States and the EC in developing a cutting-edge QT ecosystem. The declaration highlights objectives such as coordinating research and development programs, promoting public and private investment, supporting businesses, strengthening the European supply chain, and developing a network of quantum competence centers."</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| public_space     | <ul><li>'- − Organization of events as part of Science Month. Status\nIN PREPARATION/BEING IMPLEMENTED\nExpected results after implementation of the measure\n- − Awareness campaign: 1 (UNESCO)\n- − Events held: 1 (Science Month)\nMeasure implementer\nMVZI (State Agency for UNESCO)'</li><li>'Women make up just over a quarter of the STEM workforce,24 (<>)making it challenging to achieve a diverse future workforce that is representative of society. Centres for Doctoral Training\nDefence Science and Technology Laboratory\nNational Physical Laboratory\nPeople and know-how are driving the early development of the emerging sector. We have provided strong support for growing the highly-specialised skills needed in the quantum sector since 2014.'</li><li>"- • The NQCO should work with Agencies and the broader QIST ecosystem to amplify public outreach activities and incorporate clear and realistic descriptions of QIST advances, challenges, and opportunities. - • Government sponsored efforts that include workforce development activities, such as DOE's National Quantum Information Science Research Centers and NSF’s Quantum Leap Challenge Institutes, should strive to create a positive and accurate branding of QIST. They should focus attention on realistic possibilities, and highlight ongoing efforts to create an environment that encourages, welcomes, and inspires involvement by everyone who might wish to participate."</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("There are few new start-ups or university or academic spin-offs in the Czech Republic that focus on the application of quantum technologies. There are investment funds that focus, among other things, on investments in quantum technologies, such as Tensor Ventures and Presto Ventures. The first companies are also emerging that are attempting to use and create quantum algorithms for their own use.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 7   | 91.35  | 390 |

| Label            | Training Sample Count |
|:-----------------|:----------------------|
| knowledge_space  | 15                    |
| innovation_space | 15                    |
| consensus_space  | 15                    |
| public_space     | 15                    |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0059 | 1    | 0.2235        | -               |
| 0.2959 | 50   | 0.1635        | -               |
| 0.5917 | 100  | 0.0262        | -               |
| 0.8876 | 150  | 0.0022        | -               |

### Framework Versions
- Python: 3.11.10
- SetFit: 1.1.3
- Sentence Transformers: 5.3.0
- Transformers: 4.57.6
- PyTorch: 2.6.0+cu124
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->