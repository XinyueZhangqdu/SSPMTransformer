# SSPMTransformer
Semi-supervised portrait matting using transformer

## News

December 3, 2022: codes are released 🔥  

November 24, 2022: Paper accepted  🎉  

## Abstract

Transformer is successfully applied in many tasks but is not sensible to directly embed in the portrait
matting. Such an operation can effectively consider global features ignoring local features, which leads
to the difficulty of identifying the complete and edge-refined portraits. What is more, the existing
supervised architectures are feeble against the unlabeled images. To achieve the effective excavation
of both local and global features, we design a semi-supervised network that leverages Transformer to
capture the global features. The arduous task of compensating more local features is left to the portrait
detailed decoding module (PDDM) that we designed. In addition, to provide the possibility of improving
effectiveness when faced with unlabeled images, we design an intelligent pseudo-label generation
strategy to embed in our semi-supervised network. This strategy can generate more detailed pseudolabels than predicted results through redundant foreground filtering and edge adjustment. Compared to
existing portrait matting methods, our network successfully achieves performance improvements with a
small number of datasets and has the ability to train on unlabeled datasets.
