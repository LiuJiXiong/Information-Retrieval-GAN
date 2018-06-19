### Implement IRAGN with manual gradients' derivation.
Derive the details of the algorithm [IRGAN](http://delivery.acm.org/10.1145/3090000/3080786/p515-wang.pdf?ip=116.7.245.182&id=3080786&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2E5FBA890B628FA01E%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1529417675_0857de03d85272bf5544ae082151d644) and implement it using manual gradient's derivation. And final performances show that details that we derivate are same as paper's author.
1. IRGAN-multiprocessing：Using multi process to get higher speed. Meanwhile, there is a pre-train method of generative model as paper's author does.
2. IRGAN-multiprocessing-copy: Repair the bug while updating the generative model with some repeated items in sampling.
