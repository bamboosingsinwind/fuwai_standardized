## Setup
1. Linux Operating System
2. NVIDIA GTX4090 GPU
3. Python 3.8 and PyTorch 1.11.0
## Data
1. MAC 5500 or FX-8322 ECG machine in Fuwai Hospital (Beijing); and Bene Heart R12 ECG machine in Yunnan Fuwai Hospital
2. Extracted from XML-formatted data, the ECG signal comprises 12 leads with a sampling rate of 500 Hz over a duration of 10 seconds, resulting in a time series of 5000 data points across the 12 channels. 
3. Denoising: Low-pass, high-pass and median filters.
4. Standardization: z-score normalization.
## Model
We developed a deep neural network to rule-in and rule-out suspected CAD patients for further assessment based on ECGs, which mainly consisted of a feature extraction module, a feature fusion module, and a prediction module.
1. Feature extraction module
    ##### 1.1. 1D optimized ResNet (local features of the waveform)
    ##### 1.2. Transformer's Encoder (long-range dependencies such as interval features on ECGs)
2. Feature fusion module (a gating mechanism for weighted fusion)
3. Prediction module (FC + Softmax)

## Key Implementation
Several crucial parameters require optimization tailored to the specific ECG data.

1. ECG Signal Preprocessing: Fine-tuning the filtering threshold is essential.
2. Branch Optimization: Adjusting parameters within each branch to achieve optimal outcomes is pivotal. Subsequently, considering feature fusion becomes imperative. During branch optimization, the choice of optimizer and learning rate holds significance. Particularly, if the Transformer branch fails to optimize with Adam, experimenting with Adagrad could be beneficial.
3. Enhancement Strategies: Any performance enhancements in the ResNet branch could involve integrating the SE module.
4. Transformer Branch Adaptation: Adapting convolutional embeddings within the Transformer branch is necessary to accommodate tokens of varying scales and hardware capabilities.
5. Fusion Mechanism Selection: The fusion mechanism could entail a gating mechanism, direct concatenation, or an attention mechanism. The choice depends on the specific characteristics of the data under consideration.
