# Welcome to TruLens!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

TruLens provides a set of tools for developing and monitoring neural nets, including large language models. This includes both tools for evaluation of LLMs and LLM-based applications with TruLens-Eval and deep learning explainability with TruLens-Explain. TruLens-Eval and TruLens-Explain are housed in separate packages and can be used independently.

**TruLens-Eval** contains instrumentation and evaluation tools for large language model (LLM) based applications. It supports the iterative development and monitoring of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine. Importantly, it also gives you the tools you need to evaluate the quality of your LLM-based applications.

For more information, see [TruLens-Eval Documentation](trulens_eval/README.md).

**TruLens-Explain** is a cross-framework library for deep learning explainability. It provides a uniform abstraction over a number of different frameworks. It provides a uniform abstraction layer over TensorFlow, Pytorch, and Keras and allows input and internal explanations.

For more information, see [TruLens-Explain Documentation](trulens_explain/README.md).
