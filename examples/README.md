# GLM Examples
This is a directory that collects GLM's implementation over various NLP datasets. 
We feel lucky to collaborate with all contributors that share their implementations here.

## Make You Pull Requests (PRs)
If you also want to become a contributor of GLM, we encourage you to make PR to this repo according to the following PR rules.
The maintainer will check the validity before accept the PR.

### Directory Structure
Each PR should include the code and markdown description in a subdirectory of the current `examples` directory.
An example subdirectory tree is as follows:

```
└── examples
    └── <Your PR directory>: Huggingface Datasets identifier (recommended) or customized name
        ├── README.md
        ├── requirements.txt
        └── <Your code>
```

Please exclude data files in the PR as they take up too much space, and describe the method to acquire the data in your `README.md`.

### Task Description (README.md)
Please at least include the following sections in your README to help its better use:

+ **Dataset Name**: serves as the markdown title.
+ **Authors**: Your name(s), contacts (email), and the url to your homepage(s) (if available).
+ **Task Description**: A short paragraph to briefly introduce what the dataset and corresponding task is about.
+ **Running Commands**: Provide the bash/shell commands for preprocessing, training, and inference.
+ **Results**: Please provide your implementation's final performance, along with other available comparison methods'. Ensure that they are reproducible once using your provided `Running Commands`.
+ **Reference**: Proper citation information for the dataset and related comparison methods.

### Environment Requirements (requirements.txt)
Please include the necessary python packages in the file for other users to reproduce your results.

## Example List (To Be Updated)
TODO
