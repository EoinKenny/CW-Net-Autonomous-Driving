This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to reproduce the results published by the author on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- MATLAB/MOSEK/Stata licenses where applicable

# Instructions

## The computational environment (Docker image)

This capsule is private and its environment cannot be downloaded at this time. You will need to rebuild the environment locally.

> If there's any software requiring a license that needs to be run during the build stage, you'll need to make your license available. See [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information.

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag 0ba13245-9965-4842-8550-704e3b77aff6; cd ..
```

> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

## Running the capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --workdir /code \
  --volume "$PWD/data":/data \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  0ba13245-9965-4842-8550-704e3b77aff6 bash run
```
